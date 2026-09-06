import argparse
from dataclasses import dataclass, field
import itertools
import multiprocessing
from multiprocessing import Queue
import os
import queue
import random
import re
import sys
import threading
import time

from typing import (
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
)

from .candidate import CandidateResult
from .compiler import Compiler
from .error import CandidateConstructionFailure
from .helpers import (
    get_settings,
    get_default_randomization_weights,
    json_prop,
    json_dict,
    merge_randomization_weights,
    plural,
    static_assert_unreachable,
    trim_source,
)

from .permuter import (
    EvalError,
    EvalResult,
    Feedback,
    Finished,
    Message,
    NeedMoreWork,
    Permuter,
    Reseed,
    Task,
    WorkDone,
)
from .preprocess import preprocess
from .printer import Printer
from .profiler import Profiler
from .randomizer import RANDOMIZATION_PASSES
from .scorer import Scorer

MIN_PRIO = 0.01
MAX_PRIO = 2.0

# The probability that the randomizer continues transforming the output it
# generated last time.
DEFAULT_RAND_KEEP_PROB = 0.6


@dataclass
class Options:
    directories: List[str]
    show_errors: bool = False
    show_timings: bool = False
    print_diffs: bool = False
    stack_differences: bool = False
    algorithm: str = "difflib"
    abort_exceptions: bool = False
    better_only: bool = False
    best_only: bool = False
    score_threshold: Optional[int] = None
    quiet: bool = False
    stop_on_zero: bool = False
    keep_prob: float = DEFAULT_RAND_KEEP_PROB
    force_seed: Optional[str] = None
    threads: int = 1
    use_network: bool = False
    network_debug: bool = False
    network_priority: float = 1.0
    no_context_output: bool = False
    debug_mode: bool = False
    speed: int = 100
    ign_branch_targets: bool = False
    # Opt-in evolutionary/population mode (see the module-level docstring
    # near run_inner's generation-dispatch code below for the full design).
    # 0 (default) fully disables it and preserves pre-existing behavior.
    population_size: int = 0
    generation_interval: float = 60.0
    population_fresh_fraction: float = 0.5


def restricted_float(lo: float, hi: float) -> Callable[[str], float]:
    def convert(x: str) -> float:
        try:
            ret = float(x)
        except ValueError:
            raise argparse.ArgumentTypeError(f"invalid float value: '{x}'")

        if ret < lo or ret > hi:
            raise argparse.ArgumentTypeError(
                f"value {x} is out of range (must be between {lo} and {hi})"
            )
        return ret

    return convert


# --- Population/evolutionary mode -------------------------------------
#
# Kero-local addition, opt-in via --population-size (0 = off, the default,
# preserving exact upstream behavior).
#
# Background: each permuter worker (one OS process per -j slot) runs ONE
# CONTINUOUS random walk via --keep-prob: each iteration has `keep_prob`
# probability of mutating from its own last-accepted candidate rather than
# resetting to base.c. A single long-running worker can wander into an
# unproductive corner of the search space and stay stuck there indefinitely
# -- restarting (fresh OS-entropy seed, back to base.c) was previously the
# ONLY way to escape a rut, with no way to branch a restart off a good
# candidate already found instead of either continuing the stuck trajectory
# or throwing away all progress back to base.c.
#
# This adds a population of the top `--population-size` candidates found so
# far for each permuter (seeded with base.c itself), tracked centrally in
# the coordinator process (populated in post_score, below, at the same
# point candidates are written to output-N-M/ -- the ONE place the
# coordinator already sees every accepted candidate's real score and full
# source). Every `--generation-interval` seconds (a "generation"), the
# coordinator directly enqueues one Reseed control message (see
# permuter.py) per currently-alive worker onto the shared task queue --
# `--population-fresh-fraction` of them reset a worker to base.c (for
# diversity), the rest branch it onto a random population member (to
# compound progress instead of either always getting stuck in one rut or
# always throwing away everything). Reseed just re-points a worker's own
# Permuter.reseed_from_source (its _permutations/_cur_cand/_last_score) --
# base_score/base_hash/hashes, the ground truth used for scoring and dedup,
# are never touched, so scores stay directly comparable across the whole
# run regardless of what generation a candidate came from.
#
# This only makes sense in the multiprocess/local-worker path (multiple
# concurrent workers pulling from one shared queue to branch across); it is
# a no-op with a one-time warning in the single-threaded and pure -J
# (permuter@home network-only) paths.
# ------------------------------------------------------------------------


@dataclass
class EvalContext:
    options: Options
    printer: Printer = field(default_factory=Printer)
    iteration: int = 0
    errors: int = 0
    internal_errors: int = 0
    internal_error_stack_traces: Set[str] = field(default_factory=set)
    overall_profiler: Profiler = field(default_factory=Profiler)
    permuters: List[Permuter] = field(default_factory=list)
    # perm_index -> list of (score, source), sorted ascending by score,
    # capped at options.population_size. Only populated/consulted when
    # options.population_size > 0.
    populations: Dict[int, List[Tuple[int, str]]] = field(default_factory=dict)
    generations_run: int = 0


def write_candidate(
    perm: Permuter, result: CandidateResult, no_context_output: bool
) -> None:
    """Write the candidate's C source and score to the next output directory"""
    ctr = 0
    while True:
        ctr += 1
        try:
            output_dir = os.path.join(perm.dir, f"output-{result.score}-{ctr}")
            os.mkdir(output_dir)
            break
        except FileExistsError:
            pass
    source = result.source

    assert source is not None, "Permuter._need_to_send_source is wrong!"
    with open(os.path.join(output_dir, "source.c"), "x", encoding="utf-8") as f:
        if no_context_output:
            f.write(trim_source(source, perm.fn_name))
        else:
            f.write(source)
    with open(os.path.join(output_dir, "score.txt"), "x", encoding="utf-8") as f:
        f.write(f"{result.score}\n")
    with open(os.path.join(output_dir, "diff.txt"), "x", encoding="utf-8") as f:
        f.write(perm.diff(source) + "\n")
    print(f"wrote to {output_dir}")


def _update_population(
    context: EvalContext, perm_index: int, score: int, source: str
) -> None:
    """Record an accepted candidate into its permuter's elite population
    (see the module docstring above). Deduplicates by exact source text (an
    --best-only run can otherwise re-report the same tie repeatedly) and
    keeps only the top `population_size` by score."""
    cap = context.options.population_size
    pop = context.populations.setdefault(perm_index, [])
    if any(s == source for _, s in pop):
        return
    pop.append((score, source))
    pop.sort(key=lambda item: item[0])
    del pop[cap:]


def post_score(
    context: EvalContext,
    permuter: Permuter,
    result: EvalResult,
    who: Optional[str],
    perm_index: Optional[int] = None,
) -> bool:
    if isinstance(result, EvalError):
        context.internal_errors += 1
        if result.exc_str is not None:
            if result.exc_str in context.internal_error_stack_traces:
                return False
            context.internal_error_stack_traces.add(result.exc_str)
            context.printer.print(
                "internal permuter failure.", permuter, who, keep_progress=True
            )
            print(result.exc_str)
        if result.seed is not None:
            seed_str = str(result.seed[1])
            if result.seed[0] != 0:
                seed_str = f"{result.seed[0]},{seed_str}"
            print(f"To reproduce the failure, rerun with: --seed {seed_str}")
        if context.options.abort_exceptions:
            sys.exit(1)
        else:
            return False

    if context.options.print_diffs:
        assert result.source is not None, "Permuter._need_to_send_source is wrong"
        print()
        print(permuter.diff(result.source))
        input("Press any key to continue...")

    profiler = result.profiler
    score_value = result.score

    if profiler is not None:
        for stattype in profiler.time_stats:
            context.overall_profiler.add_stat(stattype, profiler.time_stats[stattype])

    context.iteration += 1
    if score_value == permuter.scorer.PENALTY_INF:
        disp_score = "inf"
        context.errors += 1
    else:
        disp_score = str(score_value)

    status_line = f"iteration {context.iteration}, {context.errors} errors, "
    if context.internal_errors:
        status_line += f"{context.internal_errors} permuter failures, "
    status_line += f"score = {disp_score}"
    if context.options.show_timings:
        status_line += "  \t" + context.overall_profiler.get_str_stats()

    if permuter.should_output(result):
        former_best = permuter.best_score
        permuter.record_result(result)
        if score_value < former_best:
            color = "\u001b[32;1m"
            msg = f"found new best score! ({score_value} vs {permuter.base_score})"
        elif score_value == former_best:
            color = "\u001b[32;1m"
            msg = f"tied best score! ({score_value} vs {permuter.base_score})"
        elif score_value < permuter.base_score:
            color = "\u001b[33m"
            msg = f"found a better score! ({score_value} vs {permuter.base_score})"
        else:
            color = "\u001b[33m"
            msg = f"found different asm with same score ({score_value})"
        context.printer.print(msg, permuter, who, color=color)
        write_candidate(permuter, result, context.options.no_context_output)
        if (
            context.options.population_size > 0
            and perm_index is not None
            and result.source is not None
        ):
            _update_population(context, perm_index, score_value, result.source)
    if not context.options.quiet:
        context.printer.progress(status_line)
    return score_value == 0


def cycle_seeds(permuters: List[Permuter]) -> Iterable[Tuple[int, int]]:
    """
    Return all possible (permuter index, seed) pairs, cycling over permuters.
    If a permuter is randomized, it will keep repeating seeds infinitely.
    """
    iterators: List[Iterator[Tuple[int, int]]] = []
    for perm_ind, permuter in enumerate(permuters):
        it = permuter.seed_iterator()
        iterators.append(zip(itertools.repeat(perm_ind), it))

    i = 0
    while iterators:
        i %= len(iterators)
        item = next(iterators[i], None)
        if item is None:
            del iterators[i]
            i -= 1
        else:
            yield item
            i += 1


def multiprocess_worker(
    permuters: List[Permuter],
    input_queue: "Queue[Task]",
    output_queue: "Queue[Feedback]",
) -> None:
    try:
        while True:
            # Read a work item from the queue. If none is immediately available,
            # tell the main thread to fill the queues more, and then block on
            # the queue.
            queue_item: Task
            try:
                queue_item = input_queue.get(block=False)
            except queue.Empty:
                output_queue.put((NeedMoreWork(), -1, None))
                queue_item = input_queue.get()
            if isinstance(queue_item, Finished):
                output_queue.put((queue_item, -1, None))
                output_queue.close()
                break
            if isinstance(queue_item, Reseed):
                # Population-mode control message (see main.py's module
                # docstring): just re-point this worker's own trajectory and
                # loop back around for real work. Produces no WorkDone/output
                # of its own, so it needs no NeedMoreWork bookkeeping beyond
                # what the top of the loop already does on its next pass.
                permuters[queue_item.perm_index].reseed_from_source(queue_item.source)
                continue
            permuter_index, seed = queue_item
            permuter = permuters[permuter_index]

            start = time.time()

            result = permuter.try_eval_candidate(seed)
            if isinstance(result, CandidateResult) and permuter.should_output(result):
                permuter.record_result(result)

            if permuter.speed != 100:
                end = time.time()

                sleep_time = (end - start) * ((100 / permuter.speed) - 1)
                time.sleep(sleep_time)

            output_queue.put((WorkDone(permuter_index, result), -1, None))
            output_queue.put((NeedMoreWork(), -1, None))
    except KeyboardInterrupt:
        # Don't clutter the output with stack traces; Ctrl+C is the expected
        # way to quit and sends KeyboardInterrupt to all processes.
        # A heartbeat thing here would be good but is too complex.
        # Don't join the queue background thread -- thread joins in relation
        # to KeyboardInterrupt usually result in deadlocks.
        input_queue.cancel_join_thread()
        output_queue.cancel_join_thread()


def _maybe_dispatch_generation(
    context: EvalContext,
    options: Options,
    worker_task_queue: "Queue[Task]",
    active_workers: int,
    last_generation_time: float,
) -> float:
    """If --generation-interval seconds have passed, start a new
    population-mode generation: enqueue one Reseed control message per
    currently-alive local worker, split between resetting to base.c (for
    diversity) and branching onto a random member of that permuter's elite
    population (to compound progress) per --population-fresh-fraction. A
    no-op (returns `last_generation_time` unchanged) when population mode is
    off, there are no local workers, or not enough time has passed yet."""
    if options.population_size <= 0 or active_workers <= 0:
        return last_generation_time
    now = time.time()
    if now - last_generation_time < options.generation_interval:
        return last_generation_time

    candidates = [perm_index for perm_index, pop in context.populations.items() if pop]
    if not candidates:
        return now

    context.generations_run += 1
    from_pop = 0
    from_base = 0
    for _ in range(active_workers):
        perm_index = random.choice(candidates)
        if random.uniform(0, 1) < options.population_fresh_fraction:
            worker_task_queue.put(Reseed(perm_index, None))
            from_base += 1
        else:
            pop = context.populations[perm_index]
            _score, source = random.choice(pop)
            worker_task_queue.put(Reseed(perm_index, source))
            from_pop += 1

    pop_sizes = {i: len(p) for i, p in context.populations.items()}
    print(
        f"\n[generation {context.generations_run}] reseeded {active_workers} "
        f"worker(s): {from_base} -> base.c, {from_pop} -> population member "
        f"(population sizes: {pop_sizes})",
        flush=True,
    )
    return now


def run(options: Options) -> List[int]:
    last_time = time.time()
    try:

        def heartbeat() -> None:
            nonlocal last_time
            last_time = time.time()

        return run_inner(options, heartbeat)
    except KeyboardInterrupt:
        if time.time() - last_time > 5:
            print()
            print("Aborting stuck process.")
            raise
        print()
        print("Exiting.")
        sys.exit(0)


def run_inner(options: Options, heartbeat: Callable[[], None]) -> List[int]:
    print("Loading...")

    context = EvalContext(options)

    force_seed: Optional[int] = None
    force_rng_seed: Optional[int] = None
    if options.force_seed:
        seed_parts = list(map(int, options.force_seed.split(",")))
        force_rng_seed = seed_parts[-1]
        force_seed = 0 if len(seed_parts) == 1 else seed_parts[0]

    name_counts: Dict[str, int] = {}
    for d in options.directories:
        heartbeat()
        compile_cmd = os.path.join(d, "compile.sh")
        target_o = os.path.join(d, "target.o")
        base_c = os.path.join(d, "base.c")
        for fname in [compile_cmd, target_o, base_c]:
            if not os.path.isfile(fname):
                print(f"Missing file {fname}", file=sys.stderr)
                sys.exit(1)

        settings: Mapping[str, object] = get_settings(d)

        compiler_type = json_prop(settings, "compiler_type", str, "base")

        default_weights = get_default_randomization_weights(compiler_type)
        weight_overrides = json_dict(
            json_prop(settings, "weight_overrides", dict, {}), float
        )
        randomization_weights = merge_randomization_weights(
            default_weights, weight_overrides
        )

        fn_name: Optional[str] = None
        if "func_name" in settings:
            fn_name = json_prop(settings, "func_name", str)

        if not fn_name:
            try:
                with open(os.path.join(d, "function.txt"), encoding="utf-8") as f:
                    fn_name = f.read().strip()
            except FileNotFoundError:
                pass

        if fn_name:
            print(f"{base_c} ({fn_name})")
        else:
            print(base_c)

        compiler = Compiler(
            compile_cmd, show_errors=options.show_errors, debug_mode=options.debug_mode
        )

        objdump_command = json_prop(settings, "objdump_command", str, "") or None

        scorer = Scorer(
            target_o,
            stack_differences=options.stack_differences,
            algorithm=options.algorithm,
            debug_mode=options.debug_mode,
            ign_branch_targets=options.ign_branch_targets,
            objdump_command=objdump_command,
        )
        c_source = preprocess(base_c)

        try:
            permuter = Permuter(
                d,
                fn_name,
                compiler,
                scorer,
                base_c,
                c_source,
                randomization_weights=randomization_weights,
                force_seed=force_seed,
                force_rng_seed=force_rng_seed,
                keep_prob=options.keep_prob,
                need_profiler=options.show_timings,
                need_all_sources=options.print_diffs,
                show_errors=options.show_errors,
                best_only=options.best_only,
                better_only=options.better_only,
                score_threshold=options.score_threshold,
                debug_mode=options.debug_mode,
                speed=options.speed,
            )
        except CandidateConstructionFailure as e:
            print(e.message, file=sys.stderr)
            sys.exit(1)

        context.permuters.append(permuter)
        name_counts[permuter.fn_name] = name_counts.get(permuter.fn_name, 0) + 1
    print()

    if not context.permuters:
        print("No permuters!")
        return []

    for perm_index, permuter in enumerate(context.permuters):
        if name_counts[permuter.fn_name] > 1:
            permuter.unique_name += f" ({permuter.dir})"
        print(f"[{permuter.unique_name}] base score = {permuter.best_score}")
        if options.population_size > 0:
            # Seed each permuter's elite population with its own base
            # candidate, so the very first generation already has something
            # to (optionally) branch off of.
            context.populations[perm_index] = [
                (permuter.best_score, permuter.base_source)
            ]

    if options.population_size > 0 and (options.threads == 1 and not options.use_network):
        print(
            "note: --population-size has no effect in single-threaded mode "
            "(no other workers to branch across) -- pass -j N to use it."
        )

    if options.debug_mode:
        print("End of Debug Mode... Exiting")
        sys.exit(0)

    found_zero = False
    if options.threads == 1 and not options.use_network:
        # Simple single-threaded mode. This is not technically needed, but
        # makes the permuter easier to debug.
        for permuter_index, seed in cycle_seeds(context.permuters):
            heartbeat()
            permuter = context.permuters[permuter_index]

            start = time.time()

            result = permuter.try_eval_candidate(seed)
            if post_score(context, permuter, result, None, permuter_index):
                found_zero = True
                if options.stop_on_zero:
                    break

            if permuter.speed != 100:
                end = time.time()

                sleep_time = (end - start) * ((100 / permuter.speed) - 1)
                time.sleep(sleep_time)
    else:
        seed_iterators: List[Optional[Iterator[int]]] = [
            permuter.seed_iterator() for permuter in context.permuters
        ]
        seed_iterators_remaining = len(seed_iterators)
        next_iterator_index = 0

        # Create queues.
        worker_task_queue: "Queue[Task]" = Queue()
        feedback_queue: "Queue[Feedback]" = Queue()

        # Connect to network and create client threads and queues.
        net_conns: "List[Tuple[threading.Thread, Queue[Task]]]" = []
        if options.use_network:
            # Importing the networking modules here so dependencies don't need to be loaded when not using the network mode
            from .net.client import start_client
            from .net.core import ServerError, connect, enable_debug_mode

            print("Connecting to permuter@home...")
            if options.network_debug:
                enable_debug_mode()
            first_stats: Optional[Tuple[int, int, float]] = None
            for perm_index in range(len(context.permuters)):
                try:
                    port = connect()
                except (EOFError, ServerError) as e:
                    print("Error:", e)
                    sys.exit(1)
                thread, queue, stats = start_client(
                    port,
                    context.permuters[perm_index],
                    perm_index,
                    feedback_queue,
                    options.network_priority,
                )
                net_conns.append((thread, queue))
                if first_stats is None:
                    first_stats = stats
            assert first_stats is not None, "has at least one permuter"
            clients_str = plural(first_stats[0], "other client")
            servers_str = plural(first_stats[1], "server")
            cores_str = plural(int(first_stats[2]), "core")
            print(f"Connected! {servers_str} online ({cores_str}, {clients_str})")

        # Start local worker threads
        processes: List[multiprocessing.Process] = []
        for _i in range(options.threads):
            p = multiprocessing.Process(
                target=multiprocess_worker,
                args=(context.permuters, worker_task_queue, feedback_queue),
            )
            p.start()
            processes.append(p)

        active_workers = len(processes)

        if not active_workers and not net_conns:
            print("No workers available! Exiting.")
            sys.exit(1)

        def process_finish(finish: Finished, source: int) -> None:
            nonlocal active_workers

            if finish.reason:
                permuter: Optional[Permuter] = None
                if source != -1 and len(context.permuters) > 1:
                    permuter = context.permuters[source]
                context.printer.print(finish.reason, permuter, None, keep_progress=True)

            if source == -1:
                active_workers -= 1

        def process_result(work: WorkDone, who: Optional[str]) -> bool:
            permuter = context.permuters[work.perm_index]
            return post_score(context, permuter, work.result, who, work.perm_index)

        def get_task(perm_index: int) -> Optional[Tuple[int, int]]:
            nonlocal next_iterator_index, seed_iterators_remaining
            if perm_index == -1:
                while seed_iterators_remaining > 0:
                    task = get_task(next_iterator_index)
                    next_iterator_index += 1
                    next_iterator_index %= len(seed_iterators)
                    if task is not None:
                        return task
            else:
                it = seed_iterators[perm_index]
                if it is not None:
                    seed = next(it, None)
                    if seed is None:
                        seed_iterators[perm_index] = None
                        seed_iterators_remaining -= 1
                    else:
                        return (perm_index, seed)
            return None

        # Feed the task queue with work and read from results queue.
        # We generally match these up one-by-one to avoid overfilling queues,
        # but workers can ask us to add more tasks into the system if they run
        # out of work. (This will happen e.g. at the very beginning, when the
        # queues are empty.)
        last_generation_time = time.time()
        while seed_iterators_remaining > 0:
            heartbeat()
            last_generation_time = _maybe_dispatch_generation(
                context, options, worker_task_queue, active_workers, last_generation_time
            )
            feedback, source, who = feedback_queue.get()
            if isinstance(feedback, Finished):
                process_finish(feedback, source)
            elif isinstance(feedback, Message):
                context.printer.print(feedback.text, None, who, keep_progress=True)
            elif isinstance(feedback, WorkDone):
                if process_result(feedback, who):
                    # Found score 0!
                    found_zero = True
                    if options.stop_on_zero:
                        break
            elif isinstance(feedback, NeedMoreWork):
                task = get_task(source)
                if task is not None:
                    if source == -1:
                        worker_task_queue.put(task)
                    else:
                        net_conns[source][1].put(task)
            else:
                static_assert_unreachable(feedback)

        # Signal workers to stop.
        for _i in range(active_workers):
            worker_task_queue.put(Finished())

        for conn in net_conns:
            conn[1].put(Finished())

        # Await final results.
        while active_workers > 0 or net_conns:
            heartbeat()
            feedback, source, who = feedback_queue.get()
            if isinstance(feedback, Finished):
                process_finish(feedback, source)
            elif isinstance(feedback, Message):
                context.printer.print(feedback.text, None, who, keep_progress=True)
            elif isinstance(feedback, WorkDone):
                if not (options.stop_on_zero and found_zero):
                    if process_result(feedback, who):
                        found_zero = True
            elif isinstance(feedback, NeedMoreWork):
                pass
            else:
                static_assert_unreachable(feedback)

        # Wait for workers to finish.
        for p in processes:
            p.join()

        # Wait for network connections to close (currently does not happen).
        for conn in net_conns:
            conn[0].join()

    if found_zero:
        print("\nFound zero score! Exiting.")
    return [permuter.best_score for permuter in context.permuters]


class PrintRandomizationPassesAction(argparse.Action):
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: object,
        option_string: Optional[str] = None,
    ) -> None:
        # TODO include weights from get_default_randomization_weights("base")
        for method in RANDOMIZATION_PASSES:
            print(f"{method.__name__}:")
            docs = (method.__doc__ or "").strip()
            docs = re.sub(r"^(    )?", "  ", docs, flags=re.MULTILINE)
            print(docs)
            print()
        sys.exit(0)


def main() -> None:
    multiprocessing.freeze_support()
    sys.setrecursionlimit(10000)

    # Ideally we would do:
    #  multiprocessing.set_start_method("spawn")
    # here, to make multiprocessing behave the same across operating systems.
    # However, that means that arguments to Process are passed across using
    # pickling, which mysteriously breaks with pycparser...
    # (AttributeError: 'CParser' object has no attribute 'p_abstract_declarator_opt')
    # So, for now we live with the defaults, which make multiprocessing work on Linux,
    # where it uses fork and doesn't pickle arguments, and break on Windows. Sigh.

    parser = argparse.ArgumentParser(
        description="Randomly permute C files to better match a target binary."
    )
    parser.add_argument(
        "directories",
        nargs="+",
        metavar="directory",
        help="Directory containing base.c, target.o and compile.sh. Multiple directories may be given.",
    )
    parser.add_argument(
        "--help=randomization-passes",
        nargs=0,
        action=PrintRandomizationPassesAction,
        help="Show documentation for all the available randomization passes.",
    )
    parser.add_argument(
        "--show-errors",
        dest="show_errors",
        action="store_true",
        help="Display compiler error/warning messages, and keep .c files for failed compiles.",
    )
    parser.add_argument(
        "--show-timings",
        dest="show_timings",
        action="store_true",
        help="Display the time taken by permuting vs. compiling vs. scoring.",
    )
    parser.add_argument(
        "--print-diffs",
        dest="print_diffs",
        action="store_true",
        help="Instead of compiling generated sources, display diffs against a base version.",
    )
    parser.add_argument(
        "--abort-exceptions",
        dest="abort_exceptions",
        action="store_true",
        help="Stop execution when an internal permuter exception occurs.",
    )
    parser.add_argument(
        "--better-only",
        dest="better_only",
        action="store_true",
        help="Only report scores better than the base.",
    )
    parser.add_argument(
        "--best-only",
        dest="best_only",
        action="store_true",
        help="Only report ties or new high scores.",
    )
    parser.add_argument(
        "--stop-on-zero",
        dest="stop_on_zero",
        action="store_true",
        help="Stop after producing an output with score 0.",
    )
    parser.add_argument(
        "--quiet",
        dest="quiet",
        action="store_true",
        help="Don't print a status line with the number of iterations.",
    )
    parser.add_argument(
        "--stack-diffs",
        dest="stack_differences",
        action="store_true",
        help="Take stack differences into account when computing the score.",
    )
    parser.add_argument(
        "--algorithm",
        dest="algorithm",
        default="difflib",
        choices=["difflib", "levenshtein"],
        help="Diff algorithm to use",
    )
    parser.add_argument(
        "--keep-prob",
        dest="keep_prob",
        metavar="PROB",
        type=restricted_float(0.0, 1.0),
        default=DEFAULT_RAND_KEEP_PROB,
        help="""Continue randomizing the previous output with the given probability
            (float in 0..1, default %(default)s).""",
    )
    parser.add_argument("--seed", dest="force_seed", type=str, help=argparse.SUPPRESS)
    parser.add_argument(
        "-j",
        dest="threads",
        type=int,
        default=0,
        help="Number of own threads to use (default: 1 without -J, 0 with -J).",
    )
    parser.add_argument(
        "-J",
        dest="use_network",
        action="store_true",
        help="Harness extra compute power through cyberspace (permuter@home).",
    )
    parser.add_argument(
        "--pah-debug",
        dest="network_debug",
        action="store_true",
        help="Enable debug prints for permuter@home.",
    )
    parser.add_argument(
        "--priority",
        dest="network_priority",
        metavar="PRIORITY",
        type=restricted_float(MIN_PRIO, MAX_PRIO),
        default=1.0,
        help=f"""Proportion of server resources to use when multiple people
            are using -J at the same time.
            Defaults to 1.0, meaning resources are split equally, but can be
            set to any value within [{MIN_PRIO}, {MAX_PRIO}].
            Each server runs with a priority threshold, which defaults to 0.1,
            below which they will not run permuter jobs at all.""",
    )
    parser.add_argument(
        "--no-context-output",
        dest="no_context_output",
        action="store_true",
        help="Removes all text above the target function in the output source files",
    )
    parser.add_argument(
        "--only-if-below",
        dest="score_threshold",
        type=int,
        help="Only report scores better (lower) than this value",
    )
    parser.add_argument(
        "--debug",
        dest="debug_mode",
        action="store_true",
        help="Debug mode, only compiles and scores the base for debugging issues",
    )
    parser.add_argument(
        "--speed",
        dest="speed",
        type=int,
        help="Speed%% to run at to reduce resources. Default 100",
        choices=range(1, 101),
        metavar="[1-100]",
        default=100,
    )
    parser.add_argument(
        "--no-ignore-branch-targets",
        dest="ign_branch_targets",
        action="store_false",
        help="Take branch targets into account when computing the score. "
        "Not recommended: it increases score volatility in combination "
        "with insertions/deletions.",
    )
    parser.add_argument(
        "--population-size",
        dest="population_size",
        type=int,
        default=0,
        metavar="N",
        help="""Opt-in evolutionary/population mode: maintain a population of
            the top N candidates found so far (by score, seeded with base.c)
            and periodically branch some workers' search onto an elite
            population member instead of relying purely on each worker's own
            continuous --keep-prob random walk or a full reset to base.c.
            0 (default) disables this and preserves exactly today's
            behavior. Requires multiple local workers (-j > 1); has no
            effect in single-threaded or pure -J (permuter@home-only) runs.
            A reasonable starting value is 5-10.""",
    )
    parser.add_argument(
        "--generation-interval",
        dest="generation_interval",
        type=float,
        default=60.0,
        metavar="SECONDS",
        help="""Wall-clock seconds between population-mode 'generations'
            (reseed dispatches). Only used when --population-size > 0.
            Default %(default)s.""",
    )
    parser.add_argument(
        "--population-fresh-fraction",
        dest="population_fresh_fraction",
        type=restricted_float(0.0, 1.0),
        default=0.5,
        metavar="PROB",
        help="""Fraction of each generation's reseed slots that reset a
            worker to the original base.c (for diversity) rather than
            branching it onto a random elite population member (to compound
            progress). Only used when --population-size > 0. Default
            %(default)s.""",
    )
    parser.add_argument(
        "--genetic",
        action="store_true",
        help="""Shorthand for turning on population/evolutionary mode with
            a reasonable default (--population-size 8), so you don't have
            to spell out --population-size yourself. --generation-interval
            and --population-fresh-fraction keep their own defaults either
            way - pass them explicitly if you want something other than
            the defaults. An explicit --population-size (any value,
            including 0) always takes precedence over this.""",
    )

    args = parser.parse_args()

    if args.genetic and args.population_size <= 0:
        args.population_size = 8

    threads = args.threads
    if not threads and not args.use_network:
        threads = 1

    options = Options(
        directories=args.directories,
        show_errors=args.show_errors,
        show_timings=args.show_timings,
        print_diffs=args.print_diffs,
        abort_exceptions=args.abort_exceptions,
        score_threshold=args.score_threshold,
        better_only=args.better_only,
        best_only=args.best_only,
        quiet=args.quiet,
        stack_differences=args.stack_differences,
        algorithm=args.algorithm,
        stop_on_zero=args.stop_on_zero,
        keep_prob=args.keep_prob,
        force_seed=args.force_seed,
        threads=threads,
        use_network=args.use_network,
        network_debug=args.network_debug,
        network_priority=args.network_priority,
        no_context_output=args.no_context_output,
        debug_mode=args.debug_mode,
        speed=args.speed,
        ign_branch_targets=args.ign_branch_targets,
        population_size=args.population_size,
        generation_interval=args.generation_interval,
        population_fresh_fraction=args.population_fresh_fraction,
    )

    run(options)


if __name__ == "__main__":
    main()
