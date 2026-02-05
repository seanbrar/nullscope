"""Core benchmarks for nullscope.

Run with: python benchmarks/bench_core.py

These benchmarks measure the hot paths to establish baselines before optimization.
"""

import gc
import os
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass

# Ensure we can import nullscope from the source tree
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@dataclass
class BenchResult:
    name: str
    iterations: int
    total_ns: float
    per_call_ns: float
    std_dev_ns: float
    samples: list[float]

    def __str__(self) -> str:
        return (
            f"{self.name:<45} | "
            f"{self.per_call_ns:>8.1f} ns/call | "
            f"±{self.std_dev_ns:>6.1f} ns | "
            f"({self.iterations:,} iterations)"
        )


def bench(
    name: str,
    fn: Callable[[], None],
    iterations: int = 100_000,
    warmup: int = 1_000,
    samples: int = 5,
) -> BenchResult:
    """Run a benchmark with warmup and multiple samples."""
    # Warmup
    for _ in range(warmup):
        fn()

    # Collect samples
    sample_times: list[float] = []
    for _ in range(samples):
        gc.disable()
        start = time.perf_counter_ns()
        for _ in range(iterations):
            fn()
        end = time.perf_counter_ns()
        gc.enable()
        sample_times.append(end - start)

    total_ns = statistics.mean(sample_times)
    per_call_ns = total_ns / iterations
    std_dev_ns = statistics.stdev(sample_times) / iterations if len(sample_times) > 1 else 0

    return BenchResult(
        name=name,
        iterations=iterations,
        total_ns=total_ns,
        per_call_ns=per_call_ns,
        std_dev_ns=std_dev_ns,
        samples=sample_times,
    )


def run_baseline_benchmarks() -> list[BenchResult]:
    """Baseline measurements for comparison."""
    results: list[BenchResult] = []

    # Raw perf_counter call (absolute baseline)
    def raw_perf_counter() -> None:
        time.perf_counter()

    results.append(bench("baseline: time.perf_counter()", raw_perf_counter))

    # Function call overhead
    def noop() -> None:
        pass

    results.append(bench("baseline: empty function call", noop))

    # Context manager baseline (minimal)
    class MinimalCM:
        def __enter__(self) -> "MinimalCM":
            return self

        def __exit__(self, *args: object) -> None:
            pass

    cm = MinimalCM()

    def minimal_cm() -> None:
        with cm:
            pass

    results.append(bench("baseline: minimal context manager", minimal_cm))

    return results


def run_disabled_benchmarks() -> list[BenchResult]:
    """Benchmarks with NULLSCOPE_ENABLED unset (no-op mode)."""
    # Ensure disabled mode
    os.environ.pop("NULLSCOPE_ENABLED", None)

    # Force reimport to get disabled state
    if "nullscope" in sys.modules:
        del sys.modules["nullscope"]

    from nullscope import TelemetryContext

    telemetry = TelemetryContext()
    assert not telemetry.is_enabled, "Expected disabled mode"

    results: list[BenchResult] = []

    # Scope entry/exit (the critical path)
    def scope_entry_exit() -> None:
        with telemetry("test"):
            pass

    results.append(bench("disabled: scope entry/exit", scope_entry_exit))

    # Nested scopes
    def nested_scopes_3() -> None:
        with telemetry("a"):
            with telemetry("b"):
                with telemetry("c"):
                    pass

    results.append(bench("disabled: 3 nested scopes", nested_scopes_3, iterations=50_000))

    # timed() decorator creation
    def timed_decorator_creation() -> None:
        telemetry.timed("test")

    results.append(bench("disabled: timed() decorator creation", timed_decorator_creation))

    # timed() decorator application
    @telemetry.timed("bench_fn")
    def decorated_fn() -> None:
        pass

    results.append(bench("disabled: timed() decorated call", decorated_fn))

    # metric() call
    def metric_call() -> None:
        telemetry.metric("test", 1)

    results.append(bench("disabled: metric() call", metric_call))

    # count() call
    def count_call() -> None:
        telemetry.count("test")

    results.append(bench("disabled: count() call", count_call))

    # gauge() call
    def gauge_call() -> None:
        telemetry.gauge("test", 1.0)

    results.append(bench("disabled: gauge() call", gauge_call))

    # is_enabled property access
    def is_enabled_check() -> None:
        _ = telemetry.is_enabled

    results.append(bench("disabled: is_enabled check", is_enabled_check))

    return results


def run_enabled_benchmarks() -> list[BenchResult]:
    """Benchmarks with NULLSCOPE_ENABLED=1 (active mode)."""
    os.environ["NULLSCOPE_ENABLED"] = "1"

    # Force reimport to get enabled state
    if "nullscope" in sys.modules:
        del sys.modules["nullscope"]

    from nullscope import SimpleReporter, TelemetryContext

    # Use a reporter with high max_entries to avoid deque rotation overhead
    reporter = SimpleReporter(max_entries_per_scope=1_000_000)
    telemetry = TelemetryContext(reporter)
    assert telemetry.is_enabled, "Expected enabled mode"

    results: list[BenchResult] = []

    # Scope entry/exit (the critical path)
    def scope_entry_exit() -> None:
        with telemetry("test"):
            pass

    results.append(bench("enabled: scope entry/exit", scope_entry_exit, iterations=50_000))

    # Nested scopes
    def nested_scopes_3() -> None:
        with telemetry("a"):
            with telemetry("b"):
                with telemetry("c"):
                    pass

    results.append(bench("enabled: 3 nested scopes", nested_scopes_3, iterations=20_000))

    # Deep nesting
    def nested_scopes_5() -> None:
        with telemetry("a"):
            with telemetry("b"):
                with telemetry("c"):
                    with telemetry("d"):
                        with telemetry("e"):
                            pass

    results.append(bench("enabled: 5 nested scopes", nested_scopes_5, iterations=10_000))

    # timed() decorator creation
    def timed_decorator_creation() -> None:
        telemetry.timed("test")

    results.append(bench("enabled: timed() decorator creation", timed_decorator_creation))

    # timed() decorator application
    @telemetry.timed("bench_fn")
    def decorated_fn() -> None:
        pass

    results.append(bench("enabled: timed() decorated call", decorated_fn, iterations=50_000))

    # metric() call
    def metric_call() -> None:
        telemetry.metric("test", 1)

    results.append(bench("enabled: metric() call", metric_call, iterations=50_000))

    # count() call
    def count_call() -> None:
        telemetry.count("test")

    results.append(bench("enabled: count() call", count_call, iterations=50_000))

    # gauge() call
    def gauge_call() -> None:
        telemetry.gauge("test", 1.0)

    results.append(bench("enabled: gauge() call", gauge_call, iterations=50_000))

    # Scope with metadata
    def scope_with_metadata() -> None:
        with telemetry("test", method="GET", path="/api"):
            pass

    results.append(
        bench("enabled: scope with 2 metadata keys", scope_with_metadata, iterations=50_000)
    )

    return results


def main() -> None:
    print("=" * 80)
    print("NULLSCOPE BENCHMARK SUITE")
    print("=" * 80)
    print(f"Python: {sys.version}")
    print(f"Platform: {sys.platform}")
    print()

    print("-" * 80)
    print("BASELINES (reference measurements)")
    print("-" * 80)
    for result in run_baseline_benchmarks():
        print(result)
    print()

    print("-" * 80)
    print("DISABLED MODE (NULLSCOPE_ENABLED unset)")
    print("-" * 80)
    for result in run_disabled_benchmarks():
        print(result)
    print()

    print("-" * 80)
    print("ENABLED MODE (NULLSCOPE_ENABLED=1)")
    print("-" * 80)
    for result in run_enabled_benchmarks():
        print(result)
    print()

    print("=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
