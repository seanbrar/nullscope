"""Nullscope: Zero-cost telemetry for Python.

Provides ultra-low overhead no-op behavior when disabled and rich, contextual
metrics when enabled via environment flags.
"""

import inspect
import logging
import os
import re
import time
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from types import TracebackType
from typing import Any, Final, Protocol, TypeAlias, TypedDict, TypeVar, cast, runtime_checkable

log = logging.getLogger(__name__)

# Context-aware state for thread/async safety
_scope_stack_var: ContextVar[tuple[str, ...]] = ContextVar(
    "scope_stack",
    default=(),
)
_call_count_var: ContextVar[int] = ContextVar("call_count", default=0)

# Minimal-overhead optimization - evaluated once at import time
# Nullscope is enabled only via explicit env toggle to avoid unintended overhead
_NULLSCOPE_ENABLED = os.getenv("NULLSCOPE_ENABLED") == "1"
_STRICT_SCOPES = os.getenv("NULLSCOPE_STRICT") == "1"

# Built-in metadata keys (exported for clarity & resilience)
DEPTH: Final[str] = "depth"
PARENT_SCOPE: Final[str] = "parent_scope"
CALL_COUNT: Final[str] = "call_count"
METRIC_TYPE: Final[str] = "metric_type"
START_MONOTONIC_S: Final[str] = "start_monotonic_s"  # High-precision monotonic seconds
END_MONOTONIC_S: Final[str] = "end_monotonic_s"
START_WALL_TIME_S: Final[str] = "start_wall_time_s"  # Epoch seconds (correlation)
END_WALL_TIME_S: Final[str] = "end_wall_time_s"

_SCOPE_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*$")
_REQUIRED_REPORTER_METHODS: Final[tuple[str, ...]] = ("record_timing", "record_metric")

F = TypeVar("F", bound=Callable[..., Any])


class _TelemetryMetadata(TypedDict, total=False):
    depth: int
    call_count: int
    parent_scope: str | None
    start_monotonic_s: float
    end_monotonic_s: float
    start_wall_time_s: float
    end_wall_time_s: float


@runtime_checkable
class TelemetryReporter(Protocol):
    """Duck-typed protocol for telemetry reporters."""

    def record_timing(self, scope: str, duration: float, **metadata: Any) -> None: ...  # noqa: D102
    def record_metric(self, scope: str, value: Any, **metadata: Any) -> None: ...  # noqa: D102


def _validate_scope_name(name: str, *, kind: str) -> str:
    """Validate user-provided scope names with explicit errors."""
    if not isinstance(name, str):
        raise TypeError(f"{kind} must be a string, got {type(name).__name__}")
    if not name:
        raise ValueError(f"{kind} must be a non-empty string")
    if _STRICT_SCOPES and not _SCOPE_NAME_PATTERN.fullmatch(name):
        raise ValueError(
            f"Invalid {kind} '{name}'. Expected dot-separated lowercase segments "
            "using [a-z0-9_], e.g. 'http.request'.",
        )
    return name


def _validate_reporters(reporters: tuple[Any, ...]) -> None:
    """Check reporter shape early so failures are obvious at setup time."""
    for index, reporter in enumerate(reporters):
        missing = [
            method
            for method in _REQUIRED_REPORTER_METHODS
            if not callable(getattr(reporter, method, None))
        ]
        if missing:
            missing_list = ", ".join(missing)
            raise TypeError(
                f"Reporter at position {index} ({type(reporter).__name__}) is invalid. "
                f"Missing required callable method(s): {missing_list}.",
            )


@dataclass(frozen=True, slots=True)
class _NoOpTelemetryContext:
    """An immutable and stateless no-op context, optimized for negligible overhead."""

    def __call__(self, name: str, **metadata: Any) -> "_NoOpTelemetryContext":  # noqa: ARG002
        return self  # Self is already a context manager

    def __enter__(self) -> "_NoOpTelemetryContext":
        return self

    def __exit__(
        self,
        _: type[BaseException] | None,
        __: BaseException | None,
        ___: TracebackType | None,
    ) -> bool | None:
        return None

    @property
    def is_enabled(self) -> bool:
        return False

    def metric(self, name: str, value: Any, **metadata: Any) -> None:
        pass

    def time(self, name: str, **metadata: Any) -> "_NoOpTelemetryContext":  # noqa: ARG002
        return self

    def count(self, name: str, increment: int = 1, **metadata: Any) -> None:
        pass

    def gauge(self, name: str, value: float, **metadata: Any) -> None:
        pass

    def timed(self, name: str, **metadata: Any) -> Callable[[F], F]:  # noqa: ARG002
        """No-op decorator that returns the original function unchanged."""
        return lambda func: func

    def flush(self) -> None:
        pass

    def shutdown(self) -> None:
        pass


class _EnabledTelemetryContext:
    """Full-featured telemetry context when enabled."""

    __slots__ = ("reporters",)

    def __init__(self, *reporters: TelemetryReporter):
        self.reporters = reporters

    def __call__(
        self, name: str, **metadata: Any
    ) -> AbstractContextManager["_EnabledTelemetryContext"]:
        return self._create_scope(name, **metadata)

    @contextmanager
    def _create_scope(
        self, name: str, **metadata: Any
    ) -> Iterator["_EnabledTelemetryContext"]:
        """The actual scope implementation when enabled."""
        _validate_scope_name(name, kind="Scope name")

        scope_stack = _scope_stack_var.get()
        call_count = _call_count_var.get()
        scope_path = ".".join((*scope_stack, name))
        start_monotonic_s = time.perf_counter()
        start_wall_time_s = time.time()

        scope_token = _scope_stack_var.set((*scope_stack, name))
        count_token = _call_count_var.set(call_count + 1)

        try:
            yield self  # Return self so chained methods work
        finally:
            end_monotonic_s = time.perf_counter()
            duration = end_monotonic_s - start_monotonic_s
            end_wall_time_s = start_wall_time_s + duration
            _scope_stack_var.reset(scope_token)
            _call_count_var.reset(count_token)

            final_stack = _scope_stack_var.get()
            final_count = _call_count_var.get()

            built: _TelemetryMetadata = {
                "depth": len(final_stack),
                "call_count": final_count,
                "parent_scope": ".".join(final_stack) if final_stack else None,
                "start_monotonic_s": start_monotonic_s,
                "end_monotonic_s": end_monotonic_s,
                "start_wall_time_s": start_wall_time_s,
                "end_wall_time_s": end_wall_time_s,
            }
            enhanced_metadata: dict[str, Any] = {**built, **metadata}

            for reporter in self.reporters:
                try:
                    reporter.record_timing(scope_path, duration, **enhanced_metadata)
                except Exception as e:
                    log.error(
                        "Telemetry reporter '%s' failed: %s",
                        type(reporter).__name__,
                        e,
                        exc_info=True,
                    )

    def metric(self, name: str, value: Any, **metadata: Any) -> None:
        """Record a metric within current scope context."""
        scope_stack = _scope_stack_var.get()
        scope_path = ".".join((*scope_stack, name))
        built: _TelemetryMetadata = {
            "depth": len(scope_stack),
            "parent_scope": ".".join(scope_stack) if scope_stack else None,
        }
        enhanced_metadata: dict[str, Any] = {**built, **metadata}
        for reporter in self.reporters:
            try:
                reporter.record_metric(scope_path, value, **enhanced_metadata)
            except Exception as e:
                log.error(
                    "Telemetry reporter '%s' failed: %s",
                    type(reporter).__name__,
                    e,
                    exc_info=True,
                )

    # Convenience methods for chaining
    def time(
        self, name: str, **metadata: Any
    ) -> AbstractContextManager["_EnabledTelemetryContext"]:
        """Alias for scope() - more intuitive for timing operations."""
        return self(name, **metadata)

    def count(self, name: str, increment: int = 1, **metadata: Any) -> None:
        """Record a counter metric."""
        self.metric(name, increment, metric_type="counter", **metadata)

    def gauge(self, name: str, value: float, **metadata: Any) -> None:
        """Record a gauge metric."""
        self.metric(name, value, metric_type="gauge", **metadata)

    def timed(self, name: str, **metadata: Any) -> Callable[[F], F]:
        """Decorator that times function execution under a fixed scope name."""
        scope_name = _validate_scope_name(name, kind="Decorator scope name")

        def decorator(func: F) -> F:
            if inspect.iscoroutinefunction(func):
                @wraps(func)
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    with self(scope_name, **metadata):
                        return await func(*args, **kwargs)

                return cast("F", async_wrapper)

            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                with self(scope_name, **metadata):
                    return func(*args, **kwargs)

            return cast("F", wrapper)

        return decorator

    def _call_reporter_hook(self, hook_name: str) -> None:
        """Call an optional lifecycle hook on all reporters."""
        for reporter in self.reporters:
            hook = getattr(reporter, hook_name, None)
            if hook is None or not callable(hook):
                continue
            try:
                hook()
            except Exception as e:
                log.error(
                    "Reporter '%s' failed during %s: %s",
                    type(reporter).__name__,
                    hook_name,
                    e,
                    exc_info=True,
                )

    def flush(self) -> None:
        """Flush reporters that implement optional lifecycle hooks."""
        self._call_reporter_hook("flush")

    def shutdown(self) -> None:
        """Shutdown reporters that implement optional lifecycle hooks."""
        self._call_reporter_hook("shutdown")

    @property
    def is_enabled(self) -> bool:
        return True


_NO_OP_SINGLETON = _NoOpTelemetryContext()

TelemetryContextProtocol: TypeAlias = _EnabledTelemetryContext | _NoOpTelemetryContext


def TelemetryContext(*reporters: TelemetryReporter) -> TelemetryContextProtocol:  # noqa: N802
    """Return a telemetry context.

    Behavior:
    - When nullscope is enabled (``NULLSCOPE_ENABLED=1``), return an enabled
      context. If no reporters are provided, install a default in-memory
      ``SimpleReporter`` for convenience.
    - When nullscope is disabled, return a shared no-op instance for negligible
      overhead.
    """
    if _NULLSCOPE_ENABLED:
        if reporters:
            _validate_reporters(reporters)
        reps = reporters or (SimpleReporter(),)
        return _EnabledTelemetryContext(*reps)
    # Always return the same, pre-existing no-op instance.
    return _NO_OP_SINGLETON


class SimpleReporter:
    """Built-in reporter for development and debugging.

    Collects telemetry in memory for inspection. Not intended for production use.
    To view collected data, call `print_report()` or `get_report()`.
    """

    def __init__(self, max_entries_per_scope: int = 1000):
        self.max_entries = max_entries_per_scope
        self.timings: dict[str, deque[tuple[float, dict[str, Any]]]] = {}
        self.metrics: dict[str, deque[tuple[Any, dict[str, Any]]]] = {}

    def record_timing(self, scope: str, duration: float, **metadata: Any) -> None:
        if scope not in self.timings:
            self.timings[scope] = deque(maxlen=self.max_entries)
        self.timings[scope].append((duration, metadata))

    def record_metric(self, scope: str, value: Any, **metadata: Any) -> None:
        if scope not in self.metrics:
            self.metrics[scope] = deque(maxlen=self.max_entries)
        self.metrics[scope].append((value, metadata))

    def print_report(self) -> None:
        """Prints the report to stdout if any data was collected."""
        if self.timings or self.metrics:
            print(self.get_report())  # noqa: T201

    def reset(self) -> None:
        """Clear all collected telemetry (testing convenience)."""
        self.timings.clear()
        self.metrics.clear()

    def flush(self) -> None:
        """No-op lifecycle method for API compatibility."""

    def shutdown(self) -> None:
        """No-op lifecycle method for API compatibility."""

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of collected data (testing)."""
        return {
            "timings": {key: list(values) for key, values in self.timings.items()},
            "metrics": {key: list(values) for key, values in self.metrics.items()},
        }

    def get_report(self) -> str:
        """Generate hierarchical telemetry report."""
        lines = ["=== Nullscope Report ===\n"]

        # Group by hierarchy
        timing_tree = self._build_hierarchy(self.timings)
        self._format_tree(timing_tree, lines, "Timings")

        if self.metrics:
            lines.append("\n--- Metrics ---")
            for scope, values in sorted(self.metrics.items()):
                total = sum(v[0] for v in values if isinstance(v[0], int | float))
                lines.append(
                    f"{scope:<40} | Count: {len(values):<4} | Total: {total:,.0f}",
                )

        return "\n".join(lines)

    def _build_hierarchy(
        self,
        data: dict[str, deque[tuple[float, dict[str, Any]]]],
    ) -> dict[str, Any]:
        """Build tree structure from dot-separated scope names."""
        tree: dict[str, Any] = {}
        for scope, values in data.items():
            parts = scope.split(".")
            current = tree
            for part in parts[:-1]:
                current = current.setdefault(part, {})
            current[parts[-1]] = values
        return tree

    def _format_tree(
        self,
        tree: dict[str, Any],
        lines: list[str],
        title: str,
        depth: int = 0,
    ) -> None:
        """Format hierarchical tree with indentation."""
        if depth == 0:
            lines.append(f"\n--- {title} ---")

        for key, value in sorted(tree.items()):
            indent = "  " * depth
            if isinstance(value, dict):
                lines.append(f"{indent}{key}:")
                self._format_tree(value, lines, title, depth + 1)
            else:
                # Leaf node - actual timing data
                durations = [v[0] for v in value]
                avg_time = sum(durations) / len(durations)
                total_time = sum(durations)
                lines.append(
                    f"{indent}{key:<30} | "
                    f"Calls: {len(durations):<4} | "
                    f"Avg: {avg_time:.4f}s | "
                    f"Total: {total_time:.4f}s",
                )
