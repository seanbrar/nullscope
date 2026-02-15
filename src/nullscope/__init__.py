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
from collections.abc import Callable
from contextlib import AbstractContextManager
from contextvars import ContextVar, Token
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
ERROR: Final[str] = "error"
ERROR_TYPE: Final[str] = "error_type"
ERROR_MESSAGE: Final[str] = "error_message"

_SCOPE_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*$")
_REQUIRED_REPORTER_METHODS: Final[tuple[str, ...]] = ("record_timing", "record_metric")

F = TypeVar("F", bound=Callable[..., Any])


def _identity(func: F) -> F:
    """Identity function for no-op decorator (avoids lambda allocation)."""
    return func


class _TelemetryMetadata(TypedDict, total=False):
    depth: int
    call_count: int
    parent_scope: str | None
    start_monotonic_s: float
    end_monotonic_s: float
    start_wall_time_s: float
    end_wall_time_s: float
    error: bool
    error_type: str
    error_message: str


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
        return _identity

    def flush(self) -> None:
        pass

    def shutdown(self) -> None:
        pass


class _Scope:
    """Explicit context manager for scope timing (replaces @contextmanager for performance)."""

    __slots__ = (
        "_ctx",
        "_name",
        "_scope_path",
        "_metadata",
        "_start_monotonic_s",
        "_start_wall_time_s",
        "_scope_token",
        "_count_token",
    )

    def __init__(
        self, ctx: "_EnabledTelemetryContext", name: str, metadata: dict[str, Any]
    ) -> None:
        _validate_scope_name(name, kind="Scope name")
        self._ctx = ctx
        self._name = name
        self._metadata = metadata

        # These are set in __enter__ to maintain context manager contract
        self._scope_path: str = ""
        self._scope_token: Token[tuple[str, ...]] | None = None
        self._count_token: Token[int] | None = None
        self._start_monotonic_s: float = 0.0
        self._start_wall_time_s: float = 0.0

    def __enter__(self) -> "_EnabledTelemetryContext":
        # Capture current state and compute new stack
        scope_stack = _scope_stack_var.get()
        new_stack = (*scope_stack, self._name)
        self._scope_path = ".".join(new_stack)

        # Set context vars here (not in __init__) to maintain context manager contract
        self._scope_token = _scope_stack_var.set(new_stack)
        self._count_token = _call_count_var.set(_call_count_var.get() + 1)

        self._start_monotonic_s = time.perf_counter()
        self._start_wall_time_s = time.time()
        return self._ctx

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        end_monotonic_s = time.perf_counter()
        duration = end_monotonic_s - self._start_monotonic_s
        end_wall_time_s = self._start_wall_time_s + duration

        # Reset context vars (tokens are guaranteed set by __enter__)
        assert self._scope_token is not None
        assert self._count_token is not None
        _scope_stack_var.reset(self._scope_token)
        _call_count_var.reset(self._count_token)

        # Capture final state for metadata
        final_stack = _scope_stack_var.get()
        final_count = _call_count_var.get()

        # Build metadata (user metadata overrides built-in)
        built: _TelemetryMetadata = {
            "depth": len(final_stack),
            "call_count": final_count,
            "parent_scope": ".".join(final_stack) if final_stack else None,
            "start_monotonic_s": self._start_monotonic_s,
            "end_monotonic_s": end_monotonic_s,
            "start_wall_time_s": self._start_wall_time_s,
            "end_wall_time_s": end_wall_time_s,
        }
        if exc_type is not None:
            built["error"] = True
            built["error_type"] = exc_type.__qualname__
            built["error_message"] = str(exc_val) if exc_val else ""
        enhanced_metadata: dict[str, Any] = {**built, **self._metadata}

        # Report to all reporters
        for reporter in self._ctx.reporters:
            try:
                reporter.record_timing(self._scope_path, duration, **enhanced_metadata)
            except Exception as e:
                log.error(
                    "Telemetry reporter '%s' failed: %s",
                    type(reporter).__name__,
                    e,
                    exc_info=True,
                )


class _EnabledTelemetryContext:
    """Full-featured telemetry context when enabled."""

    __slots__ = ("reporters",)

    def __init__(self, *reporters: TelemetryReporter):
        self.reporters = reporters

    def __call__(
        self, name: str, **metadata: Any
    ) -> AbstractContextManager["_EnabledTelemetryContext"]:
        return _Scope(self, name, metadata)

    def metric(self, name: str, value: Any, **metadata: Any) -> None:
        """Record a metric within current scope context."""
        scope_stack = _scope_stack_var.get()
        parent_scope = ".".join(scope_stack) if scope_stack else None
        scope_path = f"{parent_scope}.{name}" if parent_scope else name
        built: _TelemetryMetadata = {
            "depth": len(scope_stack),
            "parent_scope": parent_scope,
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
        """Return a JSON-serializable snapshot of collected data."""
        return {
            "timings": {
                scope: [{"duration": d, **m} for d, m in entries]
                for scope, entries in self.timings.items()
            },
            "metrics": {
                scope: [{"value": v, **m} for v, m in entries]
                for scope, entries in self.metrics.items()
            },
        }

    def get_report(self) -> str:
        """Generate hierarchical telemetry report."""
        lines = ["=== Nullscope Report ===\n"]
        self._format_timings(lines)
        self._format_metrics(lines)
        return "\n".join(lines)

    def _format_timings(self, lines: list[str]) -> None:
        if not self.timings:
            return
        lines.append("\n--- Timings ---")
        scope_set = set(self.timings)
        # Identify prefixes that need structural header lines
        headers: set[str] = set()
        for scope in scope_set:
            parts = scope.split(".")
            for i in range(1, len(parts)):
                prefix = ".".join(parts[:i])
                if prefix not in scope_set:
                    headers.add(prefix)

        for item in sorted(scope_set | headers):
            parts = item.split(".")
            indent = "  " * (len(parts) - 1)
            name = parts[-1]
            if item in headers:
                lines.append(f"{indent}{name}:")
            else:
                values = self.timings[item]
                durations = [v[0] for v in values]
                avg = sum(durations) / len(durations)
                total = sum(durations)
                lines.append(
                    f"{indent}{name:<30} | "
                    f"Calls: {len(durations):<4} | "
                    f"Avg: {avg:.4f}s | "
                    f"Total: {total:.4f}s",
                )

    def _format_metrics(self, lines: list[str]) -> None:
        if not self.metrics:
            return
        lines.append("\n--- Metrics ---")
        for scope, values in sorted(self.metrics.items()):
            count = len(values)
            numeric = [v[0] for v in values if isinstance(v[0], int | float)]
            # Check metric_type from any entry's metadata
            metric_type = None
            for _, meta in values:
                if "metric_type" in meta:
                    metric_type = meta["metric_type"]
                    break
            if metric_type == "gauge" and numeric:
                last = numeric[-1]
                lines.append(f"{scope:<40} | Count: {count:<4} | Last: {last:,.2f}")
            elif numeric:
                total = sum(numeric)
                lines.append(f"{scope:<40} | Count: {count:<4} | Total: {total:,.0f}")
            else:
                lines.append(f"{scope:<40} | Count: {count:<4}")


class LogReporter:
    """Reporter that emits telemetry as structured log records.

    Useful in production environments with existing log aggregation.
    Scope and primary value appear in the message; full metadata is
    attached via the ``extra`` dict for formatters that support it.
    """

    __slots__ = ("_logger", "_level")

    def __init__(
        self, logger: logging.Logger | None = None, level: int = logging.DEBUG
    ) -> None:
        self._logger = logger or logging.getLogger("nullscope.telemetry")
        self._level = level

    def record_timing(self, scope: str, duration: float, **metadata: Any) -> None:
        self._logger.log(
            self._level,
            "timing scope=%s duration=%.4fs",
            scope,
            duration,
            extra={"telemetry_metadata": metadata},
        )

    def record_metric(self, scope: str, value: Any, **metadata: Any) -> None:
        self._logger.log(
            self._level,
            "metric scope=%s value=%s",
            scope,
            value,
            extra={"telemetry_metadata": metadata},
        )
