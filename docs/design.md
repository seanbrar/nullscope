# Nullscope Design

This document explains the architectural decisions behind Nullscope.

## 1) The Zero-Cost Abstraction Pattern

Nullscope's core design principle is that telemetry should have **zero runtime cost when disabled**. This is achieved through a singleton no-op pattern.

### How It Works

When `NULLSCOPE_ENABLED` is not set (or not `"1"`), calling `TelemetryContext()` returns a pre-allocated singleton instance of `_NoOpTelemetryContext`:

```python
_NO_OP_SINGLETON = _NoOpTelemetryContext()

def TelemetryContext(*reporters):
    if _NULLSCOPE_ENABLED:
        return _EnabledTelemetryContext(*reporters)
    return _NO_OP_SINGLETON  # Always the same instance
```

The no-op context is:

- **Immutable** (`@dataclass(frozen=True, slots=True)`)
- **Stateless** (no instance variables)
- **Self-returning** (`__call__` returns `self`, making it its own context manager)

This means disabled telemetry reduces to:

1. One dict lookup for `_NULLSCOPE_ENABLED` (evaluated once at import)
2. Return of a pre-existing object reference
3. No allocations, no context manager overhead, no timing calls

### Why Environment Variables at Import Time

Nullscope evaluates environment variables once at module import:

```python
_NULLSCOPE_ENABLED = os.getenv("NULLSCOPE_ENABLED") == "1"
```

This deliberate design choice means:

1. **No runtime checks**: Every `with telemetry("scope")` doesn't need to check env vars
2. **Branch prediction friendly**: The enabled/disabled path is fixed for the process lifetime
3. **Predictable behavior**: Telemetry state can't change mid-request

The tradeoff is that you can't dynamically enable/disable telemetry. This is intentional—if you need dynamic control, use the reporter layer instead. In tests, reload `nullscope` after changing env vars.

## 2) Context Variables for Async Safety

Nullscope uses Python's `contextvars` module to track scope hierarchy:

```python
_scope_stack_var: ContextVar[tuple[str, ...]] = ContextVar("scope_stack", default=())
_call_count_var: ContextVar[int] = ContextVar("call_count", default=0)
```

This provides automatic isolation for:

- **Async tasks**: Each `asyncio.Task` gets its own scope stack
- **Threads**: Each thread has independent context
- **Nested scopes**: Child scopes correctly report parent relationships

When you write:

```python
async def handle_request():
    with telemetry("request"):
        await process_data()  # Other tasks have their own scope stack
        with telemetry("validation"):
            ...  # Correctly nested under "request"
```

The scope hierarchy "request.validation" is maintained correctly even with concurrent async operations.

## 3) Reporter Protocol Design

Reporters implement a simple duck-typed protocol:

```python
class TelemetryReporter(Protocol):
    def record_timing(self, scope: str, duration: float, **metadata: Any) -> None: ...
    def record_metric(self, scope: str, value: Any, **metadata: Any) -> None: ...
```

### Design Decisions

1. **Two methods, not many**: Rather than separate methods for counters, gauges, histograms, etc., we have `record_timing` (for scopes) and `record_metric` (for everything else). The `metric_type` metadata key distinguishes counter vs gauge.

2. **Keyword-only metadata**: All contextual information flows through `**metadata`. This makes the protocol stable—new metadata keys don't require protocol changes.

3. **Error isolation**: Reporter failures are logged but don't propagate. Your application continues running even if telemetry export fails.

4. **Multiple reporters**: `TelemetryContext()` accepts multiple reporters. All receive the same data.

## 4) Scope Hierarchy Implementation

Scopes form a hierarchy through dot-separated names:

```python
with telemetry("http"):
    with telemetry("parse"):
        ...  # scope = "http.parse"
```

The implementation:

1. Maintains a stack of scope names in context vars
2. On scope entry: pushes name, starts timing
3. On scope exit: pops name, calculates duration, reports with full path

## 5) Metadata Contract

Nullscope automatically includes metadata with every timing:

- `depth`: Nesting level (0 = root)
- `parent_scope`: Dot-joined parent path (or `None` at root)
- `call_count`: Incrementing counter for ordering
- `start_monotonic_s` / `end_monotonic_s`: Monotonic timestamps
- `start_wall_time_s` / `end_wall_time_s`: Wall clock timestamps

These keys are exported as constants from `nullscope` for reporter implementations:

```python
from nullscope import DEPTH, PARENT_SCOPE, CALL_COUNT, START_WALL_TIME_S
```

This allows reporters to reference keys without hardcoding strings, and provides a stable contract for what metadata is always present.
