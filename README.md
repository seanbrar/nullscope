# Nullscope

[![PyPI](https://img.shields.io/pypi/v/nullscope)](https://pypi.org/project/nullscope/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Zero-cost telemetry for Python. No-op when disabled, rich context when enabled.

## Why Nullscope?

Most telemetry libraries have runtime cost even when you don't need them. Nullscope is different:

- **Disabled**: Returns a singleton no-op object. No allocations, no timing calls, no overhead.
- **Enabled**: Full-featured timing and metrics with automatic scope hierarchy.

This makes Nullscope ideal for **libraries** (users can enable telemetry if they want) and **applications** where you want zero production overhead but rich debugging capability.

```python
from nullscope import TelemetryContext

# When NULLSCOPE_ENABLED != "1", this is literally just returning a cached object
telemetry = TelemetryContext()

with telemetry("database.query"):  # No-op when disabled
    results = db.execute(query)
```

## What Nullscope Is Not

- **A distributed tracing system.** No trace propagation, no span IDs, no context injection for cross-service correlation. If you need that, use OpenTelemetry directly. Nullscope can *feed* OTel, but it doesn't replace it.

- **A metrics aggregation layer.** Nullscope reports raw events to reporters. It doesn't compute percentiles, histograms, or roll up data. That's the reporter's job (or the backend's).

- **Auto-instrumentation.** Nullscope won't patch your HTTP client or database driver. You instrument what you want, explicitly.

- **A logging framework.** Scopes are for timing and metrics, not structured log events. (Though a reporter *could* emit logs.)

## Installation

```bash
pip install nullscope
```

With OpenTelemetry support:

```bash
pip install nullscope[otel]
```

## Quick Start

```python
import os
os.environ["NULLSCOPE_ENABLED"] = "1"  # Enable telemetry

from nullscope import TelemetryContext, SimpleReporter

# Create a reporter to see output
reporter = SimpleReporter()
telemetry = TelemetryContext(reporter)

# Time operations with automatic hierarchy
with telemetry("request"):
    with telemetry("auth"):
        validate_token()

    with telemetry("handler"):
        process_data()

# See what was collected
reporter.print_report()
```

Output:

```text
=== Nullscope Report ===

--- Timings ---
request:
  auth                           | Calls: 1    | Avg: 0.0012s | Total: 0.0012s
  handler                        | Calls: 1    | Avg: 0.0234s | Total: 0.0234s
```

## Configuration

| Environment Variable  | Description                          |
| --------------------- | ------------------------------------ |
| `NULLSCOPE_ENABLED=1` | Enable telemetry (default: disabled) |

Note: environment flags are read at import time. In tests, reload `nullscope` after changing env vars.

## API

### TelemetryContext

```python
from nullscope import TelemetryContext

telemetry = TelemetryContext()  # Uses default SimpleReporter when enabled
telemetry = TelemetryContext(my_reporter)  # Custom reporter
telemetry = TelemetryContext(reporter1, reporter2)  # Multiple reporters
```

### Scopes (Timing)

```python
with telemetry("operation"):
    do_work()

# With metadata
with telemetry("http.request", method="GET", path="/api/users"):
    handle_request()
```

### Metrics

```python
telemetry.count("cache.hit")  # Increment counter
telemetry.count("items.processed", 5)  # Increment by N
telemetry.gauge("queue.depth", len(queue))  # Point-in-time value
telemetry.metric("custom", value, metric_type="counter")  # Generic
```

### Check Status

```python
if telemetry.is_enabled:
    # Do expensive debug logging
    pass
```

## OpenTelemetry Adapter

Export to OpenTelemetry backends:

```python
from nullscope import TelemetryContext
from nullscope.adapters.opentelemetry import OTelReporter

# Configure OTel SDK first (providers, exporters, etc.)
# Then use Nullscope with OTel reporter
telemetry = TelemetryContext(OTelReporter(service_name="my-service"))
```

The adapter emits:

- **Timings** → Histogram (seconds) + synthetic Span when wall-clock bounds are present
- **Counters** → Counter
- **Gauges** → Histogram (sampled values, since Python OTel sync gauge support is limited)

## Documentation

- [Design](docs/design.md) - Architecture and implementation details
- [Examples](docs/examples.md) - Real-world usage patterns
- [Comparison](docs/comparison.md) - When to use Nullscope vs alternatives
- [Roadmap](ROADMAP.md) - Version milestones and planned features

## License

[MIT](LICENSE)
