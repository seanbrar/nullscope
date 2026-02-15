# Nullscope Examples

Real-world usage patterns for Nullscope.

## Basic Usage

```python
from nullscope import TelemetryContext

# Create a context (no-op if NULLSCOPE_ENABLED != "1")
telemetry = TelemetryContext()

# Time an operation
with telemetry("database.query"):
    results = db.execute(query)

# Record metrics
telemetry.count("cache.hit")
telemetry.gauge("queue.depth", len(queue))
```

## FastAPI Middleware Integration

```python
from fastapi import FastAPI, Request
from nullscope import TelemetryContext

app = FastAPI()
telemetry = TelemetryContext()

@app.middleware("http")
async def telemetry_middleware(request: Request, call_next):
    with telemetry("http.request", method=request.method, path=request.url.path):
        response = await call_next(request)
        telemetry.gauge("http.status", response.status_code)
        return response
```

## CLI Tool Instrumentation

```python
import click
from nullscope import TelemetryContext, SimpleReporter

@click.command()
@click.option("--verbose", is_flag=True)
def main(verbose):
    # Use SimpleReporter for CLI tools to see output
    reporter = SimpleReporter()
    telemetry = TelemetryContext(reporter)

    with telemetry("cli.run"):
        with telemetry("load_config"):
            config = load_config()

        with telemetry("process"):
            process(config)

    if verbose:
        reporter.print_report()
```

## Nested Scopes

```python
telemetry = TelemetryContext()

with telemetry("request"):
    with telemetry("auth"):
        validate_token()  # scope: request.auth

    with telemetry("handler"):
        with telemetry("db"):
            fetch_data()  # scope: request.handler.db

        with telemetry("render"):
            render_template()  # scope: request.handler.render
```

## Decorator-Based Timing

```python
from nullscope import TelemetryContext

telemetry = TelemetryContext()

@telemetry.timed("http.handler", route="/users")
def handle_users():
    return query_users()

@telemetry.timed("jobs.refresh_cache")
async def refresh_cache():
    ...
```

## Logging Reporter

```python
import logging
from nullscope import TelemetryContext, LogReporter

# Uses the "nullscope.telemetry" logger at DEBUG level by default
telemetry = TelemetryContext(LogReporter())

# Or configure with a custom logger and level
logger = logging.getLogger("myapp.telemetry")
telemetry = TelemetryContext(LogReporter(logger=logger, level=logging.INFO))

with telemetry("request"):
    handle_request()
# Emits: "timing scope=request duration=0.0234s"
```

## Custom Reporter Implementation

```python
import json
from nullscope import TelemetryReporter

class JsonFileReporter(TelemetryReporter):
    def __init__(self, path: str):
        self.path = path
        self.file = open(path, "a", encoding="utf-8")

    def record_timing(self, scope: str, duration: float, **metadata) -> None:
        self.file.write(json.dumps({
            "type": "timing",
            "scope": scope,
            "duration": duration,
            **metadata
        }) + "\n")
        self.file.flush()

    def record_metric(self, scope: str, value, **metadata) -> None:
        self.file.write(json.dumps({
            "type": "metric",
            "scope": scope,
            "value": value,
            **metadata
        }) + "\n")
        self.file.flush()

# Use it
telemetry = TelemetryContext(JsonFileReporter("/var/log/telemetry.jsonl"))
```

## OpenTelemetry Export

```python
from nullscope import TelemetryContext
from nullscope.adapters.opentelemetry import OTelReporter

# Configure your OTel SDK first (exporters, processors, etc.)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry import trace

provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)

# Then use Nullscope with OTel reporter
telemetry = TelemetryContext(OTelReporter(service_name="my-service"))

with telemetry("operation"):
    do_work()  # Creates OTel span + histogram metric
```

## Multiple Reporters

```python
from nullscope import TelemetryContext, SimpleReporter
from nullscope.adapters.opentelemetry import OTelReporter

# Send to multiple destinations
telemetry = TelemetryContext(
    SimpleReporter(),  # In-memory for debugging
    OTelReporter(),    # Export to OTel backend
)
```

## Reporter Lifecycle

```python
from nullscope import TelemetryContext

telemetry = TelemetryContext(...)

# Flush any buffered reporter state before process exits
telemetry.flush()

# Shutdown reporters with explicit lifecycle cleanup
telemetry.shutdown()
```

## Async Context Support

Nullscope uses context variables, so it works correctly with async code:

```python
import asyncio
from nullscope import TelemetryContext

telemetry = TelemetryContext()

async def fetch_user(user_id: int):
    with telemetry("fetch_user", user_id=user_id):
        await asyncio.sleep(0.1)  # Simulate I/O
        return {"id": user_id}

async def main():
    with telemetry("batch_fetch"):
        # Each task maintains its own scope stack
        users = await asyncio.gather(
            fetch_user(1),
            fetch_user(2),
            fetch_user(3),
        )
```

Each task keeps isolated context; nested task scopes do not leak across concurrent work.

## Conditional Metrics

```python
telemetry = TelemetryContext()

with telemetry("process_order") as scope:
    order = get_order()

    if order.is_premium:
        telemetry.count("premium_orders")

    if order.total > 1000:
        telemetry.gauge("high_value_order", order.total)

    process(order)
```

## Testing with SimpleReporter

```python
import os
import importlib
from unittest.mock import patch
from nullscope import TelemetryContext, SimpleReporter

def test_my_function():
    with patch.dict(os.environ, {"NULLSCOPE_ENABLED": "1"}):
        import nullscope
        importlib.reload(nullscope)

        reporter = SimpleReporter()
        telemetry = nullscope.TelemetryContext(reporter)

        with telemetry("my_function"):
            result = my_function()

        # Assert on collected data
        assert "my_function" in reporter.timings
        timing_data = reporter.timings["my_function"][0]
        duration, metadata = timing_data
        assert duration > 0
        assert metadata["depth"] == 0
```
