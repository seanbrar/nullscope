# Nullscope vs Alternatives

When to use Nullscope versus other observability tools.

## vs OpenTelemetry SDK

**OpenTelemetry** is the industry standard for observability. It provides comprehensive tracing, metrics, and logging with a rich ecosystem of exporters and integrations.

| Aspect | Nullscope | OpenTelemetry SDK |
|--------|-----------|-------------------|
| Zero-cost when disabled | Yes (singleton no-op) | No (still creates spans/contexts) |
| Setup complexity | Minimal | Moderate to high |
| Ecosystem | Limited | Extensive |
| Protocol compliance | None | OTLP standard |
| Best for | Application code, libraries | Production observability |

**Use Nullscope when:**
- You're writing a library and want optional telemetry
- You need truly zero overhead in production
- You want simple, focused timing/metrics without full tracing
- You'll export to OTel anyway (via the adapter)

**Use OpenTelemetry directly when:**
- You need distributed tracing across services
- You want baggage propagation, trace context, etc.
- You're building production infrastructure
- You need W3C Trace Context compliance

**Best of both worlds:** Use Nullscope in your application code with `OTelReporter` to get zero-cost no-op behavior with OTel export when enabled.

## vs structlog

**structlog** is a structured logging library that makes logs more useful with context and formatting.

| Aspect | Nullscope | structlog |
|--------|-----------|-----------|
| Primary purpose | Timing and metrics | Structured logging |
| Output format | Reporter-defined | Log events |
| Context tracking | Scope hierarchy | Bound loggers |
| Zero-cost disable | Yes | No (logs are written) |

**Use Nullscope when:**
- You need timing data, not log events
- You want to record numeric metrics
- Zero overhead when disabled is critical

**Use structlog when:**
- You need rich, structured log output
- You want contextual logging throughout your app
- You're building audit trails or debug logs

**They complement each other:** Use structlog for logging, Nullscope for metrics/timing. They solve different problems.

## vs prometheus_client

**prometheus_client** is the official Python client for Prometheus metrics.

| Aspect | Nullscope | prometheus_client |
|--------|-----------|-------------------|
| Metric types | Counters, gauges, timings | Full Prometheus types |
| Scope hierarchy | Built-in | Manual labels |
| Zero-cost disable | Yes | No |
| Export format | Reporter-defined | Prometheus exposition |
| Spans/timing | Yes (with wall clock) | Histograms only |

**Use Nullscope when:**
- You want automatic scope hierarchy
- Zero overhead when disabled matters
- You're not running Prometheus

**Use prometheus_client when:**
- You have a Prometheus infrastructure
- You need summaries, histograms with buckets
- You want direct Prometheus integration

## vs timing decorators / manual timing

Many projects use simple timing:

```python
import time
start = time.perf_counter()
do_work()
duration = time.perf_counter() - start
```

| Aspect | Nullscope | Manual timing |
|--------|-----------|---------------|
| Zero-cost disable | Yes | No (unless you add conditionals) |
| Scope hierarchy | Automatic | Manual |
| Multiple reporters | Built-in | DIY |
| Async safety | Context vars | DIY |

**Use Nullscope when:**
- You want consistent telemetry patterns
- You need hierarchy tracking
- You might disable telemetry in production

**Use manual timing when:**
- You have one or two timing points
- You don't need hierarchy
- Simplicity is paramount

## Summary: When to Choose Nullscope

Choose Nullscope if you value:

1. **Zero overhead** when telemetry is disabled
2. **Simple API** for timing and metrics
3. **Automatic hierarchy** for nested operations
4. **Library-friendly** design (no global state pollution)
5. **Async-safe** out of the box

Consider alternatives if you need:

1. **Full distributed tracing** → OpenTelemetry
2. **Structured logging** → structlog
3. **Prometheus-native metrics** → prometheus_client
4. **One-off timing** → manual `time.perf_counter()`
