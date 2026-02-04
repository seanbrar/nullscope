"""OpenTelemetry adapter for Nullscope reporters.

This adapter intentionally keeps a small surface area:
- timings -> Histogram (seconds), optional synthetic Span
- counters -> Counter
- gauges -> Histogram (sampled values distribution)
"""

from collections.abc import Mapping
from typing import Any

from nullscope import TelemetryReporter

try:
    from opentelemetry import metrics, trace
except ModuleNotFoundError:  # pragma: no cover - exercised in runtime environments
    metrics = None  # type: ignore[assignment]
    trace = None  # type: ignore[assignment]


def _sanitize_attributes(
    attributes: Mapping[str, Any],
) -> dict[str, bool | str | bytes | int | float]:
    """Convert metadata into OpenTelemetry-compatible attribute values."""
    out: dict[str, bool | str | bytes | int | float] = {}
    for key, value in attributes.items():
        if value is None:
            continue
        if isinstance(value, bool | str | bytes | int | float):
            out[key] = value
            continue
        out[key] = str(value)
    return out


class OTelReporter(TelemetryReporter):
    """Minimal OpenTelemetry adapter for metrics-first telemetry export."""

    def __init__(self, service_name: str = "nullscope"):
        """Create a reporter bound to the given OpenTelemetry service name."""
        if trace is None or metrics is None:
            raise RuntimeError(
                "OpenTelemetry is not installed. "
                "Install optional dependency: nullscope[otel]",
            )
        self.tracer = trace.get_tracer(service_name)
        self.meter = metrics.get_meter(service_name)
        self._histograms: dict[tuple[str, str], Any] = {}
        self._counters: dict[str, Any] = {}

    def record_timing(self, scope: str, duration: float, **metadata: Any) -> None:
        """Record timing as a histogram sample and optional synthetic span."""
        attributes = _sanitize_attributes(metadata)
        self._get_timing_histogram(scope).record(duration, attributes=attributes)

        start_s = metadata.get("start_wall_time_s")
        end_s = metadata.get("end_wall_time_s")
        if isinstance(start_s, int | float) and isinstance(end_s, int | float):
            start_ns = int(start_s * 1e9)
            end_ns = int(end_s * 1e9)
            if end_ns >= start_ns:
                span = self.tracer.start_span(
                    scope,
                    start_time=start_ns,
                    attributes=attributes,
                )
                span.end(end_time=end_ns)

    def record_metric(self, scope: str, value: Any, **metadata: Any) -> None:
        """Record `counter` and `gauge` metrics using OpenTelemetry instruments."""
        metric_type = metadata.get("metric_type", "counter")
        attributes = _sanitize_attributes(metadata)

        if metric_type == "counter" and isinstance(value, int | float) and value >= 0:
            self._get_counter(scope).add(value, attributes=attributes)
            return

        if metric_type == "gauge" and isinstance(value, int | float):
            # OTel sync Gauge support is limited in Python; record value samples instead.
            self._get_value_histogram(scope).record(value, attributes=attributes)

    def _get_timing_histogram(self, scope: str) -> Any:
        key = ("timing", scope)
        histogram = self._histograms.get(key)
        if histogram is None:
            histogram = self.meter.create_histogram(scope, unit="s")
            self._histograms[key] = histogram
        return histogram

    def _get_value_histogram(self, scope: str) -> Any:
        key = ("value", scope)
        histogram = self._histograms.get(key)
        if histogram is None:
            histogram = self.meter.create_histogram(scope)
            self._histograms[key] = histogram
        return histogram

    def _get_counter(self, scope: str) -> Any:
        counter = self._counters.get(scope)
        if counter is None:
            counter = self.meter.create_counter(scope)
            self._counters[scope] = counter
        return counter
