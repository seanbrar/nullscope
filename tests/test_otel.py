from unittest.mock import MagicMock

import pytest

pytest.importorskip("opentelemetry")

from nullscope.adapters.opentelemetry import OTelReporter


def test_otel_reporter_timing() -> None:
    reporter = OTelReporter()
    reporter.tracer = MagicMock()
    reporter.meter = MagicMock()
    reporter._histograms = {}  # Reset

    metadata = {
        "start_wall_time_s": 100.0,
        "end_wall_time_s": 101.0,
        "custom": "value"
    }

    # Mock histogram
    mock_hist = MagicMock()
    reporter.meter.create_histogram.return_value = mock_hist

    reporter.record_timing("my.scope", 1.0, **metadata)

    # Verify histogram record
    reporter.meter.create_histogram.assert_called_with("my.scope", unit="s")
    mock_hist.record.assert_called_with(1.0, attributes=metadata)

    # Verify span creation
    reporter.tracer.start_span.assert_called()
    _, kwargs = reporter.tracer.start_span.call_args
    assert kwargs["start_time"] == 100_000_000_000
    assert kwargs["attributes"] == metadata

    span = reporter.tracer.start_span.return_value
    span.end.assert_called_with(end_time=101_000_000_000)

def test_otel_reporter_metric() -> None:
    reporter = OTelReporter()
    reporter.meter = MagicMock()
    reporter._counters = {}

    mock_counter = MagicMock()
    reporter.meter.create_counter.return_value = mock_counter

    reporter.record_metric("my.counter", 5, metric_type="counter", tag="foo")

    reporter.meter.create_counter.assert_called_with("my.counter")
    mock_counter.add.assert_called_with(5, attributes={"metric_type": "counter", "tag": "foo"})


def test_otel_reporter_gauge_maps_to_histogram() -> None:
    reporter = OTelReporter()
    reporter.meter = MagicMock()
    reporter._histograms = {}

    mock_hist = MagicMock()
    reporter.meter.create_histogram.return_value = mock_hist

    reporter.record_metric("my.gauge", 12.3, metric_type="gauge")

    reporter.meter.create_histogram.assert_called_with("my.gauge")
    mock_hist.record.assert_called_with(12.3, attributes={"metric_type": "gauge"})


def test_otel_reporter_negative_counter_is_ignored() -> None:
    reporter = OTelReporter()
    reporter.meter = MagicMock()
    reporter._counters = {}

    reporter.record_metric("my.counter", -1, metric_type="counter")

    reporter.meter.create_counter.assert_not_called()
