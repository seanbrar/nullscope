import asyncio
import importlib
import logging
import os
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest


def _reload_nullscope() -> ModuleType:
    import nullscope

    return importlib.reload(nullscope)


def test_nullscope_disabled_by_default() -> None:
    with patch.dict(os.environ, {}, clear=True):
        nullscope = _reload_nullscope()

        ctx = nullscope.TelemetryContext()
        assert isinstance(ctx, nullscope._NoOpTelemetryContext)
        assert ctx.is_enabled is False
        assert ctx("scope") is ctx
        ctx.metric("foo", 1)


def test_nullscope_enabled() -> None:
    with patch.dict(os.environ, {"NULLSCOPE_ENABLED": "1"}):
        nullscope = _reload_nullscope()

        ctx = nullscope.TelemetryContext()
        assert isinstance(ctx, nullscope._EnabledTelemetryContext)
        assert ctx.is_enabled is True

        with ctx("scope") as scope:
            assert scope is ctx


def test_nullscope_recording() -> None:
    mock_reporter = MagicMock()

    with patch.dict(os.environ, {"NULLSCOPE_ENABLED": "1"}):
        nullscope = _reload_nullscope()
        ctx = nullscope.TelemetryContext(mock_reporter)

        with ctx("test.scope", extra="data"):
            pass

        assert mock_reporter.record_timing.called
        args, kwargs = mock_reporter.record_timing.call_args
        assert args[0] == "test.scope"
        assert isinstance(args[1], float)
        assert kwargs["extra"] == "data"
        assert "start_wall_time_s" in kwargs


def test_nested_scopes() -> None:
    mock_reporter = MagicMock()
    with patch.dict(os.environ, {"NULLSCOPE_ENABLED": "1"}):
        nullscope = _reload_nullscope()
        ctx = nullscope.TelemetryContext(mock_reporter)

        with ctx("parent"), ctx("child"):
            pass

        child_calls = [
            c for c in mock_reporter.record_timing.mock_calls if c.args[0] == "parent.child"
        ]
        assert len(child_calls) == 1


def test_timed_decorator_records_sync_functions() -> None:
    with patch.dict(os.environ, {"NULLSCOPE_ENABLED": "1"}):
        nullscope = _reload_nullscope()

        reporter = nullscope.SimpleReporter()
        telemetry = nullscope.TelemetryContext(reporter)

        @telemetry.timed("decorated.sync", endpoint="/health")  # type: ignore[untyped-decorator]
        def run() -> int:
            return 200

        assert run() == 200
        assert "decorated.sync" in reporter.timings
        _, metadata = reporter.timings["decorated.sync"][0]
        assert metadata["endpoint"] == "/health"


def test_timed_decorator_is_identity_when_disabled() -> None:
    with patch.dict(os.environ, {}, clear=True):
        nullscope = _reload_nullscope()
        telemetry = nullscope.TelemetryContext()

        def run() -> int:
            return 7

        decorated = telemetry.timed("noop.scope")(run)
        assert decorated is run
        assert decorated() == 7


def test_timed_decorator_records_async_functions() -> None:
    with patch.dict(os.environ, {"NULLSCOPE_ENABLED": "1"}):
        nullscope = _reload_nullscope()

        reporter = nullscope.SimpleReporter()
        telemetry = nullscope.TelemetryContext(reporter)

        @telemetry.timed("decorated.async")  # type: ignore[untyped-decorator]
        async def run() -> str:
            await asyncio.sleep(0)
            return "ok"

        assert asyncio.run(run()) == "ok"
        assert "decorated.async" in reporter.timings


def test_async_scope_stack_is_isolated_by_task() -> None:
    with patch.dict(os.environ, {"NULLSCOPE_ENABLED": "1"}):
        nullscope = _reload_nullscope()

        reporter = nullscope.SimpleReporter()
        telemetry = nullscope.TelemetryContext(reporter)

        async def worker(task_id: int) -> None:
            with telemetry("task", task_id=task_id):
                await asyncio.sleep(0)
                with telemetry("leaf", task_id=task_id):
                    await asyncio.sleep(0)

        async def run() -> None:
            with telemetry("batch"):
                await asyncio.gather(worker(1), worker(2))

        asyncio.run(run())
        assert len(reporter.timings["batch.task.leaf"]) == 2
        parent_scopes = [meta["parent_scope"] for _, meta in reporter.timings["batch.task.leaf"]]
        assert parent_scopes == ["batch.task", "batch.task"]


def test_reporter_lifecycle_hooks_are_forwarded() -> None:
    reporter = MagicMock()
    with patch.dict(os.environ, {"NULLSCOPE_ENABLED": "1"}):
        nullscope = _reload_nullscope()
        telemetry = nullscope.TelemetryContext(reporter)

        telemetry.flush()
        telemetry.shutdown()

        reporter.flush.assert_called_once_with()
        reporter.shutdown.assert_called_once_with()


def test_invalid_reporter_error_is_explicit() -> None:
    class BrokenReporter:
        def record_timing(self, scope: str, duration: float, **metadata: object) -> None:
            return None

    with patch.dict(os.environ, {"NULLSCOPE_ENABLED": "1"}):
        nullscope = _reload_nullscope()

        with pytest.raises(TypeError, match="Missing required callable method"):
            nullscope.TelemetryContext(BrokenReporter())


def test_timed_rejects_invalid_scope_names() -> None:
    with patch.dict(os.environ, {"NULLSCOPE_ENABLED": "1", "NULLSCOPE_STRICT": "1"}):
        nullscope = _reload_nullscope()
        telemetry = nullscope.TelemetryContext()

        with pytest.raises(ValueError, match="Invalid Decorator scope name"):
            telemetry.timed("Invalid.Scope")


def test_report_shows_nested_and_parent_scopes() -> None:
    with patch.dict(os.environ, {"NULLSCOPE_ENABLED": "1"}):
        nullscope = _reload_nullscope()
        reporter = nullscope.SimpleReporter()
        telemetry = nullscope.TelemetryContext(reporter)

        with telemetry("request"), telemetry("auth"):
            pass

        report = reporter.get_report()
        assert "request" in report
        assert "auth" in report
        # Parent scope should show its own stats, not just be a header
        assert report.count("Calls:") == 2


def test_report_shows_structural_headers() -> None:
    with patch.dict(os.environ, {"NULLSCOPE_ENABLED": "1"}):
        nullscope = _reload_nullscope()
        reporter = nullscope.SimpleReporter()
        telemetry = nullscope.TelemetryContext(reporter)

        # Only record "a.b.c" — no "a" or "a.b" scopes
        with telemetry("a"), telemetry("b"), telemetry("c"):
            pass

        # Manually clear parent scopes to simulate only having "a.b.c"
        reporter.timings = {k: v for k, v in reporter.timings.items() if k == "a.b.c"}

        report = reporter.get_report()
        # "a:" and "b:" should appear as structural headers
        assert "a:" in report
        assert "b:" in report
        assert "Calls:" in report


def test_as_dict_returns_dicts_with_named_keys() -> None:
    with patch.dict(os.environ, {"NULLSCOPE_ENABLED": "1"}):
        nullscope = _reload_nullscope()
        reporter = nullscope.SimpleReporter()
        telemetry = nullscope.TelemetryContext(reporter)

        with telemetry("op"):
            telemetry.count("items", 3)

        data = reporter.as_dict()
        # Timings entries should be dicts with "duration" key
        timing_entry = data["timings"]["op"][0]
        assert "duration" in timing_entry
        assert isinstance(timing_entry["duration"], float)
        assert "depth" in timing_entry

        # Metrics entries should be dicts with "value" key
        metric_entry = data["metrics"]["op.items"][0]
        assert "value" in metric_entry
        assert metric_entry["value"] == 3


def test_report_gauge_shows_last_value() -> None:
    with patch.dict(os.environ, {"NULLSCOPE_ENABLED": "1"}):
        nullscope = _reload_nullscope()
        reporter = nullscope.SimpleReporter()
        telemetry = nullscope.TelemetryContext(reporter)

        telemetry.gauge("queue.depth", 42)
        telemetry.gauge("queue.depth", 38)

        report = reporter.get_report()
        assert "Last: 38" in report
        # Should NOT show "Total: 80"
        assert "Total: 80" not in report


def test_exception_metadata_recorded_on_scope_exit() -> None:
    with patch.dict(os.environ, {"NULLSCOPE_ENABLED": "1"}):
        nullscope = _reload_nullscope()
        reporter = nullscope.SimpleReporter()
        telemetry = nullscope.TelemetryContext(reporter)

        with pytest.raises(ValueError, match="bad"), telemetry("op"):
            raise ValueError("bad")

        # Exception should have propagated (not swallowed)
        _, metadata = reporter.timings["op"][0]
        assert metadata["error"] is True
        assert metadata["error_type"] == "ValueError"
        assert metadata["error_message"] == "bad"


def test_exception_metadata_absent_on_success() -> None:
    with patch.dict(os.environ, {"NULLSCOPE_ENABLED": "1"}):
        nullscope = _reload_nullscope()
        reporter = nullscope.SimpleReporter()
        telemetry = nullscope.TelemetryContext(reporter)

        with telemetry("op"):
            pass

        _, metadata = reporter.timings["op"][0]
        assert "error" not in metadata


def test_log_reporter_emits_timing_record() -> None:
    with patch.dict(os.environ, {"NULLSCOPE_ENABLED": "1"}):
        nullscope = _reload_nullscope()
        logger = MagicMock()
        reporter = nullscope.LogReporter(logger=logger, level=logging.INFO)
        telemetry = nullscope.TelemetryContext(reporter)

        with telemetry("op"):
            pass

        logger.log.assert_called()
        args, kwargs = logger.log.call_args
        assert args[0] == logging.INFO
        assert "timing" in args[1]
        assert "scope=" in args[1]
        assert "telemetry_metadata" in kwargs["extra"]
