import asyncio
import importlib
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
