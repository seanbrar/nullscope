import os
from unittest.mock import MagicMock, patch


def test_nullscope_disabled_by_default() -> None:
    """Verify strictly no-op when env var is missing."""
    with patch.dict(os.environ, {}, clear=True):
        # We need to reload the module because the flag is checked at import time
        import importlib

        import nullscope
        importlib.reload(nullscope)

        ctx = nullscope.TelemetryContext()
        assert isinstance(ctx, nullscope._NoOpTelemetryContext)
        assert ctx.is_enabled is False

        # Verify call returns self
        assert ctx("scope") is ctx

        # Verify attributes checks don't crash
        ctx.metric("foo", 1)

def test_nullscope_enabled() -> None:
    """Verify enabled context when env var is set."""
    with patch.dict(os.environ, {"NULLSCOPE_ENABLED": "1"}):
        import importlib

        import nullscope
        importlib.reload(nullscope)

        ctx = nullscope.TelemetryContext()
        assert isinstance(ctx, nullscope._EnabledTelemetryContext)
        assert ctx.is_enabled is True

        # Verify scope creation
        with ctx("scope") as scope:
            assert scope is ctx

def test_nullscope_recording() -> None:
    """Verify that reporters receive data."""
    mock_reporter = MagicMock()

    with patch.dict(os.environ, {"NULLSCOPE_ENABLED": "1"}):
        import importlib

        import nullscope
        importlib.reload(nullscope)

        ctx = nullscope.TelemetryContext(mock_reporter)

        with ctx("test.scope", extra="data"):
            pass

        assert mock_reporter.record_timing.called
        args, kwargs = mock_reporter.record_timing.call_args
        # scope (might imply hierarchy if stack existed, but here it's root)
        assert args[0] == "test.scope"
        assert isinstance(args[1], float) # duration
        assert kwargs["extra"] == "data"
        assert "start_wall_time_s" in kwargs

def test_nested_scopes() -> None:
    """Verify hierarchical path generation."""
    mock_reporter = MagicMock()
    with patch.dict(os.environ, {"NULLSCOPE_ENABLED": "1"}):
        import importlib

        import nullscope
        importlib.reload(nullscope)

        ctx = nullscope.TelemetryContext(mock_reporter)

        with ctx("parent"), ctx("child"):
            pass

        # Check call for child
        # Expected scope path: parent.child
        # We need to find the call where scope is parent.child
        child_calls = [
            c for c in mock_reporter.record_timing.mock_calls
            if c.args[0] == "parent.child"
        ]
        assert len(child_calls) == 1
