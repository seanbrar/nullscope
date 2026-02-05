import asyncio
import importlib
import os
from types import ModuleType
from unittest.mock import patch


def _reload_nullscope() -> ModuleType:
    import nullscope

    return importlib.reload(nullscope)


def test_integration_runtime_contract() -> None:
    with patch.dict(os.environ, {"NULLSCOPE_ENABLED": "1"}, clear=True):
        nullscope = _reload_nullscope()
        reporter = nullscope.SimpleReporter()
        telemetry = nullscope.TelemetryContext(reporter)

        async def worker(worker_id: int) -> None:
            with telemetry("task", worker=worker_id):
                telemetry.count("items", 1, worker=worker_id)
                await asyncio.sleep(0)
                with telemetry("leaf", worker=worker_id):
                    pass

        async def run_batch() -> None:
            with telemetry("batch", req_id="r1"):
                await asyncio.gather(worker(1), worker(2))

        asyncio.run(run_batch())

        # Timing path contract
        assert "batch.task.leaf" in reporter.timings
        assert len(reporter.timings["batch.task.leaf"]) == 2

        # Metric path contract
        assert "batch.task.items" in reporter.metrics
        values = [value for value, _ in reporter.metrics["batch.task.items"]]
        assert values == [1, 1]

        # Core metadata contract
        _, metadata = reporter.timings["batch.task.leaf"][0]
        assert metadata["parent_scope"] == "batch.task"
        assert "start_wall_time_s" in metadata
        assert "end_wall_time_s" in metadata
