"""
Background worker that periodically runs Cognee's knowledge graph building pipeline.
Integrates with Agent Zero's DeferredTask system.
"""

import asyncio
import time
from typing import Set

from helpers.defer import DeferredTask, THREAD_BACKGROUND
from helpers.print_style import PrintStyle
from .cognee_init import get_cognee_setting


class CogneeBackgroundWorker:
    _instance = None

    def __init__(self) -> None:
        self._dirty_datasets: Set[str] = set()
        self._dirty_versions: dict[str, int] = {}
        self._insert_count: int = 0
        self._last_cognify_time: float = 0
        self._running: bool = False
        self._run_scheduled: bool = False
        self._last_error: str | None = None
        self._last_run_datasets: list[str] = []
        self._last_run_success: bool = False
        self._lock = asyncio.Lock()

    @classmethod
    def get_instance(cls) -> "CogneeBackgroundWorker":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def mark_dirty(self, dataset_name: str) -> None:
        """Mark a dataset as having new data."""
        self._dirty_datasets.add(dataset_name)
        self._dirty_versions[dataset_name] = self._dirty_versions.get(dataset_name, 0) + 1
        self._insert_count += 1
        self._schedule_run_soon()

    def get_status(self) -> dict:
        """Return current status for dashboard."""
        return {
            "running": self._running,
            "dirty_datasets": list(self._dirty_datasets),
            "insert_count": self._insert_count,
            "last_cognify_time": self._last_cognify_time,
            "last_run_datasets": self._last_run_datasets,
            "last_run_success": self._last_run_success,
            "last_error": self._last_error,
        }

    def nudge_rebuild_if_unready(self, datasets: list[str], reason: str = "") -> bool:
        """Schedule a rebuild when search runs before any clean cognify pass."""
        if self._last_run_success or self._running:
            return False

        clean_datasets = [dataset for dataset in datasets if dataset]
        if not clean_datasets:
            return False

        for dataset in clean_datasets:
            self.mark_dirty(dataset)

        detail = f": {reason}" if reason else ""
        PrintStyle.warning(
            f"Cognee graph search returned no context before a successful rebuild; "
            f"scheduled cognify for {clean_datasets}{detail}"
        )
        return True

    def _get_config(self) -> dict:
        """Load cognee-related settings."""
        return {
            "cognify_interval": get_cognee_setting("cognee_cognify_interval", 5),
            "cognify_after_n_inserts": get_cognee_setting("cognee_cognify_after_n_inserts", 10),
            "temporal_enabled": get_cognee_setting("cognee_temporal_enabled", True),
            "memify_enabled": get_cognee_setting("cognee_memify_enabled", True),
        }

    async def _should_run(self) -> bool:
        """Check if pipeline should run based on time and insert thresholds."""
        config = self._get_config()
        interval_minutes = config["cognify_interval"]
        insert_threshold = config["cognify_after_n_inserts"]

        if not self._dirty_datasets:
            return False

        time_elapsed_minutes = (time.monotonic() - self._last_cognify_time) / 60
        time_trigger = time_elapsed_minutes >= interval_minutes
        insert_trigger = self._insert_count >= insert_threshold

        return time_trigger or insert_trigger

    async def run_pipeline(self) -> None:
        """Run cognify + memify on dirty datasets."""
        if not self._dirty_datasets:
            return

        config = self._get_config()
        datasets = list(self._dirty_datasets)
        dataset_versions = {
            dataset: self._dirty_versions.get(dataset, 0)
            for dataset in datasets
        }

        try:
            import cognee
        except Exception as e:
            self._last_error = f"Cognee import failed: {e}"
            PrintStyle.error(f"Cognee background: cognee import failed: {e}")
            return

        async with self._lock:
            self._running = True
            self._last_error = None
            self._last_run_datasets = datasets
            try:
                if config["temporal_enabled"]:
                    await cognee.cognify(datasets=datasets, temporal_cognify=True)
                else:
                    await cognee.cognify(datasets=datasets, temporal_cognify=False)

                PrintStyle.standard(f"Cognee cognify completed for datasets: {datasets}")

                if config["memify_enabled"]:
                    for dataset in datasets:
                        try:
                            await cognee.improve(dataset=dataset)
                            PrintStyle.standard(f"Cognee improve completed for dataset: {dataset}")
                        except Exception as e:
                            if _is_empty_graph_improve_error(e):
                                PrintStyle.warning(
                                    f"Cognee improve skipped for {dataset}: graph is empty"
                                )
                                continue
                            PrintStyle.error(f"Cognee improve failed for {dataset}: {e}")
                            self._last_error = str(e)

                for dataset in datasets:
                    if self._dirty_versions.get(dataset, 0) == dataset_versions.get(dataset):
                        self._dirty_datasets.discard(dataset)
                        self._dirty_versions.pop(dataset, None)

                if not self._dirty_datasets:
                    self._insert_count = 0
                self._last_cognify_time = time.monotonic()
                self._last_run_success = True
            except Exception as e:
                self._last_error = str(e)
                self._last_run_success = False
                PrintStyle.error("Cognee pipeline failed", str(e))
            finally:
                self._running = False
                if self._dirty_datasets:
                    self._schedule_run_soon()

    async def maybe_run_pipeline(self) -> None:
        """Check if pipeline should run based on thresholds, then run if so."""
        if await self._should_run():
            await self.run_pipeline()

    async def run_loop(self) -> None:
        """Main background loop. Checks every 60 seconds if pipeline should run."""
        PrintStyle.standard("Cognee background worker started")
        while True:
            try:
                await self.maybe_run_pipeline()
            except Exception as e:
                self._last_error = str(e)
                PrintStyle.error("Cognee background worker error", str(e))
            await asyncio.sleep(60)

    def start(self) -> DeferredTask:
        """Start the background worker using DeferredTask. Returns the task for optional cleanup."""
        task = DeferredTask(thread_name=THREAD_BACKGROUND)
        task.start_task(self.run_loop)
        return task

    def _schedule_run_soon(self) -> None:
        """Debounce a near-immediate rebuild from the current event loop."""
        if self._run_scheduled or self._running:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        self._run_scheduled = True
        delay = float(get_cognee_setting("cognee_cognify_debounce_seconds", 2))
        loop.create_task(self._run_after_delay(delay))

    async def _run_after_delay(self, delay: float) -> None:
        try:
            if delay > 0:
                await asyncio.sleep(delay)
            self._run_scheduled = False
            await self.run_pipeline()
        finally:
            self._run_scheduled = False


def _is_empty_graph_improve_error(error: Exception) -> bool:
    message = str(error).lower()
    return "entitynotfounderror" in message and "empty graph projected" in message
