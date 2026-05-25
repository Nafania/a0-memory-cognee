"""
Background worker that periodically runs Cognee's knowledge graph building pipeline.
Integrates with Agent Zero's DeferredTask system.
"""

import asyncio
import threading
import time
from typing import Set

from helpers.defer import DeferredTask, THREAD_BACKGROUND
from helpers.print_style import PrintStyle
from .cognee_graph import read_dataset_graphs
from .cognee_init import get_cognee_setting
from .cognee_ops import run_cognee_operation


class CogneeBackgroundWorker:
    _instance = None
    _instance_lock = threading.Lock()

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
        self._dataset_readiness: dict[str, dict[str, object]] = {}
        self._retry_attempts: dict[str, int] = {}
        self._needs_pipeline_reset: Set[str] = set()
        self._task: DeferredTask | None = None
        self._state_lock = threading.RLock()

    @classmethod
    def get_instance(cls) -> "CogneeBackgroundWorker":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    def mark_dirty(self, dataset_name: str, *, reset_retry: bool = True) -> None:
        """Mark a dataset as having new data."""
        with self._state_lock:
            self._dirty_datasets.add(dataset_name)
            self._dirty_versions[dataset_name] = self._dirty_versions.get(dataset_name, 0) + 1
            if reset_retry:
                self._retry_attempts.pop(dataset_name, None)
            self._insert_count += 1
            self._set_dataset_state_locked(
                dataset_name,
                "dirty",
                "Cognee memory graph rebuild pending",
            )
        self._schedule_run_soon()

    def get_status(self) -> dict:
        """Return current status for dashboard."""
        self._refresh_stale_rebuilds()
        with self._state_lock:
            return {
                "running": self._running,
                "dirty_datasets": list(self._dirty_datasets),
                "insert_count": self._insert_count,
                "last_cognify_time": self._last_cognify_time,
                "last_run_datasets": self._last_run_datasets,
                "last_run_success": self._last_run_success,
                "last_error": self._last_error,
                "dataset_readiness": {
                    dataset: dict(state)
                    for dataset, state in self._dataset_readiness.items()
                },
                "retry_attempts": dict(self._retry_attempts),
                "pipeline_reset_datasets": sorted(self._needs_pipeline_reset),
            }

    def get_search_block_reason(self, datasets: list[str]) -> str | None:
        """Return why graph search must wait, or None when datasets are searchable."""
        self._refresh_stale_rebuilds()
        clean_datasets = [dataset for dataset in datasets if dataset]
        if not clean_datasets:
            return None

        pending: list[str] = []
        rebuilding: list[str] = []
        failed: list[str] = []
        with self._state_lock:
            for dataset in clean_datasets:
                state = str(
                    self._dataset_readiness.get(dataset, {}).get("state") or ""
                )
                if not state or state == "ready":
                    continue
                if state == "rebuilding":
                    rebuilding.append(dataset)
                elif state == "failed":
                    failed.append(dataset)
                else:
                    pending.append(dataset)

        if rebuilding:
            return f"Cognee memory graph rebuild running for dataset(s): {rebuilding}"
        if pending:
            return f"Cognee memory graph rebuild pending for dataset(s): {pending}"
        if failed:
            failure_reasons = []
            with self._state_lock:
                for dataset in failed[:3]:
                    reason = self._dataset_readiness.get(dataset, {}).get("reason")
                    if reason:
                        failure_reasons.append(f"{dataset}: {reason}")
            detail = f". Reason: {'; '.join(failure_reasons)}" if failure_reasons else ""
            return f"Cognee memory graph rebuild failed for dataset(s): {failed}{detail}"
        return None

    def _set_dataset_state_locked(
        self,
        dataset_name: str,
        state: str,
        reason: str | None = None,
    ) -> None:
        self._dataset_readiness[dataset_name] = {
            "state": state,
            "reason": reason,
            "updated_at": time.monotonic(),
        }

    def _refresh_stale_rebuilds(self) -> None:
        """Expire rebuild states that outlived the worker operation timeout path."""
        config = self._get_config()
        stale_after = float(config["rebuild_stale_after"])
        if stale_after <= 0:
            return

        stale_rebuilds: list[tuple[str, str]] = []
        should_schedule_retry = False
        with self._state_lock:
            now = time.monotonic()
            for dataset, readiness in list(self._dataset_readiness.items()):
                if readiness.get("state") != "rebuilding":
                    continue

                updated_at = readiness.get("updated_at")
                if not isinstance(updated_at, (int, float)):
                    continue

                age_seconds = now - float(updated_at)
                if age_seconds < stale_after:
                    continue

                if self._running:
                    detail = (
                        "worker still reports running; stale running flag cleared "
                        "and retry scheduled"
                    )
                    self._running = False
                else:
                    detail = "retry scheduled"
                self._run_scheduled = False
                should_schedule_retry = True

                reason = (
                    "Cognee memory graph rebuild state is stale "
                    f"after {age_seconds:.0f}s ({detail})"
                )
                self._dirty_datasets.add(dataset)
                self._needs_pipeline_reset.add(dataset)
                self._last_error = reason
                self._last_run_success = False
                self._set_dataset_state_locked(dataset, "failed", reason)
                stale_rebuilds.append((dataset, reason))

        for dataset, reason in stale_rebuilds:
            PrintStyle.warning(
                f"Cognee rebuild state expired for dataset {dataset}: {reason}"
            )

        if should_schedule_retry:
            self._schedule_run_soon(float(config["retry_min_delay"]))

    def _mark_unfinished_rebuilds_failed_locked(
        self,
        retry_delay: float,
    ) -> list[str]:
        unfinished: list[str] = []
        for dataset in self._last_run_datasets:
            readiness = self._dataset_readiness.get(dataset, {})
            if readiness.get("state") != "rebuilding":
                continue

            reason = (
                "Cognee memory graph rebuild interrupted before readiness update; "
                f"retry scheduled in {retry_delay:.0f}s"
            )
            self._dirty_datasets.add(dataset)
            self._needs_pipeline_reset.add(dataset)
            self._last_error = reason
            self._last_run_success = False
            self._set_dataset_state_locked(dataset, "failed", reason)
            unfinished.append(dataset)
        return unfinished

    def nudge_rebuild_if_unready(self, datasets: list[str], reason: str = "") -> bool:
        """Legacy empty-search hook.

        Empty search results are not enough to prove a graph is stale, so this
        intentionally does not schedule rebuilds. Startup graph checks and
        explicit dirty marks own rebuild state.
        """
        clean_datasets = [dataset for dataset in datasets if dataset]
        if not clean_datasets:
            return False

        with self._state_lock:
            if self._running:
                return False
        return False

    def _get_config(self) -> dict:
        """Load cognee-related settings."""
        return {
            "cognify_interval": get_cognee_setting("cognee_cognify_interval", 5),
            "cognify_after_n_inserts": get_cognee_setting("cognee_cognify_after_n_inserts", 10),
            "temporal_enabled": get_cognee_setting("cognee_temporal_enabled", True),
            "memify_enabled": get_cognee_setting("cognee_memify_enabled", True),
            "retry_min_delay": get_cognee_setting("cognee_rebuild_retry_min_seconds", 30),
            "retry_max_delay": get_cognee_setting("cognee_rebuild_retry_max_seconds", 300),
            "operation_timeout": get_cognee_setting("cognee_operation_timeout_seconds", 1800),
            "rebuild_stale_after": get_cognee_setting("cognee_rebuild_stale_after_seconds", 3600),
        }

    async def _should_run(self) -> bool:
        """Check if pipeline should run based on time and insert thresholds."""
        config = self._get_config()
        interval_minutes = config["cognify_interval"]
        insert_threshold = config["cognify_after_n_inserts"]

        with self._state_lock:
            dirty_count = len(self._dirty_datasets)
            insert_count = self._insert_count
            last_cognify_time = self._last_cognify_time

        if not dirty_count:
            return False

        time_elapsed_minutes = (time.monotonic() - last_cognify_time) / 60
        time_trigger = time_elapsed_minutes >= interval_minutes
        insert_trigger = insert_count >= insert_threshold

        return time_trigger or insert_trigger

    async def run_pipeline(self) -> None:
        """Run cognify + memify on dirty datasets."""
        should_reschedule = False
        reschedule_delay: float | None = None
        with self._state_lock:
            if self._running or not self._dirty_datasets:
                return

            self._running = True
            self._last_error = None
            datasets = sorted(self._dirty_datasets)
            dataset_versions = {
                dataset: self._dirty_versions.get(dataset, 0)
                for dataset in datasets
            }
            self._last_run_datasets = datasets
            dataset_states = {
                dataset: dict(self._dataset_readiness.get(dataset, {}))
                for dataset in datasets
            }
            pipeline_reset_datasets = set(self._needs_pipeline_reset)
            for dataset in datasets:
                self._set_dataset_state_locked(
                    dataset,
                    "rebuilding",
                    "Cognee memory graph rebuild running",
                )

        config = self._get_config()

        try:
            import cognee
        except Exception as e:
            config = self._get_config()
            retry_delay = float(config["retry_min_delay"])
            with self._state_lock:
                error = f"Cognee import failed: {e}"
                self._last_error = error
                self._last_run_success = False
                for dataset in datasets:
                    self._dirty_datasets.add(dataset)
                    self._set_dataset_state_locked(
                        dataset,
                        "failed",
                        f"{error}; retry scheduled in {retry_delay:.0f}s",
                    )
                self._running = False
                should_reschedule = bool(self._dirty_datasets)
            PrintStyle.error(f"Cognee background: cognee import failed: {e}")
            if should_reschedule:
                self._schedule_run_soon(retry_delay)
            return

        try:
            failed_datasets: list[str] = []
            retry_delays: list[float] = []
            for dataset in datasets:
                try:
                    previous_state = str(
                        dataset_states.get(dataset, {}).get("state") or ""
                    )
                    needs_pipeline_reset = (
                        previous_state == "failed"
                        or dataset in pipeline_reset_datasets
                    )
                    if needs_pipeline_reset:
                        await _reset_pipeline_status_for_rebuild(dataset)

                    if config["temporal_enabled"]:
                        await run_cognee_operation(
                            "cognee.cognify background",
                            cognee.cognify,
                            datasets=[dataset],
                            temporal_cognify=True,
                            operation_timeout=float(config["operation_timeout"]),
                        )
                    else:
                        await run_cognee_operation(
                            "cognee.cognify background",
                            cognee.cognify,
                            datasets=[dataset],
                            temporal_cognify=False,
                            operation_timeout=float(config["operation_timeout"]),
                        )

                    PrintStyle.standard(f"Cognee cognify completed for dataset: {dataset}")

                    readiness_error = await _verify_cognify_ready(cognee, [dataset])
                    if readiness_error:
                        raise RuntimeError(
                            "Cognee cognify completed but "
                            f"{readiness_error} for dataset: {dataset}"
                        )

                    if config["memify_enabled"]:
                        try:
                            await run_cognee_operation(
                                "cognee.improve background",
                                cognee.improve,
                                dataset=dataset,
                                operation_timeout=float(config["operation_timeout"]),
                            )
                            PrintStyle.standard(f"Cognee improve completed for dataset: {dataset}")
                        except Exception as e:
                            if _is_empty_graph_improve_error(e):
                                PrintStyle.warning(
                                    f"Cognee improve skipped for {dataset}: graph is empty"
                                )
                            else:
                                raise

                        readiness_error = await _verify_cognify_ready(cognee, [dataset])
                        if readiness_error:
                            raise RuntimeError(
                                "Cognee improve completed but "
                                f"{readiness_error} for dataset: {dataset}"
                            )
                    with self._state_lock:
                        self._retry_attempts.pop(dataset, None)
                except Exception as e:
                    failed_datasets.append(dataset)
                    with self._state_lock:
                        self._needs_pipeline_reset.add(dataset)
                        attempt = self._retry_attempts.get(dataset, 0) + 1
                        self._retry_attempts[dataset] = attempt
                        retry_delay = min(
                            float(config["retry_max_delay"]),
                            float(config["retry_min_delay"]) * (2 ** (attempt - 1)),
                        )
                        retry_delays.append(retry_delay)
                        self._last_error = str(e)
                        self._set_dataset_state_locked(
                            dataset,
                            "failed",
                            f"{e}; retry scheduled in {retry_delay:.0f}s",
                        )
                    PrintStyle.error(f"Cognee pipeline failed for dataset {dataset}", str(e))

            with self._state_lock:
                for dataset in datasets:
                    if dataset in failed_datasets:
                        continue
                    if self._dirty_versions.get(dataset, 0) == dataset_versions.get(dataset):
                        self._dirty_datasets.discard(dataset)
                        self._dirty_versions.pop(dataset, None)
                        self._needs_pipeline_reset.discard(dataset)
                        self._set_dataset_state_locked(
                            dataset,
                            "ready",
                            "Cognee memory graph rebuild completed",
                        )
                    else:
                        self._set_dataset_state_locked(
                            dataset,
                            "dirty",
                            "Cognee memory graph changed during rebuild",
                        )

                if failed_datasets:
                    should_reschedule = bool(self._dirty_datasets)
                    reschedule_delay = max(retry_delays) if retry_delays else None
                    self._last_run_success = False
                elif self._dirty_datasets:
                    should_reschedule = True
                    self._last_run_success = True
                else:
                    should_reschedule = False
                    self._insert_count = 0
                    self._last_run_success = True
                self._last_cognify_time = time.monotonic()
        except Exception as e:
            with self._state_lock:
                self._last_error = str(e)
                self._last_run_success = False
                for dataset in self._last_run_datasets:
                    self._needs_pipeline_reset.add(dataset)
                    self._set_dataset_state_locked(dataset, "failed", str(e))
                should_reschedule = False
                reschedule_delay = None
            PrintStyle.error("Cognee pipeline failed", str(e))
        finally:
            unfinished_datasets: list[str] = []
            with self._state_lock:
                retry_delay = float(config["retry_min_delay"])
                unfinished_datasets = self._mark_unfinished_rebuilds_failed_locked(
                    retry_delay,
                )
                if unfinished_datasets:
                    should_reschedule = True
                    reschedule_delay = max(
                        retry_delay,
                        float(reschedule_delay or 0),
                    )
                self._running = False
            for dataset in unfinished_datasets:
                PrintStyle.error(
                    "Cognee pipeline did not complete readiness update",
                    f"dataset={dataset}",
                )
            self._log_rebuild_readiness(
                retry_scheduled=should_reschedule,
                retry_delay=reschedule_delay,
            )
            if should_reschedule:
                self._schedule_run_soon(reschedule_delay)

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
                with self._state_lock:
                    self._last_error = str(e)
                PrintStyle.error("Cognee background worker error", str(e))
            await asyncio.sleep(60)

    def start(self) -> DeferredTask:
        """Start the background worker using DeferredTask. Returns the task for optional cleanup."""
        with self._state_lock:
            if self._task and self._task.is_alive():
                return self._task

            task = DeferredTask(thread_name=THREAD_BACKGROUND)
            task.start_task(self.run_loop)
            self._task = task
            return task

    def _schedule_run_soon(self, delay: float | None = None) -> None:
        """Debounce a near-immediate rebuild from the current event loop."""
        with self._state_lock:
            if self._run_scheduled or self._running:
                return
            self._run_scheduled = True
            if delay is None:
                delay = float(get_cognee_setting("cognee_cognify_debounce_seconds", 2))
            task = self._task

        loop = getattr(getattr(task, "event_loop_thread", None), "loop", None)
        if loop and loop.is_running():
            asyncio.run_coroutine_threadsafe(self._run_after_delay(delay), loop)
            return

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            with self._state_lock:
                self._run_scheduled = False
            return

        loop.create_task(self._run_after_delay(delay))

    async def _run_after_delay(self, delay: float) -> None:
        try:
            if delay > 0:
                await asyncio.sleep(delay)
            with self._state_lock:
                self._run_scheduled = False
            await self.run_pipeline()
        finally:
            with self._state_lock:
                self._run_scheduled = False

    def _log_rebuild_readiness(
        self,
        *,
        retry_scheduled: bool,
        retry_delay: float | None,
    ) -> None:
        with self._state_lock:
            datasets = list(self._last_run_datasets)
            dirty = sorted(self._dirty_datasets)
            running = self._running
            last_error = self._last_error
            blocked_states = []
            ready = []
            for dataset_name, state in self._dataset_readiness.items():
                state_name = str(state.get("state") or "")
                if state_name == "ready":
                    ready.append(dataset_name)
                elif state_name:
                    blocked_states.append(f"{dataset_name}:{state_name}")

        if not datasets:
            return

        if running or dirty or blocked_states:
            PrintStyle.warning(
                "Cognee rebuild readiness: BLOCKED; recall may be unavailable. "
                f"last_run_datasets={datasets}; ready={_short_list(sorted(ready))}; "
                f"dirty={dirty}; blocked_states={_short_list(sorted(blocked_states))}; "
                f"retry_scheduled={retry_scheduled}; retry_delay={retry_delay}; "
                f"last_error={last_error}"
            )
            return

        PrintStyle.standard(
            "Cognee rebuild readiness: READY; recall enabled. "
            f"last_run_datasets={datasets}; ready={_short_list(sorted(ready))}"
        )


def _is_empty_graph_improve_error(error: Exception) -> bool:
    message = str(error).lower()
    return "entitynotfounderror" in message and "empty graph projected" in message


async def _verify_cognify_ready(cognee, datasets: list[str]) -> str:
    """Return an error reason if cognify did not produce a readable graph."""
    dataset_graphs = await run_cognee_operation(
        "cognee.graph readiness",
        read_dataset_graphs,
        cognee,
        datasets,
        skip_empty_data=False,
        repair_unreadable=True,
    )
    if not dataset_graphs:
        return "graph readiness could not be verified"

    errors = [graph for graph in dataset_graphs if graph.error]
    if errors:
        details = "; ".join(
            f"{graph.dataset_name}: {graph.error}" for graph in errors[:3]
        )
        PrintStyle.warning(f"Could not read Cognee dataset graph(s): {details}")
        return "graph data could not be read"

    relevant_graphs = [
        graph
        for graph in dataset_graphs
        if graph.data_count is None or graph.data_count > 0
    ]
    if not relevant_graphs:
        return ""

    if any(graph.graph_empty is False for graph in relevant_graphs):
        return ""

    dataset_issue = _describe_graph_dataset_results(relevant_graphs)
    if any(graph.nodes for graph in relevant_graphs):
        return ""

    PrintStyle.warning(
        f"Cognee graph is still empty after cognify. {dataset_issue}"
    )
    return "graph is still empty"


async def _reset_pipeline_status_for_rebuild(dataset: str) -> None:
    """Clear Cognee pipeline status before retrying a failed rebuild."""
    try:
        from .cognee_init import reset_cognify_status_for_dataset_names

        reset = await reset_cognify_status_for_dataset_names([dataset])
        if reset:
            PrintStyle.standard(
                f"Reset Cognee pipeline status before rebuild retry: {reset}"
            )
    except Exception as e:
        PrintStyle.warning(
            f"Could not reset Cognee pipeline status before rebuild retry for {dataset}: {e}"
        )


def _describe_graph_dataset_results(dataset_graphs: list) -> str:
    non_empty_datasets = [
        graph.dataset_name
        for graph in dataset_graphs
        if graph.data_count is not None and graph.data_count > 0
    ]
    unknown_datasets = [
        graph.dataset_name
        for graph in dataset_graphs
        if graph.data_count is None
    ]

    details = []
    if non_empty_datasets:
        details.append(f"Non-empty dataset(s): {non_empty_datasets}")
    if unknown_datasets:
        details.append(f"unverified dataset(s): {unknown_datasets}")

    return "; ".join(details)


def _short_list(items: list[str], limit: int = 12) -> list[str]:
    if len(items) <= limit:
        return items
    return [*items[:limit], f"... +{len(items) - limit} more"]


async def _describe_non_empty_or_unverified_datasets(cognee, datasets: list[str]) -> str:
    non_empty_datasets = []
    unknown_datasets = []
    for dataset in datasets:
        data_count = await _get_dataset_data_count(cognee, dataset)
        if data_count is None:
            unknown_datasets.append(dataset)
        elif data_count > 0:
            non_empty_datasets.append(dataset)

    details = []
    if non_empty_datasets:
        details.append(f"Non-empty dataset(s): {non_empty_datasets}")
    if unknown_datasets:
        details.append(f"unverified dataset(s): {unknown_datasets}")

    return "; ".join(details)


async def _get_dataset_data_count(cognee, dataset_name: str) -> int | None:
    datasets_api = getattr(cognee, "datasets", None)
    if datasets_api is None:
        return None

    try:
        datasets = await datasets_api.list_datasets()
    except Exception as e:
        PrintStyle.warning(f"Could not list Cognee datasets for readiness check: {e}")
        return None

    target = None
    for dataset in datasets:
        if getattr(dataset, "name", None) == dataset_name:
            target = dataset
            break

    if target is None:
        return 0

    try:
        data_items = await datasets_api.list_data(target.id)
        return len(data_items or [])
    except Exception as e:
        PrintStyle.warning(
            f"Could not list Cognee data for readiness check ({dataset_name}): {e}"
        )
        return None
