import asyncio
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from helpers.defer import DeferredTask, THREAD_BACKGROUND
from helpers.print_style import PrintStyle


@dataclass
class MemoryWriteJob:
    agent: Any
    text: str
    area: str
    metadata: dict[str, Any] = field(default_factory=dict)
    cfg: dict[str, Any] = field(default_factory=dict)
    use_consolidation: bool = True
    replace_threshold: float = 0.9
    similarity_threshold: float = 0.7


def _cfg_float(cfg: dict[str, Any], key: str, default: float) -> float:
    try:
        return float(cfg.get(key, default))
    except (TypeError, ValueError):
        return float(default)


class MemoryWriteWorker:
    _instance = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._queue: deque[MemoryWriteJob] = deque()
        self._state_lock = threading.RLock()
        self._task: DeferredTask | None = None
        self._running = False
        self._last_error: str | None = None
        self._processed_count = 0

    @classmethod
    def get_instance(cls) -> "MemoryWriteWorker":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    def enqueue(
        self,
        *,
        agent: Any,
        text: str,
        area: str,
        metadata: dict[str, Any] | None = None,
        cfg: dict[str, Any] | None = None,
        use_consolidation: bool = True,
        replace_threshold: float = 0.9,
        similarity_threshold: float = 0.7,
    ) -> int:
        job = MemoryWriteJob(
            agent=agent,
            text=str(text),
            area=str(area),
            metadata=dict(metadata or {}),
            cfg=dict(cfg or {}),
            use_consolidation=bool(use_consolidation),
            replace_threshold=float(replace_threshold),
            similarity_threshold=float(similarity_threshold),
        )
        with self._state_lock:
            self._queue.append(job)
            queued = len(self._queue)
        self._ensure_started()
        return queued

    def get_status(self) -> dict[str, Any]:
        with self._state_lock:
            return {
                "queued": len(self._queue),
                "processed": self._processed_count,
                "last_error": self._last_error,
            }

    def _ensure_started(self) -> None:
        with self._state_lock:
            if self._running:
                return
            task = DeferredTask(thread_name=THREAD_BACKGROUND)
            self._task = task
            self._running = True
        try:
            task.start_task(self.run_loop)
        except Exception:
            with self._state_lock:
                self._running = False
            raise

    def _pop_job(self) -> MemoryWriteJob | None:
        with self._state_lock:
            if not self._queue:
                return None
            return self._queue.popleft()

    async def run_loop(self) -> None:
        try:
            while True:
                job = self._pop_job()
                if job is None:
                    return
                try:
                    await self._wait_until_idle(job)
                    await self._process_job(job)
                    with self._state_lock:
                        self._processed_count += 1
                        self._last_error = None
                except Exception as e:
                    with self._state_lock:
                        self._last_error = str(e)
                    PrintStyle.warning(f"Memory write worker job failed: {e}")
        finally:
            with self._state_lock:
                self._running = False
                should_restart = bool(self._queue)
            if should_restart:
                self._ensure_started()

    async def _wait_until_idle(self, job: MemoryWriteJob) -> None:
        idle_seconds = _cfg_float(job.cfg, "memory_consolidation_idle_seconds", 60)
        if idle_seconds <= 0:
            return

        from .cognee_background import CogneeBackgroundWorker

        while True:
            if CogneeBackgroundWorker.get_instance().is_memory_idle(idle_seconds):
                return
            await asyncio.sleep(min(max(idle_seconds / 4, 1), 5))

    async def _process_job(self, job: MemoryWriteJob) -> dict[str, Any]:
        if job.use_consolidation:
            from .memory_consolidation import create_memory_consolidator

            consolidator = create_memory_consolidator(
                job.agent,
                similarity_threshold=job.similarity_threshold,
                replace_similarity_threshold=job.replace_threshold,
                processing_timeout_seconds=_cfg_float(
                    job.cfg,
                    "memory_consolidation_timeout_seconds",
                    30,
                ),
                keyword_extraction_enabled=bool(
                    job.cfg.get("memory_consolidation_keyword_extraction_enabled", True)
                ),
                utility_timeout_seconds=_cfg_float(
                    job.cfg,
                    "memory_consolidation_utility_timeout_seconds",
                    10,
                ),
            )
            return await consolidator.process_new_memory(
                new_memory=job.text,
                area=job.area,
                metadata=job.metadata,
                log_item=None,
            )

        from .memory import Memory, insert_with_simple_dedup

        db = await Memory.get(job.agent, preload_knowledge=False)
        memory_id = await insert_with_simple_dedup(
            db,
            job.text,
            job.area,
            job.replace_threshold,
        )
        return {"success": bool(memory_id), "memory_ids": [memory_id] if memory_id else []}
