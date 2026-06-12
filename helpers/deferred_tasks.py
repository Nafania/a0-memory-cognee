import threading
from collections import deque
from typing import Any


_MAX_TRACKED_TASKS = 128
_TRACKED_TASKS: deque[Any] = deque()
_TRACKED_TASKS_LOCK = threading.Lock()


def track_deferred_task(task: Any) -> Any:
    """Keep Agent Zero DeferredTask alive until it finishes.

    Agent Zero extension execution ignores non-awaitable return values. Without
    a plugin-held reference, a returned DeferredTask can be garbage-collected
    immediately and its __del__ cancels the background coroutine.
    """
    if task is None:
        return task

    with _TRACKED_TASKS_LOCK:
        alive_tasks: list[Any] = []
        while _TRACKED_TASKS:
            existing = _TRACKED_TASKS.popleft()
            try:
                if existing.is_alive():
                    alive_tasks.append(existing)
            except Exception:
                continue

        for existing in alive_tasks[-(_MAX_TRACKED_TASKS - 1):]:
            _TRACKED_TASKS.append(existing)
        _TRACKED_TASKS.append(task)

    return task


def tracked_deferred_task_count() -> int:
    with _TRACKED_TASKS_LOCK:
        return len(_TRACKED_TASKS)
