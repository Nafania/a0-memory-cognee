import asyncio
import inspect
import threading
import time
from collections.abc import Callable
from typing import Any, TypeVar


T = TypeVar("T")

_COGNEE_OP_LOCK = threading.Lock()
_DEFAULT_WAIT_TIMEOUT = 180.0
_POLL_INTERVAL = 0.05


async def run_cognee_operation(
    label: str,
    operation: Callable[..., Any],
    *args,
    timeout: float = _DEFAULT_WAIT_TIMEOUT,
    **kwargs,
) -> T:
    """Serialize Cognee backend operations across async tasks and threads."""
    deadline = time.monotonic() + timeout if timeout > 0 else None
    acquired = False

    while not acquired:
        acquired = _COGNEE_OP_LOCK.acquire(blocking=False)
        if acquired:
            break
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for Cognee operation gate: {label}")
        await asyncio.sleep(_POLL_INTERVAL)

    try:
        result = operation(*args, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result
    finally:
        _COGNEE_OP_LOCK.release()
