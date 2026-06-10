import asyncio
import gc
import inspect
import multiprocessing
import os
import threading
import time
from collections.abc import Callable
from typing import Any, TypeVar


T = TypeVar("T")

_COGNEE_OP_LOCK = threading.Lock()
_COGNEE_CHILD_PROCESS_PIDS: set[int] = set()
_DEFAULT_WAIT_TIMEOUT = 180.0
_POLL_INTERVAL = 0.05


def _child_pid(child: Any) -> int | None:
    pid = getattr(child, "pid", None)
    if pid is None:
        return None
    try:
        return int(pid)
    except (TypeError, ValueError):
        return None


def _active_child_pids() -> set[int]:
    return {
        pid
        for child in multiprocessing.active_children()
        if (pid := _child_pid(child)) is not None
    }


def _child_holds_cognee_file(pid: int) -> bool:
    fd_dir = f"/proc/{pid}/fd"
    if not os.path.isdir(fd_dir):
        return False

    try:
        fd_names = os.listdir(fd_dir)
    except OSError:
        return False

    for fd_name in fd_names:
        try:
            target = os.readlink(os.path.join(fd_dir, fd_name))
        except OSError:
            continue
        if "/cognee/" in target or "/cognee_system/" in target:
            return True
    return False


def cleanup_cognee_child_processes(
    label: str,
    *,
    baseline_pids: set[int] | None = None,
    include_cognee_fd_holders: bool = True,
) -> None:
    """Release Cognee/FastEmbed multiprocessing children after backend calls."""
    baseline = baseline_pids or set()
    children = []
    for child in multiprocessing.active_children():
        pid = _child_pid(child)
        known_child = pid in _COGNEE_CHILD_PROCESS_PIDS if pid is not None else False
        stale_cognee_fd = (
            include_cognee_fd_holders
            and pid is not None
            and _child_holds_cognee_file(pid)
        )
        new_child = pid not in baseline if pid is not None else True
        if not (new_child or known_child or stale_cognee_fd):
            continue
        if pid is not None:
            _COGNEE_CHILD_PROCESS_PIDS.add(pid)
        children.append(child)

    if not children:
        gc.collect()
        return

    for child in children:
        try:
            if child.is_alive():
                child.terminate()
        except Exception:
            pass

    still_alive = []
    for child in children:
        try:
            child.join(timeout=5)
            if child.is_alive():
                still_alive.append(child)
        except Exception:
            pass

    for child in still_alive:
        try:
            child.kill()
            child.join(timeout=2)
        except Exception:
            pass

    for child in children:
        pid = _child_pid(child)
        if pid is None:
            continue
        try:
            if not child.is_alive():
                _COGNEE_CHILD_PROCESS_PIDS.discard(pid)
        except Exception:
            pass

    gc.collect()


async def run_cognee_operation(
    label: str,
    operation: Callable[..., Any],
    *args,
    timeout: float = _DEFAULT_WAIT_TIMEOUT,
    operation_timeout: float | None = None,
    a0_agent: Any = None,
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

    pre_existing_pids = _active_child_pids()
    baseline_pids = pre_existing_pids
    try:
        from .cognee_init import ensure_cognee_llm_config_current

        ensure_cognee_llm_config_current(a0_agent)
        cleanup_cognee_child_processes(
            f"{label} pre",
            baseline_pids=pre_existing_pids,
            include_cognee_fd_holders=True,
        )
        baseline_pids = _active_child_pids()
        result = operation(*args, **kwargs)
        if inspect.isawaitable(result):
            if operation_timeout and operation_timeout > 0:
                try:
                    return await asyncio.wait_for(result, timeout=operation_timeout)
                except TimeoutError as e:
                    raise TimeoutError(
                        f"Cognee operation timed out after {operation_timeout:.0f}s: {label}"
                    ) from e
            return await result
        return result
    finally:
        try:
            cleanup_cognee_child_processes(
                label,
                baseline_pids=baseline_pids,
                include_cognee_fd_holders=True,
            )
        finally:
            _COGNEE_OP_LOCK.release()
