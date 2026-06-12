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
_COGNEE_WAIT_STATE_LOCK = threading.Lock()
_USER_WAITERS = 0
_COGNEE_CHILD_PROCESS_PIDS: set[int] = set()
_COGNEE_LOOP_LOCK = threading.Lock()
_COGNEE_OPERATION_LOOP: asyncio.AbstractEventLoop | None = None
_COGNEE_OPERATION_LOOP_THREAD: threading.Thread | None = None
_COGNEE_CURRENT_OPERATION_LOCK = threading.Lock()
_CURRENT_OPERATION_LABEL: str | None = None
_CURRENT_OPERATION_STARTED_AT: float | None = None
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


def _ensure_cognee_operation_loop() -> asyncio.AbstractEventLoop:
    global _COGNEE_OPERATION_LOOP, _COGNEE_OPERATION_LOOP_THREAD

    with _COGNEE_LOOP_LOCK:
        if _COGNEE_OPERATION_LOOP and _COGNEE_OPERATION_LOOP.is_running():
            return _COGNEE_OPERATION_LOOP

        ready = threading.Event()
        loop_holder: dict[str, asyncio.AbstractEventLoop] = {}
        error_holder: list[BaseException] = []

        def _run_loop() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop_holder["loop"] = loop
            ready.set()
            try:
                loop.run_forever()
            except BaseException as e:
                error_holder.append(e)
            finally:
                pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.close()

        thread = threading.Thread(
            target=_run_loop,
            name="memory-cognee-operation-loop",
            daemon=True,
        )
        thread.start()
        if not ready.wait(timeout=5):
            raise RuntimeError("Timed out starting Cognee operation loop")
        if error_holder:
            raise RuntimeError("Cognee operation loop failed to start") from error_holder[0]

        _COGNEE_OPERATION_LOOP = loop_holder["loop"]
        _COGNEE_OPERATION_LOOP_THREAD = thread
        return _COGNEE_OPERATION_LOOP


def _stop_cognee_operation_loop() -> None:
    global _COGNEE_OPERATION_LOOP, _COGNEE_OPERATION_LOOP_THREAD

    with _COGNEE_LOOP_LOCK:
        loop = _COGNEE_OPERATION_LOOP
        thread = _COGNEE_OPERATION_LOOP_THREAD
        _COGNEE_OPERATION_LOOP = None
        _COGNEE_OPERATION_LOOP_THREAD = None

    if loop and loop.is_running():
        loop.call_soon_threadsafe(loop.stop)
    if thread and thread.is_alive():
        thread.join(timeout=5)


def _set_current_operation(label: str | None) -> None:
    global _CURRENT_OPERATION_LABEL, _CURRENT_OPERATION_STARTED_AT
    with _COGNEE_CURRENT_OPERATION_LOCK:
        _CURRENT_OPERATION_LABEL = label
        _CURRENT_OPERATION_STARTED_AT = time.monotonic() if label else None


def _gate_timeout_message(label: str) -> str:
    detail = ""
    with _COGNEE_CURRENT_OPERATION_LOCK:
        current_label = _CURRENT_OPERATION_LABEL
        started_at = _CURRENT_OPERATION_STARTED_AT
    if current_label:
        held_for = (
            time.monotonic() - started_at
            if isinstance(started_at, (int, float))
            else 0.0
        )
        detail = f"; current={current_label!r}; held_for={held_for:.1f}s"
    return f"Timed out waiting for Cognee operation gate: {label}{detail}"


async def _execute_cognee_operation(
    label: str,
    operation: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    operation_timeout: float | None,
) -> Any:
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


async def _run_on_cognee_operation_loop(coro: Any) -> Any:
    target_loop = _ensure_cognee_operation_loop()
    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None

    if running_loop is target_loop:
        return await coro

    future = asyncio.run_coroutine_threadsafe(coro, target_loop)
    try:
        return await asyncio.wrap_future(future)
    except asyncio.CancelledError:
        future.cancel()
        raise


async def run_cognee_operation(
    label: str,
    operation: Callable[..., Any],
    *args,
    timeout: float = _DEFAULT_WAIT_TIMEOUT,
    operation_timeout: float | None = None,
    a0_agent: Any = None,
    priority: str = "normal",
    **kwargs,
) -> T:
    """Serialize Cognee backend operations across async tasks and threads."""
    deadline = time.monotonic() + timeout if timeout > 0 else None
    acquired = False
    is_user_priority = priority == "user"
    is_background_priority = priority == "background"

    if is_user_priority:
        global _USER_WAITERS
        with _COGNEE_WAIT_STATE_LOCK:
            _USER_WAITERS += 1

    try:
        while not acquired:
            if is_background_priority:
                with _COGNEE_WAIT_STATE_LOCK:
                    user_waiters = _USER_WAITERS
                if user_waiters:
                    if deadline is not None and time.monotonic() >= deadline:
                        raise TimeoutError(_gate_timeout_message(label))
                    await asyncio.sleep(_POLL_INTERVAL)
                    continue

            acquired = _COGNEE_OP_LOCK.acquire(blocking=False)
            if acquired:
                _set_current_operation(label)
                break
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(_gate_timeout_message(label))
            await asyncio.sleep(_POLL_INTERVAL)
    finally:
        if is_user_priority:
            with _COGNEE_WAIT_STATE_LOCK:
                _USER_WAITERS = max(0, _USER_WAITERS - 1)

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
        return await _run_on_cognee_operation_loop(
            _execute_cognee_operation(
                label,
                operation,
                args,
                kwargs,
                operation_timeout,
            )
        )
    finally:
        try:
            cleanup_cognee_child_processes(
                label,
                baseline_pids=baseline_pids,
                include_cognee_fd_holders=True,
            )
        finally:
            _set_current_operation(None)
            _COGNEE_OP_LOCK.release()
