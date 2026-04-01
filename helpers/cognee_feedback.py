"""
Cognee memory feedback: discover session.add_feedback, forward scores, durable disk queue.

Queue: usr/cognee_feedback_queue/pending/*.json (at-least-once retries via drain).
Invalid payloads are moved to usr/cognee_feedback_queue/failed/ (or removed if quarantine fails).
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import asyncio
import uuid
from typing import Any, Callable

from helpers import files
from helpers.settings import get_settings

_log = logging.getLogger(__name__)

_UNSET = object()

VALID_FEEDBACK = frozenset({"positive", "negative"})
MAX_RETRY_ATTEMPTS = 5


class FeedbackPayloadError(ValueError):
    pass


def validate_feedback_payload(payload: dict[str, Any]) -> None:
    for key in ("context_id", "dataset", "memory_id", "feedback"):
        if key not in payload or payload[key] is None:
            raise FeedbackPayloadError(f"missing_field:{key}")
        if isinstance(payload[key], str) and not payload[key].strip():
            raise FeedbackPayloadError(f"empty_field:{key}")
    fb = payload["feedback"]
    if fb not in VALID_FEEDBACK:
        raise FeedbackPayloadError("invalid_feedback")


def discover_cognee_feedback_callable(cognee_module: Any) -> Callable[..., Any] | None:
    if cognee_module is None:
        return None
    session = getattr(cognee_module, "session", None)
    if session is not None:
        fn = getattr(session, "add_feedback", None)
        if callable(fn):
            return fn
    fn = getattr(cognee_module, "add_feedback", None)
    if callable(fn):
        return fn
    return None


def _pending_dir() -> str:
    return files.get_abs_path("usr", "cognee_feedback_queue", "pending")


def _failed_dir() -> str:
    return files.get_abs_path("usr", "cognee_feedback_queue", "failed")


def _quarantine_invalid_queue_file(path: str, reason: str) -> None:
    try:
        os.makedirs(_failed_dir(), exist_ok=True)
        base = os.path.basename(path)
        dest = os.path.join(_failed_dir(), base)
        if os.path.exists(dest):
            dest = os.path.join(_failed_dir(), f"{uuid.uuid4().hex}_{base}")
        os.replace(path, dest)
        _log.warning(
            "cognee_feedback quarantined invalid queue file path=%s reason=%s dest=%s",
            path,
            reason,
            dest,
        )
    except OSError as e:
        _log.error(
            "cognee_feedback could not quarantine path=%s err=%s; deleting to stop retry loop",
            path,
            e,
        )
        try:
            os.remove(path)
        except OSError as e2:
            _log.error("cognee_feedback could not remove poison file path=%s err=%s", path, e2)


def _feedback_score(feedback: str) -> int:
    return 5 if feedback == "positive" else 1


def _enqueue_record(payload: dict[str, Any]) -> None:
    os.makedirs(_pending_dir(), exist_ok=True)
    record = {
        "context_id": payload["context_id"],
        "dataset": payload["dataset"],
        "memory_id": payload["memory_id"],
        "feedback": payload["feedback"],
        "reason": payload.get("reason"),
        "attempts": 0,
    }
    name = f"{uuid.uuid4().hex}.json"
    path = os.path.join(_pending_dir(), name)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False)
    os.replace(tmp, path)
    _log.info("cognee_feedback queued path=%s", path)


def _rewrite_pending_record(path: str, record: dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False)
    os.replace(tmp, path)


async def _invoke_add_feedback(
    add_fn: Callable[..., Any],
    *,
    session_id: str,
    qa_id: str,
    feedback_text: str,
    feedback_score: int,
) -> Any:
    kwargs = {
        "session_id": session_id,
        "qa_id": qa_id,
        "feedback_text": feedback_text,
        "feedback_score": feedback_score,
    }
    try:
        sig = inspect.signature(add_fn)
        names = set(sig.parameters.keys())
        if names:
            filtered = {k: v for k, v in kwargs.items() if k in names}
            if filtered:
                kwargs = filtered
    except (TypeError, ValueError):
        pass
    result = add_fn(**kwargs)
    if inspect.isawaitable(result):
        result = await result
    return result


async def _try_forward(
    cognee_module: Any,
    payload: dict[str, Any],
) -> bool | None:
    """
    Returns True if Cognee accepted, False if rejected, None if no API.
    """
    add_fn = discover_cognee_feedback_callable(cognee_module)
    if add_fn is None:
        return None
    text = (payload.get("reason") or "").strip() or (
        "positive" if payload["feedback"] == "positive" else "negative"
    )
    score = _feedback_score(payload["feedback"])
    try:
        ok = await _invoke_add_feedback(
            add_fn,
            session_id=str(payload["context_id"]),
            qa_id=str(payload["memory_id"]),
            feedback_text=text,
            feedback_score=score,
        )
        if ok is False:
            _log.warning(
                "cognee_feedback not applied (explicit False) session_id=%s qa_id=%s",
                payload["context_id"],
                payload["memory_id"],
            )
            return False
        if ok is None or ok is True:
            _log.info(
                "cognee_feedback forwarded session_id=%s qa_id=%s score=%s",
                payload["context_id"],
                payload["memory_id"],
                score,
            )
            return True
        if ok:
            _log.info(
                "cognee_feedback forwarded session_id=%s qa_id=%s score=%s result=%r",
                payload["context_id"],
                payload["memory_id"],
                score,
                ok,
            )
            return True
        _log.warning(
            "cognee_feedback not applied (falsy result) session_id=%s qa_id=%s",
            payload["context_id"],
            payload["memory_id"],
        )
        return False
    except Exception as e:
        _log.warning(
            "cognee_feedback forward error session_id=%s qa_id=%s err=%s",
            payload["context_id"],
            payload["memory_id"],
            e,
        )
        return False


async def drain_feedback_queue(
    *,
    cognee_module: Any = _UNSET,
    limit: int = 50,
) -> int:
    """
    Attempt to deliver pending queue entries. Returns count successfully forwarded.
    """
    if cognee_module is _UNSET:
        try:
            from .cognee_init import get_cognee

            cognee_module, _ = get_cognee()
        except Exception:
            cognee_module = None

    pdir = _pending_dir()
    if not os.path.isdir(pdir):
        return 0
    names = sorted(f for f in os.listdir(pdir) if f.endswith(".json"))
    forwarded = 0
    for name in names[: max(0, limit)]:
        path = os.path.join(pdir, name)
        try:
            with open(path, encoding="utf-8") as f:
                record = json.load(f)
            validate_feedback_payload(record)
        except Exception as e:
            _log.warning("cognee_feedback invalid queue file=%s err=%s", path, e)
            _quarantine_invalid_queue_file(path, str(e))
            continue
        ok = await _try_forward(cognee_module, record)
        if ok is True:
            try:
                os.remove(path)
            except OSError:
                pass
            forwarded += 1
            continue
        if ok is False:
            attempts = int(record.get("attempts", 0)) + 1
            if attempts >= MAX_RETRY_ATTEMPTS:
                _quarantine_invalid_queue_file(
                    path, f"max_retries_exceeded:{attempts}"
                )
                continue
            record["attempts"] = attempts
            try:
                _rewrite_pending_record(path, record)
            except Exception as e:
                _log.warning(
                    "cognee_feedback could not persist attempts path=%s attempts=%s err=%s",
                    path,
                    attempts,
                    e,
                )
    return forwarded


async def submit_memory_feedback(
    payload: dict[str, Any],
    cognee_module: Any = _UNSET,
) -> dict[str, Any]:
    validate_feedback_payload(payload)

    try:
        asyncio.create_task(drain_feedback_queue(cognee_module=cognee_module, limit=20))
    except Exception as e:
        _log.debug("cognee_feedback background drain scheduling skipped err=%s", e)

    settings = get_settings()
    if not settings.get("cognee_feedback_enabled", True):
        try:
            _enqueue_record(payload)
        except OSError as e:
            _log.error("cognee_feedback enqueue failed (disabled path) err=%s", e)
            return {"status": "failed", "error": str(e)}
        _log.info("cognee_feedback disabled in settings; queued only")
        return {"status": "queued"}

    if cognee_module is _UNSET:
        try:
            from .cognee_init import get_cognee

            cognee_module, _ = get_cognee()
        except Exception:
            cognee_module = None

    add_fn = discover_cognee_feedback_callable(cognee_module)
    if add_fn is None:
        _log.warning(
            "Cognee feedback API unavailable; queue-only mode (no add_feedback callable)"
        )
        try:
            _enqueue_record(payload)
        except OSError as e:
            _log.error("cognee_feedback enqueue failed err=%s", e)
            return {"status": "failed", "error": str(e)}
        return {"status": "queued"}

    ok = await _try_forward(cognee_module, payload)
    if ok is True:
        return {"status": "forwarded"}

    try:
        _enqueue_record(payload)
    except OSError as e:
        _log.error("cognee_feedback enqueue after forward failure err=%s", e)
        return {"status": "failed", "error": str(e)}
    return {"status": "queued"}
