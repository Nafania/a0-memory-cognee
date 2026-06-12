import asyncio
import json
import hashlib
from typing import Any

from helpers.print_style import PrintStyle

from usr.plugins.memory_cognee.helpers.cognee_init import get_cognee
from usr.plugins.memory_cognee.helpers.cognee_ops import run_cognee_operation
from usr.plugins.memory_cognee.helpers.memory import Memory


MAX_SESSION_TEXT = 4000
DATA_NAME_LAST_SESSION_QA = "_memory_cognee_last_session_qa"
QUERY_METADATA_KEYS = {
    "attachments",
    "files",
    "images",
    "metadata",
    "meta",
    "system_message",
    "tool_calls",
    "tools",
}
TOOL_PAYLOAD_KEYS = {
    "tool_name",
    "tool_args",
    "tool_result",
    "tool_calls",
}


def extract_latest_qa(history) -> tuple[str, str]:
    try:
        outputs = history.output()
    except Exception:
        return "", ""

    current_user = ""
    latest_question = ""
    latest_answer = ""

    for item in outputs or []:
        is_ai = item.get("ai") if isinstance(item, dict) else getattr(item, "ai", False)
        content = item.get("content") if isinstance(item, dict) else getattr(item, "content", None)
        text = _content_text(content)
        if not text:
            continue
        if is_ai:
            if current_user:
                latest_question = current_user
                latest_answer = text
        else:
            current_user = text

    return latest_question.strip(), latest_answer.strip()


async def remember_session_turn(agent) -> bool:
    cfg = _plugin_config(agent)
    if not cfg.get("memory_session_enabled", True):
        return False

    session_id = str(getattr(getattr(agent, "context", None), "id", "") or "")
    if not session_id:
        return False

    question, answer = extract_latest_qa(agent.history)
    if not question or not answer:
        return False

    qa_key = _qa_key(session_id, question, answer)
    get_data = getattr(agent, "get_data", None)
    if callable(get_data) and get_data(DATA_NAME_LAST_SESSION_QA) == qa_key:
        return False

    db = await Memory.get(agent, preload_knowledge=False)
    cognee, _ = get_cognee()
    from cognee.memory import QAEntry

    entry = QAEntry(
        question=question[:MAX_SESSION_TEXT],
        answer=answer[:MAX_SESSION_TEXT],
        context="",
    )
    operation_timeout = _positive_float(cfg.get("cognee_operation_timeout_seconds"), 1800)
    _mark_memory_activity()
    result = await run_cognee_operation(
        "cognee.remember session",
        cognee.remember,
        entry,
        dataset_name=db.dataset_name,
        session_id=session_id,
        self_improvement=False,
        timeout=operation_timeout,
        operation_timeout=operation_timeout,
        a0_agent=agent,
        priority="background",
    )
    status = str(getattr(result, "status", "") or "")
    if status == "errored":
        error = str(getattr(result, "error", "") or "unknown")
        raise RuntimeError(f"Cognee session remember failed: {error}")
    set_data = getattr(agent, "set_data", None)
    if callable(set_data):
        set_data(DATA_NAME_LAST_SESSION_QA, qa_key)
    return True


def _mark_memory_activity() -> None:
    try:
        from usr.plugins.memory_cognee.helpers.cognee_background import (
            CogneeBackgroundWorker,
        )

        mark_activity = getattr(CogneeBackgroundWorker.get_instance(), "mark_activity", None)
        if callable(mark_activity):
            mark_activity()
    except Exception:
        return


def _plugin_config(agent) -> dict:
    try:
        from helpers import plugins

        return plugins.get_plugin_config("memory_cognee", agent) or {}
    except Exception:
        return {}


def _content_text(content: Any) -> str:
    if content is None:
        return ""

    if isinstance(content, str):
        text = _strip_extras(content).strip()
        if text.lstrip().startswith("{"):
            parsed = _parse_json_object(text)
            if parsed is not None:
                tool_text = _tool_payload_text(parsed)
                if tool_text is not None:
                    return tool_text
            if "tool_name" in text and "tool_args" in text:
                return ""
        return text

    if isinstance(content, dict):
        tool_text = _tool_payload_text(content)
        if tool_text is not None:
            return tool_text

        preferred_keys = ("user_message", "message", "text", "preview", "response")
        for key in preferred_keys:
            if key in content:
                text = _content_text(content.get(key))
                if text:
                    return text

        if "raw_content" in content:
            text = _content_text(content.get("raw_content"))
            if text:
                return text

        parts = [
            _content_text(value)
            for key, value in content.items()
            if key not in preferred_keys
            and key != "raw_content"
            and key not in QUERY_METADATA_KEYS
        ]
        return "\n".join(part for part in parts if part).strip()

    if isinstance(content, list):
        return "\n".join(part for part in (_content_text(item) for item in content) if part).strip()

    return str(content).strip()


def _positive_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _is_tool_payload(content: dict) -> bool:
    if "user_message" in content:
        return False
    return any(key in content for key in TOOL_PAYLOAD_KEYS)


def _tool_payload_text(content: dict) -> str | None:
    if not _is_tool_payload(content):
        return None

    if str(content.get("tool_name") or "") == "response":
        tool_args = content.get("tool_args")
        if isinstance(tool_args, dict):
            return _content_text(tool_args.get("text"))
    return ""


def _parse_json_object(text: str) -> dict | None:
    try:
        parsed = json.loads(text)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _strip_extras(text: str) -> str:
    return str(text or "").split("[EXTRAS]", 1)[0].strip()


def _qa_key(session_id: str, question: str, answer: str) -> str:
    raw = f"{session_id}\n{question}\n{answer}".encode("utf-8", errors="replace")
    return hashlib.sha256(raw).hexdigest()


async def safe_remember_session_turn(agent) -> None:
    try:
        await _wait_until_memory_idle(agent)
        await remember_session_turn(agent)
    except Exception as e:
        try:
            PrintStyle.warning(f"Cognee session memory write failed: {e}")
        except OSError:
            pass


async def _wait_until_memory_idle(agent) -> None:
    cfg = _plugin_config(agent)
    idle_seconds = _positive_float(
        cfg.get(
            "memory_session_idle_seconds",
            cfg.get("memory_consolidation_idle_seconds"),
        ),
        60,
    )
    if idle_seconds <= 0:
        return

    try:
        from usr.plugins.memory_cognee.helpers.cognee_background import (
            CogneeBackgroundWorker,
        )

        worker = CogneeBackgroundWorker.get_instance()
    except Exception:
        return

    while True:
        is_memory_idle = getattr(worker, "is_memory_idle", None)
        if not callable(is_memory_idle) or is_memory_idle(idle_seconds):
            return
        await asyncio.sleep(min(max(idle_seconds / 4, 1), 5))
