import asyncio
import json
from helpers.extension import Extension
from usr.plugins.memory_cognee.helpers import memory as memory_helpers
from usr.plugins.memory_cognee.helpers.cognee_init import is_cognee_debug_enabled
from agent import LoopData
from helpers import plugins, log
from helpers.print_style import PrintStyle
from usr.plugins.memory_cognee.helpers.cognee_ops import run_cognee_operation

DATA_NAME_TASK = "_recall_memories_task"
DATA_NAME_ITER = "_recall_memories_iter"
DEFAULT_SEARCH_TIMEOUT_SECONDS = 180
AREA_FILTER_OVERFETCH_FACTOR = 3
AREA_FILTER_MAX_TOP_K = 100
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

Memory = memory_helpers.Memory
recall_text_and_feedback_items = memory_helpers.recall_text_and_feedback_items
split_recall_answers_by_area = memory_helpers.split_recall_answers_by_area


def _touch_memory_activity() -> None:
    touch = getattr(memory_helpers, "touch_memory_activity", None)
    if callable(touch):
        touch()


class RecallMemories(Extension):

    async def execute(self, loop_data: LoopData = LoopData(), **kwargs):
        cfg = plugins.get_plugin_config("memory_cognee", self.agent)
        if not cfg:
            return

        if not cfg["memory_recall_enabled"]:
            return

        _touch_memory_activity()

        existing_task = self.agent.get_data(DATA_NAME_TASK)
        if existing_task and not existing_task.done():
            self.agent.set_data(DATA_NAME_ITER, loop_data.iteration)
            return

        if loop_data.iteration % cfg["memory_recall_interval"] == 0:
            log_item = self.agent.context.log.log(
                type="util",
                heading="Searching memories...",
            )

            search = self.search_memories(loop_data=loop_data, log_item=log_item, **kwargs)
            timeout = _recall_timeout_seconds(cfg)
            if timeout > 0:
                search = asyncio.wait_for(search, timeout=timeout)
            task = asyncio.create_task(search)
            task.add_done_callback(_log_unhandled_recall_exception)
        else:
            task = None

        self.agent.set_data(DATA_NAME_TASK, task)
        self.agent.set_data(DATA_NAME_ITER, loop_data.iteration)

    async def search_memories(self, log_item: log.LogItem, loop_data: LoopData, **kwargs):
        extras = loop_data.extras_persistent
        if "memories" in extras:
            del extras["memories"]
        if "solutions" in extras:
            del extras["solutions"]

        cfg = plugins.get_plugin_config("memory_cognee", self.agent)
        if not cfg:
            return

        query = await _prepare_recall_query(
            self.agent,
            self.agent.history,
            loop_data.user_message,
            cfg,
            log_item,
        )

        if not query or len(query) <= 3:
            log_item.update(
                query="No relevant memory query generated, skipping search",
            )
            return
        log_item.update(query=query)

        db = await Memory.get(self.agent, preload_knowledge=False)

        datasets = db.get_search_datasets()
        from usr.plugins.memory_cognee.helpers.cognee_background import (
            CogneeBackgroundWorker,
        )

        block_reason = CogneeBackgroundWorker.get_instance().get_search_block_reason(
            datasets
        )
        if block_reason:
            log_item.update(
                heading=_recall_block_heading(block_reason),
                content=block_reason,
            )
            return

        from usr.plugins.memory_cognee.helpers.cognee_init import get_cognee
        cognee, SearchType = get_cognee()
        from cognee.modules.engine.models.node_set import NodeSet

        memory_search_limit = cfg["memory_recall_memories_max_search"]
        solution_search_limit = cfg["memory_recall_solutions_max_search"]
        combined_search_limit = _combined_search_limit(
            memory_search_limit,
            solution_search_limit,
        )

        try:
            session_id = getattr(self.agent.context, 'id', None)
            if _debug_enabled(cfg):
                log_item.update(
                    cognee_search_args={
                        "query_text": query,
                        "query_type": "CHUNKS",
                        "top_k": combined_search_limit,
                        "datasets": datasets,
                        "node_type": "NodeSet",
                        "node_name": _memory_area_names(),
                        "session_id": session_id,
                        "only_context": True,
                        "verbose": True,
                    }
                )
            combined_answers = await run_cognee_operation(
                "cognee.search auto-recall",
                cognee.search,
                query_text=query,
                query_type=SearchType.CHUNKS,
                top_k=combined_search_limit,
                datasets=datasets,
                node_type=NodeSet,
                node_name=_memory_area_names(),
                session_id=session_id,
                only_context=True,
                verbose=True,
                a0_agent=self.agent,
            )
            mem_answers, sol_answers = split_recall_answers_by_area(
                combined_answers,
                memory_search_limit,
                solution_search_limit,
            )
            if _debug_enabled(cfg):
                log_item.update(
                    cognee_search_result_count=len(combined_answers or []),
                )
        except OSError as e:
            try:
                PrintStyle.error(f"cognee.search OS error (likely too many open files): {e}")
            except OSError:
                pass
            log_item.update(
                heading="Memory recall failed",
                content=f"cognee.search OS error: {e}",
            )
            return
        except Exception as e:
            try:
                PrintStyle.error(f"cognee.search failed: {e}")
            except OSError:
                pass
            log_item.update(
                heading="Memory recall failed",
                content=f"cognee.search failed: {e}",
            )
            return

        if not combined_answers:
            from usr.plugins.memory_cognee.helpers.cognee_background import (
                CogneeBackgroundWorker,
            )

            CogneeBackgroundWorker.get_instance().nudge_rebuild_if_unready(
                datasets,
                "recall returned empty context",
            )

        ctx = str(getattr(self.agent.context, "id", "") or "")
        fb_fallback = db.dataset_name
        memories, mem_fb = recall_text_and_feedback_items(
            mem_answers,
            cfg["memory_recall_memories_max_result"],
            context_id=ctx,
            fallback_dataset=fb_fallback,
            kind="memory",
        )
        solutions, sol_fb = recall_text_and_feedback_items(
            sol_answers,
            cfg["memory_recall_solutions_max_result"],
            context_id=ctx,
            fallback_dataset=fb_fallback,
            kind="solution",
        )
        _write_extras(self.agent, extras, memories, solutions, log_item, mem_fb + sol_fb)


def _recall_block_heading(block_reason: str) -> str:
    reason = (block_reason or "").lower()
    if "failed" in reason:
        return "Memory rebuild failed; skipping recall"
    if "pending" in reason:
        return "Memory rebuild pending; skipping recall"
    if "stale" in reason:
        return "Memory rebuild stale; skipping recall"
    return "Memory rebuild in progress; skipping recall"


def _combined_search_limit(memory_limit: int, solution_limit: int) -> int:
    base = max(int(memory_limit or 0), 0) + max(int(solution_limit or 0), 0)
    if base <= 0:
        return 1
    return min(base * AREA_FILTER_OVERFETCH_FACTOR, AREA_FILTER_MAX_TOP_K)


def _recall_timeout_seconds(cfg: dict) -> float:
    try:
        timeout = float(cfg.get("memory_recall_timeout_seconds", DEFAULT_SEARCH_TIMEOUT_SECONDS))
    except (TypeError, ValueError):
        return float(DEFAULT_SEARCH_TIMEOUT_SECONDS)
    return max(timeout, 0.0)


def _debug_enabled(cfg: dict) -> bool:
    try:
        return bool(is_cognee_debug_enabled())
    except Exception:
        return bool(cfg.get("cognee_debug_enabled", False))


def _memory_area_names() -> list[str]:
    try:
        return [area.value for area in Memory.Area]
    except TypeError:
        return [
            value.value
            for value in (
                Memory.Area.MAIN,
                Memory.Area.FRAGMENTS,
                Memory.Area.SOLUTIONS,
            )
        ]


def _log_unhandled_recall_exception(task: asyncio.Task) -> None:
    if task.cancelled():
        return
    try:
        exc = task.exception()
    except Exception as e:
        exc = e
    if exc:
        try:
            PrintStyle.error(f"Memory recall task failed: {exc}")
        except OSError:
            pass


def _message_query_text(message) -> str:
    if not message:
        return ""

    content = getattr(message, "content", None)
    text = _content_query_text(content)
    if content is not None:
        return text

    try:
        return str(message.output_text() or "").strip()
    except Exception:
        return ""


def _history_query_text(history) -> str:
    try:
        outputs = history.output()
    except Exception:
        try:
            return _strip_extras(str(history.output_text() or "")).strip()
        except Exception:
            return ""

    parts: list[str] = []
    for item in outputs or []:
        is_ai = item.get("ai") if isinstance(item, dict) else getattr(item, "ai", False)
        content = item.get("content") if isinstance(item, dict) else getattr(item, "content", None)
        text = _content_query_text(content)
        if text:
            role = "ai" if is_ai else "user"
            line = f"{role}: {text}"
            if line not in parts:
                parts.append(line)
    return "\n".join(reversed(parts)).strip()


def _recall_query_text(history, message, max_len: int) -> str:
    message_text = _message_query_text(message)
    history_text = _trim_query(_history_query_text(history), max_len)
    if message_text:
        parts = [message_text]
        if history_text:
            parts.append(_drop_duplicate_current_from_history(history_text, message_text))
        return "\n\n".join(part for part in parts if part).strip()
    return history_text


async def _prepare_recall_query(agent, history, message, cfg, log_item) -> str:
    message_text = _message_query_text(message)
    history_text = _trim_query(
        _history_query_text(history),
        cfg["memory_recall_history_len"],
    )

    if not message_text and not history_text:
        return ""

    if cfg.get("memory_recall_query_prep", True):
        try:
            prompt_message = agent.read_prompt(
                "memory.memories_query.msg.md",
                history=history_text,
                message=message_text or "None",
            )
            query = await agent.call_utility_model(
                system=agent.read_prompt("memory.memories_query.sys.md"),
                message=prompt_message,
            )
            query = str(query or "").strip()
        except Exception as e:
            log_item.update(
                heading="Failed to generate memory query",
                content=f"memory query preparation failed: {e}",
            )
            fallback = _recall_query_text(history, message, cfg["memory_recall_history_len"])
            if fallback:
                log_item.update(
                    query_prep_fallback="query-prep failed; using current message/history fallback",
                )
            return fallback
        if _debug_enabled(cfg):
            log_item.update(
                query_prep_message=_debug_preview(prompt_message),
                query_prep_raw=_debug_preview(query),
            )
        if not query or query == "-":
            fallback = _recall_query_text(history, message, cfg["memory_recall_history_len"])
            if fallback:
                log_item.update(
                    query_prep_raw=query,
                    query_prep_fallback="query-prep returned no query; using current message/history fallback",
                )
            return fallback
        return _query_with_current_message_first(message_text, query)

    return _recall_query_text(history, message, cfg["memory_recall_history_len"])


def _debug_preview(text: str, limit: int = 4000) -> str:
    text = str(text or "")
    if len(text) <= limit:
        return text
    return text[:limit] + "...[truncated]"


def _query_with_current_message_first(message_text: str, prepared_query: str) -> str:
    message_text = str(message_text or "").strip()
    prepared_query = str(prepared_query or "").strip()
    if not message_text:
        return prepared_query
    if not prepared_query:
        return message_text
    if prepared_query == message_text or message_text in prepared_query:
        return prepared_query
    return f"{message_text}\n\n{prepared_query}"


def _content_query_text(content) -> str:
    if content is None:
        return ""

    if isinstance(content, str):
        text = _strip_extras(content).strip()
        if text.lstrip().startswith("{"):
            parsed = _parse_json_object(text)
            if parsed is not None:
                tool_text = _tool_payload_query_text(parsed)
                if tool_text is not None:
                    return tool_text
            if "tool_name" in text and "tool_args" in text:
                return ""
        return text

    if isinstance(content, dict):
        tool_text = _tool_payload_query_text(content)
        if tool_text is not None:
            return tool_text

        preferred_keys = ("user_message", "message", "text", "preview")
        for key in preferred_keys:
            if key in content:
                text = _content_query_text(content.get(key))
                if text:
                    return text

        if "raw_content" in content:
            text = _content_query_text(content.get("raw_content"))
            if text:
                return text

        parts = [
            _content_query_text(value)
            for key, value in content.items()
            if key not in preferred_keys
            and key != "raw_content"
            and key not in QUERY_METADATA_KEYS
        ]
        return "\n".join(part for part in parts if part).strip()

    if isinstance(content, list):
        parts = [_content_query_text(item) for item in content]
        return "\n".join(part for part in parts if part).strip()

    return str(content).strip()


def _is_tool_payload(content: dict) -> bool:
    if "user_message" in content:
        return False
    return any(key in content for key in TOOL_PAYLOAD_KEYS)


def _tool_payload_query_text(content: dict) -> str | None:
    if not _is_tool_payload(content):
        return None

    if str(content.get("tool_name") or "") == "response":
        tool_args = content.get("tool_args")
        if isinstance(tool_args, dict):
            return _content_query_text(tool_args.get("text"))
    return ""


def _parse_json_object(text: str) -> dict | None:
    try:
        parsed = json.loads(text)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _strip_extras(text: str) -> str:
    return str(text or "").split("[EXTRAS]", 1)[0].strip()


def _drop_duplicate_current_from_history(history_text: str, message_text: str) -> str:
    current_forms = {
        str(message_text or "").strip(),
        f"user: {str(message_text or '').strip()}",
        f"human: {str(message_text or '').strip()}",
    }
    lines = [
        line.rstrip()
        for line in str(history_text or "").splitlines()
        if line.strip() not in current_forms
    ]
    return "\n".join(lines).strip()


def _trim_query(query: str, max_len: int) -> str:
    query = str(query or "").strip()
    max_len = int(max_len or 0)
    if max_len > 0 and len(query) > max_len:
        return query[-max_len:]
    return query


def _write_extras(agent, extras, memories, solutions, log_item, feedback_items):
    if not memories and not solutions:
        log_item.update(heading="No memories or solutions found")
        return

    log_item.update(
        heading=f"{len(memories)} memories and {len(solutions)} relevant solutions found",
    )

    memories_txt = "\n\n".join(memories) if memories else ""
    solutions_txt = "\n\n".join(solutions) if solutions else ""

    if memories_txt:
        log_item.update(memories=memories_txt)
    if solutions_txt:
        log_item.update(solutions=solutions_txt)
    if feedback_items:
        log_item.update(memory_feedback_items=feedback_items)

    if memories_txt:
        extras["memories"] = agent.parse_prompt(
            "agent.system.memories.md", memories=memories_txt
        )
    if solutions_txt:
        extras["solutions"] = agent.parse_prompt(
            "agent.system.solutions.md", solutions=solutions_txt
        )
