import asyncio
from helpers.extension import Extension
from usr.plugins.memory_cognee.helpers.memory import (
    Memory,
    recall_text_and_feedback_items,
    split_recall_answers_by_area,
)
from agent import LoopData
from helpers import plugins, log
from helpers.print_style import PrintStyle
from usr.plugins.memory_cognee.helpers.cognee_ops import run_cognee_operation

DATA_NAME_TASK = "_recall_memories_task"
DATA_NAME_ITER = "_recall_memories_iter"
SEARCH_TIMEOUT = 30


class RecallMemories(Extension):

    async def execute(self, loop_data: LoopData = LoopData(), **kwargs):
        cfg = plugins.get_plugin_config("memory_cognee", self.agent)
        if not cfg:
            return

        if not cfg["memory_recall_enabled"]:
            return

        existing_task = self.agent.get_data(DATA_NAME_TASK)
        if existing_task and not existing_task.done():
            self.agent.set_data(DATA_NAME_ITER, loop_data.iteration)
            return

        if loop_data.iteration % cfg["memory_recall_interval"] == 0:
            log_item = self.agent.context.log.log(
                type="util",
                heading="Searching memories...",
            )

            task = asyncio.create_task(
                asyncio.wait_for(
                    self.search_memories(loop_data=loop_data, log_item=log_item, **kwargs),
                    timeout=SEARCH_TIMEOUT,
                )
            )
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

        user_instruction = (
            loop_data.user_message.output_text() if loop_data.user_message else "None"
        )
        history = self.agent.history.output_text()[-cfg["memory_recall_history_len"]:]

        query = user_instruction + "\n\n" + history

        if not query or len(query) <= 3:
            log_item.update(
                query="No relevant memory query generated, skipping search",
            )
            return

        db = await Memory.get(self.agent)

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
        cognee, _ = get_cognee()
        from cognee.modules.engine.models.node_set import NodeSet

        mem_node_name = [Memory.Area.MAIN.value, Memory.Area.FRAGMENTS.value]
        sol_node_name = [Memory.Area.SOLUTIONS.value]
        combined_node_name = mem_node_name + sol_node_name
        memory_search_limit = cfg["memory_recall_memories_max_search"]
        solution_search_limit = cfg["memory_recall_solutions_max_search"]

        try:
            session_id = getattr(self.agent.context, 'id', None)
            combined_answers = await run_cognee_operation(
                "cognee.search recall",
                cognee.search,
                query_text=query,
                top_k=memory_search_limit + solution_search_limit,
                datasets=datasets,
                node_type=NodeSet,
                node_name=combined_node_name,
                session_id=session_id,
                only_context=True,
                # Cognee verbose controls result shape: objects_result carries node metadata.
                verbose=True,
            )
            mem_answers, sol_answers = split_recall_answers_by_area(
                combined_answers,
                memory_search_limit,
                solution_search_limit,
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

        if not mem_answers and not sol_answers:
            from usr.plugins.memory_cognee.helpers.cognee_background import (
                CogneeBackgroundWorker,
            )

            CogneeBackgroundWorker.get_instance().nudge_rebuild_if_unready(
                datasets,
                "recall search returned empty context",
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
