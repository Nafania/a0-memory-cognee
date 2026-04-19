import asyncio
from helpers.extension import Extension
from usr.plugins.memory_cognee.helpers.memory import Memory, recall_text_and_feedback_items
from agent import LoopData
from helpers import plugins, log
from helpers.print_style import PrintStyle

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

        from usr.plugins.memory_cognee.helpers.cognee_init import get_cognee
        cognee, _ = get_cognee()
        from cognee.modules.engine.models.node_set import NodeSet

        db = await Memory.get(self.agent)

        datasets = db.get_search_datasets()
        mem_node_name = [Memory.Area.MAIN.value, Memory.Area.FRAGMENTS.value]
        sol_node_name = [Memory.Area.SOLUTIONS.value]

        try:
            session_id = getattr(self.agent.context, 'id', None)
            mem_answers, sol_answers = await asyncio.gather(
                cognee.recall(
                    query_text=query,
                    datasets=datasets,
                    top_k=cfg["memory_recall_memories_max_search"],
                    node_type=NodeSet,
                    node_name=mem_node_name,
                    session_id=session_id,
                    only_context=True,
                    verbose=True,
                ),
                cognee.recall(
                    query_text=query,
                    datasets=datasets,
                    top_k=cfg["memory_recall_solutions_max_search"],
                    node_type=NodeSet,
                    node_name=sol_node_name,
                    session_id=session_id,
                    only_context=True,
                    verbose=True,
                ),
            )
        except OSError as e:
            try:
                PrintStyle.error(f"cognee.recall OS error (likely too many open files): {e}")
            except OSError:
                pass
            mem_answers, sol_answers = [], []
        except Exception as e:
            try:
                PrintStyle.error(f"cognee.recall failed: {e}")
            except OSError:
                pass
            mem_answers, sol_answers = [], []

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
