from helpers import plugins, errors
from helpers.extension import Extension
from usr.plugins.memory_cognee.helpers.memory import Memory
from usr.plugins.memory_cognee.helpers.llm_json import (
    format_llm_json_for_log,
    parse_llm_json_response,
)
from usr.plugins.memory_cognee.helpers.memory_write_worker import MemoryWriteWorker
from helpers.dirty_json import DirtyJson
from agent import LoopData
from helpers.log import LogItem
from helpers.defer import DeferredTask, THREAD_BACKGROUND
from usr.plugins.memory_cognee.helpers.deferred_tasks import track_deferred_task


class MemorizeMemories(Extension):

    async def execute(self, loop_data: LoopData = LoopData(), **kwargs):
        cfg = plugins.get_plugin_config("memory_cognee", self.agent)
        if not cfg:
            return

        if not cfg["memory_memorize_enabled"]:
            return

        log_item = self.agent.context.log.log(
            type="util",
            heading="Memorizing new information...",
        )

        task = DeferredTask(thread_name=THREAD_BACKGROUND)
        task.start_task(self.memorize, loop_data, log_item, None, cfg)
        return track_deferred_task(task)

    async def memorize(self, loop_data: LoopData, log_item: LogItem, db: Memory, cfg: dict, **kwargs):
        try:
            system = self.agent.read_prompt("memory.memories_sum.sys.md")
            msgs_text = self.agent.concat_messages(self.agent.history)

            memories_json = await self.agent.call_utility_model(
                system=system,
                message=msgs_text,
                background=True,
            )

            if not memories_json or not isinstance(memories_json, str):
                log_item.update(heading="No response from utility model.")
                return

            memories_json = memories_json.strip()

            if not memories_json:
                log_item.update(heading="Empty response from utility model.")
                return

            try:
                memories = parse_llm_json_response(memories_json, DirtyJson.parse_string)
            except Exception as e:
                log_item.update(
                    heading=f"Failed to parse memories response: {str(e)}",
                    content=memories_json,
                )
                return

            if memories is None:
                log_item.update(heading="No valid memories found in response.")
                return

            if not isinstance(memories, list):
                if isinstance(memories, (str, dict)):
                    memories = [memories]
                else:
                    log_item.update(heading="Invalid memories format received.")
                    return

            log_item.update(content=format_llm_json_for_log(memories))

            if not isinstance(memories, list) or len(memories) == 0:
                log_item.update(heading="No useful information to memorize.")
                return
            else:
                memories_txt = "\n\n".join([str(memory) for memory in memories]).strip()
                log_item.update(heading=f"{len(memories)} entries to memorize.", memories=memories_txt)

            use_consolidation = cfg.get("memory_memorize_consolidation", False)
            replace_threshold = cfg.get("memory_memorize_replace_threshold", 0.9)
            area = Memory.Area.FRAGMENTS.value
            worker = MemoryWriteWorker.get_instance()
            queued = 0
            for memory in memories:
                queued = worker.enqueue(
                    agent=self.agent,
                    text=f"{memory}",
                    area=area,
                    metadata={"area": area},
                    cfg=cfg,
                    use_consolidation=use_consolidation,
                    replace_threshold=replace_threshold,
                    similarity_threshold=cfg.get("memory_recall_similarity_threshold", 0.7),
                )

            log_item.update(
                result=f"{len(memories)} entries queued for memory write.",
                heading=f"{len(memories)} entries queued for memory write.",
                memory_ids=[],
                queued_memory_count=len(memories),
                memory_write_queue_size=queued,
            )

        except Exception as e:
            err = errors.format_error(e)
            self.agent.context.log.log(
                type="warning", heading="Memorize memories extension error", content=err
            )
