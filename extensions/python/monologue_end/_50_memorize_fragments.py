from helpers import plugins, errors
from helpers.extension import Extension
from usr.plugins.memory_cognee.helpers.memory import Memory
from helpers.dirty_json import DirtyJson
from agent import LoopData
from helpers.log import LogItem
from helpers.defer import DeferredTask, THREAD_BACKGROUND


class MemorizeMemories(Extension):

    async def execute(self, loop_data: LoopData = LoopData(), **kwargs):
        cfg = plugins.get_plugin_config("memory_cognee", self.agent)
        if not cfg:
            return

        if not cfg["memory_memorize_enabled"]:
            return

        db = await Memory.get(self.agent)

        log_item = self.agent.context.log.log(
            type="util",
            heading="Memorizing new information...",
        )

        task = DeferredTask(thread_name=THREAD_BACKGROUND)
        task.start_task(self.memorize, loop_data, log_item, db)
        return task

    async def memorize(self, loop_data: LoopData, log_item: LogItem, db: Memory, **kwargs):
        try:
            system = self.agent.read_prompt("memory.memories_sum.sys.md")
            msgs_text = self.agent.concat_messages(self.agent.history)

            memories_json = await self.agent.call_utility_model(
                system=system,
                message=msgs_text,
                background=True,
            )

            log_item.update(content=memories_json)

            if not memories_json or not isinstance(memories_json, str):
                log_item.update(heading="No response from utility model.")
                return

            memories_json = memories_json.strip()

            if not memories_json:
                log_item.update(heading="Empty response from utility model.")
                return

            try:
                memories = DirtyJson.parse_string(memories_json)
            except Exception as e:
                log_item.update(heading=f"Failed to parse memories response: {str(e)}")
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

            if not isinstance(memories, list) or len(memories) == 0:
                log_item.update(heading="No useful information to memorize.")
                return
            else:
                memories_txt = "\n\n".join([str(memory) for memory in memories]).strip()
                log_item.update(heading=f"{len(memories)} entries to memorize.", memories=memories_txt)

            for memory in memories:
                txt = f"{memory}"
                await db.insert_text(text=txt, metadata={"area": Memory.Area.FRAGMENTS.value})

            log_item.update(
                result=f"{len(memories)} entries memorized.",
                heading=f"{len(memories)} entries memorized.",
            )

        except Exception as e:
            err = errors.format_error(e)
            self.agent.context.log.log(
                type="warning", heading="Memorize memories extension error", content=err
            )
