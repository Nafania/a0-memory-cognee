import asyncio
from helpers.extension import Extension
from helpers.print_style import PrintStyle
from agent import LoopData
from usr.plugins._memory_cognee.extensions.python.message_loop_prompts_after._50_recall_memories import DATA_NAME_TASK as DATA_NAME_TASK_MEMORIES, DATA_NAME_ITER as DATA_NAME_ITER_MEMORIES
from helpers import plugins

class RecallWait(Extension):
    async def execute(self, loop_data: LoopData = LoopData(), **kwargs):

        set = plugins.get_plugin_config("_memory_cognee", self.agent)
        if not set:
            return

        task = self.agent.get_data(DATA_NAME_TASK_MEMORIES)
        iter = self.agent.get_data(DATA_NAME_ITER_MEMORIES) or 0

        if task and not task.done():

            if set["memory_recall_delayed"]:
                if iter == loop_data.iteration:
                    delay_text = self.agent.read_prompt("memory.recall_delay_msg.md")
                    loop_data.extras_temporary["memory_recall_delayed"] = delay_text
                    return

            try:
                await task
            except (TimeoutError, asyncio.TimeoutError):
                try:
                    PrintStyle.error("Memory recall timed out, continuing without memories")
                except OSError:
                    pass
            except asyncio.CancelledError:
                try:
                    PrintStyle.error("Memory recall was cancelled")
                except OSError:
                    pass
            except Exception as e:
                try:
                    PrintStyle.error(f"Memory recall failed: {e}")
                except OSError:
                    pass
