from helpers.extension import Extension
from agent import LoopData
from usr.plugins.memory_cognee.helpers import memory
from helpers.print_style import PrintStyle


class MemoryInit(Extension):

    async def execute(self, loop_data: LoopData = LoopData(), **kwargs):
        db = await memory.Memory.get(self.agent, preload_knowledge=False)
        try:
            from usr.plugins.memory_cognee.helpers.cognee_background import (
                CogneeBackgroundWorker,
            )

            block_reason = CogneeBackgroundWorker.get_instance().get_search_block_reason(
                db.get_search_datasets()
            )
            if block_reason:
                PrintStyle.warning(f"Cognee knowledge preload skipped: {block_reason}")
                return
        except Exception as e:
            PrintStyle.warning(f"Cognee knowledge preload readiness check failed: {e}")
            return

        await memory.Memory.get(self.agent)
