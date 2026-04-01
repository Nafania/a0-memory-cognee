from helpers.extension import Extension
from agent import LoopData
from usr.plugins.memory_cognee.helpers import memory
import asyncio


class MemoryInit(Extension):

    async def execute(self, loop_data: LoopData = LoopData(), **kwargs):
        db = await memory.Memory.get(self.agent)
