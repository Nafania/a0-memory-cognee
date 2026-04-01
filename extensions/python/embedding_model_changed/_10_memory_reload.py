from helpers.extension import Extension

from usr.plugins._memory_cognee.helpers.memory import reload as memory_reload


class MemoryReload(Extension):

    async def execute(self, **kwargs):
        memory_reload()
