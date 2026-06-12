from usr.plugins.memory_cognee.helpers.memory import Memory, SearchUnavailable
from usr.plugins.memory_cognee.tools.memory_load import DEFAULT_THRESHOLD
from helpers.tool import Tool, Response


class MemoryForget(Tool):

    async def execute(self, query="", threshold=DEFAULT_THRESHOLD, filter="", **kwargs) -> Response:
        db = await Memory.get(self.agent, preload_knowledge=False)
        try:
            dels = await db.delete_documents_by_query(query=query, threshold=threshold, filter=filter)
        except SearchUnavailable as e:
            return Response(
                message=f"Memory search unavailable: {e}",
                break_loop=False,
            )

        result = self.agent.read_prompt("fw.memories_deleted.md", memory_count=len(dels))
        return Response(message=result, break_loop=False)
