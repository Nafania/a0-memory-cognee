from usr.plugins.memory_cognee.helpers.memory import Memory, SearchUnavailable
from helpers.tool import Tool, Response

DEFAULT_THRESHOLD = 0.7
DEFAULT_LIMIT = 10


class MemoryLoad(Tool):

    async def execute(self, query="", threshold=DEFAULT_THRESHOLD, limit=DEFAULT_LIMIT, filter="", **kwargs) -> Response:
        db = await Memory.get(self.agent, preload_knowledge=False)
        session_id = getattr(self.agent.context, 'id', None)
        try:
            docs = await db.search_similarity_threshold(
                query=query,
                limit=limit,
                threshold=threshold,
                filter=filter,
                session_id=session_id,
                raise_unavailable=True,
            )
        except SearchUnavailable as e:
            return Response(
                message=f"Memory search unavailable: {e}",
                break_loop=False,
            )

        if len(docs) == 0:
            result = self.agent.read_prompt("fw.memories_not_found.md", query=query)
        else:
            text = "\n\n".join(Memory.format_docs_plain(docs))
            result = str(text)

        return Response(message=result, break_loop=False)
