from helpers.api import ApiHandler, Request, Response
from helpers import files, notification, projects, notification
from plugins._memory_cognee.helpers import memory
import os


class ReindexKnowledge(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict | Response:
        ctxid = input.get("ctxid", "")
        if not ctxid:
            raise Exception("No context id provided")
        context = self.use_context(ctxid)

        await memory.Memory.reload(context.agent0)
        context.log.set_initial_progress()

        return {
            "ok": True,
            "message": "Knowledge re-indexed",
        }
