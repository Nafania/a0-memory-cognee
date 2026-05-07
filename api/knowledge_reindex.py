from helpers.api import ApiHandler, Request, Response
from usr.plugins.memory_cognee.helpers import memory


class ReindexKnowledge(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict | Response:
        ctxid = input.get("ctxid", "")
        if not ctxid:
            raise Exception("No context id provided")
        context = self.use_context(ctxid)

        mem = await memory.Memory.reload(context.agent0)
        from usr.plugins.memory_cognee.helpers.cognee_init import (
            reset_cognify_status_for_all_datasets,
        )

        await reset_cognify_status_for_all_datasets()
        from usr.plugins.memory_cognee.helpers.cognee_background import (
            CogneeBackgroundWorker,
        )

        CogneeBackgroundWorker.get_instance().mark_dirty(mem.dataset_name)
        context.log.set_initial_progress()

        return {
            "ok": True,
            "message": "Knowledge re-indexed",
        }
