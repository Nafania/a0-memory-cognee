from helpers.api import ApiHandler, Request, Response
from usr.plugins._memory_cognee.helpers import cognee_feedback as cf


class MemoryFeedback(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict | Response:
        try:
            cf.validate_feedback_payload(input)
        except cf.FeedbackPayloadError as e:
            return {
                "success": False,
                "status": "failed",
                "error": str(e),
            }

        result = await cf.submit_memory_feedback(input)
        status = result.get("status", "failed")

        if status == "failed":
            return {
                "success": False,
                "status": "failed",
                "error": result.get("error", "unknown"),
            }

        return {"success": True, "status": status}
