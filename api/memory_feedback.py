import json

from flask import Response

from helpers.api import ApiHandler, Request
from plugins._memory_cognee.helpers import cognee_feedback as cf


class MemoryFeedback(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict | Response:
        try:
            cf.validate_feedback_payload(input)
        except cf.FeedbackPayloadError as e:
            body = {
                "success": False,
                "status": "failed",
                "error": str(e),
            }
            return Response(
                response=json.dumps(body),
                status=400,
                mimetype="application/json",
            )

        result = await cf.submit_memory_feedback(input)
        status = result.get("status", "failed")

        if status == "failed":
            body = {
                "success": False,
                "status": "failed",
                "error": result.get("error", "unknown"),
            }
            return Response(
                response=json.dumps(body),
                status=503,
                mimetype="application/json",
            )

        return {"success": True, "status": status}
