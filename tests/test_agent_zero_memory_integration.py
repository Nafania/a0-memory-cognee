import json
import os
import random
import string
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request
from http.cookiejar import CookieJar


def _integration_enabled() -> bool:
    return os.environ.get("A0_MEMORY_COGNEE_INTEGRATION") == "1"


@unittest.skipUnless(
    _integration_enabled(),
    "set A0_MEMORY_COGNEE_INTEGRATION=1 to run live Agent Zero integration test",
)
class AgentZeroMemoryIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = os.environ.get(
            "A0_INTEGRATION_BASE_URL", "http://127.0.0.1:18183"
        ).rstrip("/")
        cls.timeout = int(os.environ.get("A0_MEMORY_COGNEE_TIMEOUT", "1800"))
        cls.fact_count = int(os.environ.get("A0_MEMORY_COGNEE_FACT_COUNT", "20"))
        cls.expect_debug = os.environ.get("A0_MEMORY_COGNEE_EXPECT_DEBUG") == "1"
        if cls.fact_count < 1:
            raise AssertionError("A0_MEMORY_COGNEE_FACT_COUNT must be >= 1")
        cls.client = AgentZeroHttpClient(cls.base_url, timeout=cls.timeout)
        cls.client.initialize_csrf()
        cls.client.get_json("/api/health")

    def test_live_memory_auto_recall_across_chats(self):
        run_id = _random_id("a0cg")
        facts = _build_facts(run_id, self.fact_count)

        seed_context = self.client.create_chat()
        for fact in facts:
            result = self.client.send_message(
                fact["seed"],
                context=seed_context,
            )
            seed_context = result["context"]
            self.assertTrue(seed_context, "seed context was not returned")

        self._wait_for_cognee_ready()

        recall_context = self.client.create_chat(current_context=seed_context)
        failures = []
        for fact in facts:
            before = 0
            if recall_context:
                before = self.client.poll(recall_context).get("log_version", 0)

            result = self.client.send_message(
                fact["question"],
                context=recall_context,
            )
            recall_context = result["context"]
            self.assertTrue(recall_context, "recall context was not returned")

            snapshot = self.client.poll(recall_context, log_from=before)
            logs = snapshot.get("logs", [])
            auto_recall_text = _auto_recall_text(logs)
            response_text = str(result.get("message") or "")

            if "No relevant memory query generated" in auto_recall_text:
                failures.append(
                    f"{fact['marker']}: auto-recall query-prep skipped search"
                )
                continue
            if "skipping recall" in auto_recall_text.lower():
                failures.append(f"{fact['marker']}: auto-recall skipped recall")
                continue
            if not _has_non_empty_query(logs):
                failures.append(f"{fact['marker']}: auto-recall query missing")
                continue
            if self.expect_debug and not _has_debug_trace(logs):
                failures.append(f"{fact['marker']}: auto-recall debug trace missing")
                continue
            if fact["value"] not in auto_recall_text:
                failures.append(
                    f"{fact['marker']}: expected value missing from auto-recall; "
                    f"response_has_value={fact['value'] in response_text}"
                )

        if failures:
            self.fail(
                "Live Agent Zero auto-recall failed:\n"
                + "\n".join(failures[:20])
            )

    def _wait_for_cognee_ready(self):
        deadline = time.monotonic() + self.timeout
        last_status = {}
        while time.monotonic() < deadline:
            status = self.client.post_json(
                "/api/plugins/memory_cognee/memory_dashboard",
                {"action": "cognify_status"},
            )
            last_status = status
            if not status.get("success", False):
                raise AssertionError(f"cognify_status failed: {status}")
            last_error = status.get("last_error")
            if last_error:
                raise AssertionError(f"Cognee rebuild failed: {last_error}")
            running = bool(status.get("running"))
            dirty = status.get("dirty_datasets") or []
            if not running and not dirty:
                return
            time.sleep(5)
        raise AssertionError(f"Cognee rebuild did not finish: {last_status}")


class AgentZeroHttpClient:
    def __init__(self, base_url: str, timeout: int = 1800):
        self.base_url = base_url
        self.timeout = timeout
        self.cookie_jar = CookieJar()
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(self.cookie_jar)
        )
        self.csrf_token = ""
        self.csrf_cookie_name = ""

    def initialize_csrf(self):
        data = self.get_json("/api/csrf_token")
        token = str(data.get("token") or "")
        runtime_id = str(data.get("runtime_id") or "")
        if not token or not runtime_id:
            raise AssertionError(f"Invalid csrf response: {data}")
        self.csrf_token = token
        self.csrf_cookie_name = f"csrf_token_{runtime_id}"

    def send_message(self, text: str, context: str = "") -> dict:
        return self.post_json(
            "/api/message",
            {
                "text": text,
                "context": context,
                "message_id": _random_id("msg"),
            },
            timeout=self.timeout,
        )

    def create_chat(self, current_context: str = "") -> str:
        new_context = _random_id("ctx")
        result = self.post_json(
            "/api/chat_create",
            {
                "current_context": current_context,
                "new_context": new_context,
            },
        )
        if not result.get("ok") or result.get("ctxid") != new_context:
            raise AssertionError(f"chat_create failed: {result}")
        return new_context

    def poll(self, context: str, log_from: int = 0) -> dict:
        return self.post_json(
            "/api/poll",
            {
                "context": context,
                "log_from": int(log_from or 0),
                "notifications_from": 0,
                "timezone": "Europe/Paris",
            },
        )

    def get_json(self, path: str) -> dict:
        req = urllib.request.Request(
            self.base_url + path,
            headers={
                "Accept": "application/json",
                "Origin": self.base_url,
            },
            method="GET",
        )
        return self._open_json(req)

    def post_json(self, path: str, payload: dict, timeout: int | None = None) -> dict:
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-CSRF-Token": self.csrf_token,
            "Origin": self.base_url,
        }
        req = urllib.request.Request(
            self.base_url + path,
            data=body,
            headers=headers,
            method="POST",
        )
        return self._open_json(req, timeout=timeout)

    def _open_json(self, req: urllib.request.Request, timeout: int | None = None) -> dict:
        try:
            with self.opener.open(req, timeout=timeout or 60) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise AssertionError(
                f"HTTP {exc.code} for {req.full_url}: {body}"
            ) from exc
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise AssertionError(f"Non-JSON response from {req.full_url}: {raw}") from exc


def _build_facts(run_id: str, count: int) -> list[dict]:
    facts = []
    for idx in range(1, count + 1):
        marker = f"{run_id}-fact-{idx:02d}"
        value = f"value-{idx:02d}-{_random_id('v')}"
        facts.append(
            {
                "marker": marker,
                "value": value,
                "seed": (
                    "Запомни постоянный факт без уточнений: "
                    f"интеграционный маркер {marker} означает {value}. "
                    "Ответь одной короткой фразой, что факт сохранен."
                ),
                "question": f"Что означает интеграционный маркер {marker}?",
            }
        )
    return facts


def _auto_recall_text(logs: list[dict]) -> str:
    chunks = []
    for item in logs:
        if item.get("type") != "util":
            continue
        text = _log_item_text(item)
        lower = text.lower()
        if (
            "memory" in lower
            or "memories" in lower
            or "recall" in lower
            or "searching memories" in lower
            or "памят" in lower
        ):
            chunks.append(text)
    return "\n".join(chunks)


def _has_non_empty_query(logs: list[dict]) -> bool:
    for item in logs:
        if item.get("type") != "util":
            continue
        kvps = item.get("kvps") or {}
        query = str(kvps.get("query") or item.get("query") or "").strip()
        if query and query not in {"-", "None"} and "No relevant" not in query:
            return True
    return False


def _has_debug_trace(logs: list[dict]) -> bool:
    seen_keys = set()
    for item in logs:
        if item.get("type") != "util":
            continue
        kvps = item.get("kvps") or {}
        seen_keys.update(kvps.keys())
    return {"query_prep_message", "query_prep_raw", "cognee_search_args"}.issubset(
        seen_keys
    )


def _log_item_text(item: dict) -> str:
    parts = [
        str(item.get("heading") or ""),
        str(item.get("content") or ""),
        json.dumps(item.get("kvps") or {}, ensure_ascii=False, sort_keys=True),
    ]
    return "\n".join(part for part in parts if part)


def _random_id(prefix: str) -> str:
    alphabet = string.ascii_lowercase + string.digits
    suffix = "".join(random.choice(alphabet) for _ in range(10))
    return f"{prefix}-{suffix}"


if __name__ == "__main__":
    unittest.main()
