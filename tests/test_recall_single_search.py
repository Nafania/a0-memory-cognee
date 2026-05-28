import asyncio
import importlib.util
import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeLogItem:
    def __init__(self):
        self.fields = {}

    def update(self, **kwargs):
        self.fields.update(kwargs)


class FakeHistory:
    def output(self):
        return [
            {
                "ai": False,
                "content": {
                    "system_message": [],
                    "user_message": "previous raw user message",
                    "attachments": [],
                },
            },
            {
                "ai": True,
                "content": {
                    "response": "previous assistant response",
                },
            },
        ]

    def output_text(self):
        return "previous conversation about cognee recall"


class ResponseToolHistory:
    def output(self):
        return [
            {
                "ai": False,
                "content": "а где мы живем?",
            },
            {
                "ai": True,
                "content": (
                    '{'
                    '"thoughts":["do not include internal reasoning"],'
                    '"headline":"Residence answer",'
                    '"tool_name":"response",'
                    '"tool_args":{"text":"Only Тимофей living in Вильнюсе is reliable; '
                    'the user residence is not reliably specified."}'
                    '}'
                ),
            },
            {
                "ai": True,
                "content": (
                    '{"thoughts":["tool call"],"headline":"Loading memory",'
                    '"tool_name":"memory_load",'
                    '"tool_args":{"query":"residence"}}'
                ),
            },
        ]

    def output_text(self):
        return "raw fallback should not be used"


class DirtyHistory:
    def output(self):
        return [
            {
                "ai": True,
                "content": {
                    "thoughts": ["assistant json must not affect recall"],
                    "headline": "Greeting user",
                    "tool_name": "response",
                    "tool_args": {"text": "hello"},
                },
            },
            {
                "ai": False,
                "content": {
                    "tool_name": "memory_load",
                    "tool_result": "stale memory tool result must not affect recall",
                },
            },
            {
                "ai": False,
                "content": "older user request\n[EXTRAS]\n{\"memories\":\"stale injected memory\"}",
            },
        ]

    def output_text(self):
        return "assistant: dirty fallback should not be used"


class FakeMessage:
    content = {
        "system_message": [],
        "user_message": "debug memory search",
        "attachments": [],
    }

    def output_text(self):
        return (
            'human: {"system_message": [], "user_message": '
            '"debug memory search", "attachments": []}'
        )


class EmptyHistory:
    def output(self):
        return []

    def output_text(self):
        return ""


class MetadataOnlyMessage:
    content = {
        "system_message": ["do not leak system"],
        "user_message": "",
        "attachments": [{"name": "secret attachment"}],
    }

    def output_text(self):
        return "human: metadata-only wrapper"


class FakeAgent:
    def __init__(self):
        self.history = FakeHistory()
        self.context = types.SimpleNamespace(
            id="ctx-1",
            log=types.SimpleNamespace(log=lambda **kwargs: FakeLogItem()),
        )
        self._data = {}
        self.utility_query_response = "prepared recall query"
        self.utility_calls = []

    def parse_prompt(self, name, **kwargs):
        return f"{name}:{kwargs}"

    def read_prompt(self, name, **kwargs):
        if name == "memory.memories_query.sys.md":
            return "memory query system"
        if name == "memory.memories_query.msg.md":
            return f"message={kwargs.get('message')}\nhistory={kwargs.get('history')}"
        return f"{name}:{kwargs}"

    async def call_utility_model(self, *, system, message):
        self.utility_calls.append({"system": system, "message": message})
        return self.utility_query_response

    def get_data(self, name):
        return self._data.get(name)

    def set_data(self, name, value):
        self._data[name] = value


_DEFAULT_USER_MESSAGE = object()


class FakeLoopData:
    def __init__(self, user_message=_DEFAULT_USER_MESSAGE):
        self.extras_persistent = {}
        self.user_message = (
            FakeMessage() if user_message is _DEFAULT_USER_MESSAGE else user_message
        )
        self.iteration = 0


class FakeCognee:
    def __init__(self, search_error=None):
        self.search_calls = []
        self.recall_calls = []
        self.search_error = search_error

    async def search(self, **kwargs):
        self.search_calls.append(kwargs)
        if self.search_error:
            raise self.search_error
        return ["combined-results"]

    async def recall(self, **kwargs):
        self.recall_calls.append(kwargs)
        if self.search_error:
            raise self.search_error
        return ["combined-results"]


class AreaValue:
    def __init__(self, value):
        self.value = value


class FakeMemory:
    class Area:
        MAIN = AreaValue("main")
        FRAGMENTS = AreaValue("fragments")
        SOLUTIONS = AreaValue("solutions")

    dataset_name = "default"
    memory_subdir = "default"

    @staticmethod
    async def get(agent):
        return FakeMemory()

    def get_search_datasets(self):
        return ["default"]


class FakeWorker:
    def __init__(self, block_reason=None):
        self.block_reason = block_reason

    def get_search_block_reason(self, datasets):
        return self.block_reason

    def nudge_rebuild_if_unready(self, datasets, reason=""):
        return False


def _install_stubs(
    fake_cognee: FakeCognee,
    split_calls: list,
    block_reason=None,
):
    helpers = types.ModuleType("helpers")
    extension = types.ModuleType("helpers.extension")
    plugins = types.ModuleType("helpers.plugins")
    log = types.ModuleType("helpers.log")
    print_style = types.ModuleType("helpers.print_style")

    class Extension:
        pass

    class PrintStyle:
        @staticmethod
        def error(*args, **kwargs):
            pass

    extension.Extension = Extension
    log.LogItem = FakeLogItem
    print_style.PrintStyle = PrintStyle

    cfg = {
        "memory_recall_enabled": True,
        "memory_recall_interval": 1,
        "memory_recall_history_len": 10000,
        "memory_recall_memories_max_search": 12,
        "memory_recall_solutions_max_search": 8,
        "memory_recall_memories_max_result": 5,
        "memory_recall_solutions_max_result": 3,
        "memory_recall_query_prep": True,
        "memory_session_enabled": True,
        "cognee_debug_enabled": False,
    }
    plugins.get_plugin_config = lambda name, agent=None: cfg

    package_names = [
        "usr",
        "usr.plugins",
        "usr.plugins.memory_cognee",
        "usr.plugins.memory_cognee.helpers",
        "usr.plugins.memory_cognee.extensions",
        "usr.plugins.memory_cognee.extensions.python",
        "usr.plugins.memory_cognee.extensions.python.message_loop_prompts_after",
        "cognee",
        "cognee.modules",
        "cognee.modules.engine",
        "cognee.modules.engine.models",
    ]
    for name in package_names:
        package = types.ModuleType(name)
        package.__path__ = [str(REPO_ROOT / "helpers")] if name.endswith(".helpers") else []
        sys.modules[name] = package

    memory = types.ModuleType("usr.plugins.memory_cognee.helpers.memory")

    def split_recall_answers_by_area(answers, memory_limit, solution_limit):
        split_calls.append((answers, memory_limit, solution_limit))
        return ["memory-doc"], ["solution-doc"]

    def recall_text_and_feedback_items(answers, limit, *, context_id, fallback_dataset, kind):
        return [f"{kind}:{item}" for item in answers[:limit]], [
            {"text": str(item), "kind": kind, "context_id": context_id}
            for item in answers[:limit]
        ]

    memory.Memory = FakeMemory
    memory.split_recall_answers_by_area = split_recall_answers_by_area
    memory.recall_text_and_feedback_items = recall_text_and_feedback_items

    cognee_init = types.ModuleType("usr.plugins.memory_cognee.helpers.cognee_init")
    cognee_init.get_cognee = lambda: (
        fake_cognee,
        types.SimpleNamespace(CHUNKS="CHUNKS", RAG_COMPLETION="RAG_COMPLETION"),
    )
    cognee_init.is_cognee_debug_enabled = lambda: bool(cfg.get("cognee_debug_enabled", False))

    background = types.ModuleType("usr.plugins.memory_cognee.helpers.cognee_background")

    class CogneeBackgroundWorker:
        @staticmethod
        def get_instance():
            return FakeWorker(block_reason)

    background.CogneeBackgroundWorker = CogneeBackgroundWorker

    node_set = types.ModuleType("cognee.modules.engine.models.node_set")

    class NodeSet:
        pass

    node_set.NodeSet = NodeSet

    sys.modules.update(
        {
            "helpers": helpers,
            "helpers.extension": extension,
            "helpers.plugins": plugins,
            "helpers.log": log,
            "helpers.print_style": print_style,
            "agent": types.SimpleNamespace(LoopData=FakeLoopData),
            "usr.plugins.memory_cognee.helpers.memory": memory,
            "usr.plugins.memory_cognee.helpers.cognee_init": cognee_init,
            "usr.plugins.memory_cognee.helpers.cognee_background": background,
            "cognee.modules.engine.models.node_set": node_set,
        }
    )


def _load_recall_module(
    fake_cognee: FakeCognee,
    split_calls: list,
    block_reason=None,
):
    _install_stubs(fake_cognee, split_calls, block_reason)
    module_path = (
        REPO_ROOT
        / "extensions"
        / "python"
        / "message_loop_prompts_after"
        / "_50_recall_memories.py"
    )
    spec = importlib.util.spec_from_file_location(
        "usr.plugins.memory_cognee.extensions.python.message_loop_prompts_after._50_recall_memories",
        module_path,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RecallSingleSearchTest(unittest.IsolatedAsyncioTestCase):
    def tearDown(self):
        for name in list(sys.modules):
            if (
                name == "helpers"
                or name.startswith("helpers.")
                or name == "agent"
                or name == "cognee"
                or name.startswith("cognee.")
                or name.startswith("usr.plugins.memory_cognee")
            ):
                sys.modules.pop(name, None)

    async def test_recall_uses_chunk_search_context_with_area_split(self):
        fake_cognee = FakeCognee()
        split_calls = []
        module = _load_recall_module(fake_cognee, split_calls)

        extension = module.RecallMemories()
        extension.agent = FakeAgent()
        log_item = FakeLogItem()
        loop_data = FakeLoopData()

        await extension.search_memories(log_item, loop_data)

        self.assertEqual(fake_cognee.recall_calls, [])
        self.assertEqual(len(fake_cognee.search_calls), 1)
        call = fake_cognee.search_calls[0]
        self.assertEqual(call["query_type"], "CHUNKS")
        self.assertEqual(call["top_k"], 60)
        self.assertEqual(call["datasets"], ["default"])
        self.assertIn("node_type", call)
        self.assertEqual(call["node_name"], ["main", "fragments", "solutions"])
        self.assertIs(call["only_context"], True)
        self.assertIs(call["verbose"], True)
        self.assertEqual(
            call["query_text"],
            "debug memory search\n\nprepared recall query",
        )
        self.assertEqual(log_item.fields["query"], "debug memory search\n\nprepared recall query")
        utility_message = extension.agent.utility_calls[0]["message"]
        self.assertIn("message=debug memory search", utility_message)
        self.assertIn("history=ai: previous assistant response", utility_message)
        self.assertIn("user: previous raw user message", utility_message)
        self.assertLess(
            utility_message.index("message=debug memory search"),
            utility_message.index("history=ai: previous assistant response"),
        )
        self.assertNotIn("system_message", utility_message)
        self.assertNotIn("attachments", utility_message)
        self.assertEqual(split_calls, [(["combined-results"], 12, 8)])
        self.assertIn("memories", loop_data.extras_persistent)
        self.assertIn("solutions", loop_data.extras_persistent)
        self.assertIn("memory_feedback_items", log_item.fields)
        self.assertEqual(log_item.fields["heading"], "1 memories and 1 relevant solutions found")

    async def test_recall_query_uses_raw_user_message_not_prompt_wrapper(self):
        fake_cognee = FakeCognee()
        split_calls = []
        module = _load_recall_module(fake_cognee, split_calls)

        extension = module.RecallMemories()
        extension.agent = FakeAgent()
        await extension.search_memories(FakeLogItem(), FakeLoopData())

        self.assertEqual(
            fake_cognee.search_calls[0]["query_text"],
            "debug memory search\n\nprepared recall query",
        )
        utility_message = extension.agent.utility_calls[0]["message"]
        self.assertIn("message=debug memory search", utility_message)
        self.assertIn("history=ai: previous assistant response", utility_message)
        self.assertIn("user: previous raw user message", utility_message)
        self.assertNotIn('"user_message"', utility_message)
        self.assertNotIn("system_message", utility_message)
        self.assertNotIn("attachments", utility_message)

    async def test_recall_query_preserves_current_message_when_query_prep_rewrites(self):
        fake_cognee = FakeCognee()
        split_calls = []
        module = _load_recall_module(fake_cognee, split_calls)

        extension = module.RecallMemories()
        extension.agent = FakeAgent()
        extension.agent.utility_query_response = "integration marker a0cg-run fact 02"
        current_message = "Что означает интеграционный маркер a0cg-run-fact-02?"

        await extension.search_memories(
            FakeLogItem(),
            FakeLoopData(user_message=types.SimpleNamespace(content=current_message)),
        )

        query = fake_cognee.search_calls[0]["query_text"]
        self.assertTrue(query.startswith(current_message))
        self.assertIn("integration marker a0cg-run fact 02", query)

    async def test_recall_query_uses_history_only_when_current_message_missing(self):
        fake_cognee = FakeCognee()
        split_calls = []
        module = _load_recall_module(fake_cognee, split_calls)

        extension = module.RecallMemories()
        extension.agent = FakeAgent()
        await extension.search_memories(FakeLogItem(), FakeLoopData(user_message=None))

        self.assertEqual(fake_cognee.search_calls[0]["query_text"], "prepared recall query")
        utility_message = extension.agent.utility_calls[0]["message"]
        self.assertIn("message=None", utility_message)
        self.assertIn("user: previous raw user message", utility_message)
        self.assertIn("ai: previous assistant response", utility_message)
        self.assertLess(
            utility_message.index("ai: previous assistant response"),
            utility_message.index("user: previous raw user message"),
        )

    async def test_recall_query_keeps_response_text_context_without_tool_json(self):
        fake_cognee = FakeCognee()
        split_calls = []
        module = _load_recall_module(fake_cognee, split_calls)

        extension = module.RecallMemories()
        extension.agent = FakeAgent()
        extension.agent.history = ResponseToolHistory()

        await extension.search_memories(FakeLogItem(), FakeLoopData())

        self.assertEqual(
            fake_cognee.search_calls[0]["query_text"],
            "debug memory search\n\nprepared recall query",
        )
        utility_message = extension.agent.utility_calls[0]["message"]
        self.assertIn("message=debug memory search", utility_message)
        self.assertIn(
            "ai: Only Тимофей living in Вильнюсе is reliable; "
            "the user residence is not reliably specified.",
            utility_message,
        )
        self.assertIn("user: а где мы живем?", utility_message)
        self.assertNotIn("thoughts", utility_message)
        self.assertNotIn("tool_name", utility_message)
        self.assertNotIn("memory_load", utility_message)
        self.assertNotIn("Residence answer", utility_message)

    async def test_recall_query_skips_tool_results_and_extras_from_history(self):
        fake_cognee = FakeCognee()
        split_calls = []
        module = _load_recall_module(fake_cognee, split_calls)

        extension = module.RecallMemories()
        extension.agent = FakeAgent()
        extension.agent.history = DirtyHistory()

        await extension.search_memories(FakeLogItem(), FakeLoopData(user_message=None))

        self.assertEqual(fake_cognee.search_calls[0]["query_text"], "prepared recall query")
        utility_message = extension.agent.utility_calls[0]["message"]
        self.assertIn("history=user: older user request\nai: hello", utility_message)
        self.assertNotIn("assistant json", utility_message)
        self.assertNotIn("memory_load", utility_message)
        self.assertNotIn("tool result", utility_message)
        self.assertNotIn("EXTRAS", utility_message)
        self.assertNotIn("stale injected memory", utility_message)

    async def test_recall_query_puts_current_message_before_clean_history(self):
        fake_cognee = FakeCognee()
        split_calls = []
        module = _load_recall_module(fake_cognee, split_calls)

        extension = module.RecallMemories()
        extension.agent = FakeAgent()
        extension.agent.history = DirtyHistory()

        await extension.search_memories(FakeLogItem(), FakeLoopData())

        self.assertEqual(
            fake_cognee.search_calls[0]["query_text"],
            "debug memory search\n\nprepared recall query",
        )
        utility_message = extension.agent.utility_calls[0]["message"]
        self.assertIn("message=debug memory search", utility_message)
        self.assertIn("history=user: older user request\nai: hello", utility_message)
        self.assertLess(
            utility_message.index("message=debug memory search"),
            utility_message.index("history=user: older user request"),
        )
        self.assertNotIn("assistant json", utility_message)
        self.assertNotIn("memory_load", utility_message)
        self.assertNotIn("EXTRAS", utility_message)

    async def test_recall_falls_back_to_current_message_when_query_prep_returns_dash(self):
        fake_cognee = FakeCognee()
        split_calls = []
        module = _load_recall_module(fake_cognee, split_calls)

        extension = module.RecallMemories()
        extension.agent = FakeAgent()
        extension.agent.utility_query_response = "-"
        log_item = FakeLogItem()

        await extension.search_memories(log_item, FakeLoopData())

        self.assertEqual(fake_cognee.recall_calls, [])
        self.assertEqual(len(fake_cognee.search_calls), 1)
        query = fake_cognee.search_calls[0]["query_text"]
        self.assertIn("debug memory search", query)
        self.assertIn("previous raw user message", query)
        self.assertNotIn("No relevant memory query generated", log_item.fields["query"])
        self.assertEqual(log_item.fields["query_prep_raw"], "-")
        self.assertIn("query-prep returned no query", log_item.fields["query_prep_fallback"])

    async def test_recall_falls_back_to_current_message_when_query_prep_returns_empty(self):
        fake_cognee = FakeCognee()
        split_calls = []
        module = _load_recall_module(fake_cognee, split_calls)

        extension = module.RecallMemories()
        extension.agent = FakeAgent()
        extension.agent.utility_query_response = "   "
        log_item = FakeLogItem()

        await extension.search_memories(log_item, FakeLoopData())

        self.assertEqual(fake_cognee.recall_calls, [])
        self.assertEqual(len(fake_cognee.search_calls), 1)
        self.assertIn("debug memory search", fake_cognee.search_calls[0]["query_text"])
        self.assertIn("previous assistant response", fake_cognee.search_calls[0]["query_text"])
        self.assertEqual(log_item.fields["query_prep_raw"], "")
        self.assertIn("query-prep returned no query", log_item.fields["query_prep_fallback"])

    async def test_recall_debug_logs_query_prep_and_cognee_args(self):
        fake_cognee = FakeCognee()
        split_calls = []
        module = _load_recall_module(fake_cognee, split_calls)

        extension = module.RecallMemories()
        extension.agent = FakeAgent()
        log_item = FakeLogItem()

        cfg = sys.modules["helpers.plugins"].get_plugin_config("memory_cognee")
        cfg["cognee_debug_enabled"] = True

        await extension.search_memories(log_item, FakeLoopData())

        self.assertIn("message=debug memory search", log_item.fields["query_prep_message"])
        self.assertEqual(log_item.fields["query_prep_raw"], "prepared recall query")
        self.assertEqual(
            log_item.fields["cognee_search_args"]["query_text"],
            "debug memory search\n\nprepared recall query",
        )
        self.assertEqual(log_item.fields["cognee_search_args"]["query_type"], "CHUNKS")
        self.assertEqual(log_item.fields["cognee_search_args"]["node_name"], ["main", "fragments", "solutions"])
        self.assertEqual(log_item.fields["cognee_search_result_count"], 1)

    async def test_recall_debug_uses_cognee_init_setting(self):
        fake_cognee = FakeCognee()
        split_calls = []
        module = _load_recall_module(fake_cognee, split_calls)

        module.is_cognee_debug_enabled = lambda: True

        extension = module.RecallMemories()
        extension.agent = FakeAgent()
        log_item = FakeLogItem()

        cfg = sys.modules["helpers.plugins"].get_plugin_config("memory_cognee")
        cfg["cognee_debug_enabled"] = False

        await extension.search_memories(log_item, FakeLoopData())

        self.assertIn("query_prep_message", log_item.fields)
        self.assertIn("cognee_search_args", log_item.fields)
        self.assertEqual(log_item.fields["cognee_search_args"]["query_type"], "CHUNKS")

    async def test_recall_query_ignores_metadata_only_wrappers(self):
        fake_cognee = FakeCognee()
        split_calls = []
        module = _load_recall_module(fake_cognee, split_calls)

        extension = module.RecallMemories()
        extension.agent = FakeAgent()
        extension.agent.history = EmptyHistory()
        log_item = FakeLogItem()

        await extension.search_memories(
            log_item,
            FakeLoopData(user_message=MetadataOnlyMessage()),
        )

        self.assertEqual(fake_cognee.search_calls, [])
        self.assertEqual(fake_cognee.recall_calls, [])
        self.assertIn("No relevant memory query generated", log_item.fields["query"])

    async def test_recall_skips_search_while_rebuild_not_ready(self):
        fake_cognee = FakeCognee()
        split_calls = []
        module = _load_recall_module(
            fake_cognee,
            split_calls,
            "Cognee memory graph rebuild pending for dataset(s): ['default']",
        )

        extension = module.RecallMemories()
        extension.agent = FakeAgent()
        log_item = FakeLogItem()
        loop_data = FakeLoopData()

        await extension.search_memories(log_item, loop_data)

        self.assertEqual(fake_cognee.search_calls, [])
        self.assertEqual(fake_cognee.recall_calls, [])
        self.assertEqual(split_calls, [])
        self.assertEqual(
            log_item.fields["heading"],
            "Memory rebuild pending; skipping recall",
        )
        self.assertIn("Cognee memory graph rebuild pending", log_item.fields["content"])
        self.assertEqual(loop_data.extras_persistent, {})

    async def test_recall_uses_failed_heading_for_failed_rebuild(self):
        fake_cognee = FakeCognee()
        split_calls = []
        module = _load_recall_module(
            fake_cognee,
            split_calls,
            "Cognee memory graph rebuild failed for dataset(s): ['default']",
        )

        extension = module.RecallMemories()
        extension.agent = FakeAgent()
        log_item = FakeLogItem()

        await extension.search_memories(log_item, FakeLoopData())

        self.assertEqual(log_item.fields["heading"], "Memory rebuild failed; skipping recall")
        self.assertIn("Cognee memory graph rebuild failed", log_item.fields["content"])

    async def test_recall_exception_is_not_reported_as_empty_memory(self):
        fake_cognee = FakeCognee(RuntimeError("recall backend down"))
        split_calls = []
        module = _load_recall_module(fake_cognee, split_calls)

        extension = module.RecallMemories()
        extension.agent = FakeAgent()
        log_item = FakeLogItem()
        loop_data = FakeLoopData()

        await extension.search_memories(log_item, loop_data)

        self.assertEqual(log_item.fields["heading"], "Memory recall failed")
        self.assertIn("recall backend down", log_item.fields["content"])
        self.assertEqual(loop_data.extras_persistent, {})

    async def test_recall_uses_cognee_chunk_result_shape(self):
        fake_cognee = FakeCognee()
        split_calls = []
        module = _load_recall_module(fake_cognee, split_calls)

        extension = module.RecallMemories()
        extension.agent = FakeAgent()
        await extension.search_memories(FakeLogItem(), FakeLoopData())

        self.assertEqual(fake_cognee.recall_calls, [])
        self.assertEqual(len(fake_cognee.search_calls), 1)
        self.assertEqual(fake_cognee.search_calls[0]["query_type"], "CHUNKS")
        self.assertEqual(fake_cognee.search_calls[0]["node_name"], ["main", "fragments", "solutions"])
        self.assertIs(fake_cognee.search_calls[0]["only_context"], True)
        self.assertIs(fake_cognee.search_calls[0]["verbose"], True)

    async def test_execute_reuses_inflight_recall_task(self):
        fake_cognee = FakeCognee()
        split_calls = []
        module = _load_recall_module(fake_cognee, split_calls)

        extension = module.RecallMemories()
        extension.agent = FakeAgent()
        pending = asyncio.create_task(asyncio.sleep(1))
        extension.agent.set_data(module.DATA_NAME_TASK, pending)

        try:
            await extension.execute(FakeLoopData())

            self.assertIs(extension.agent.get_data(module.DATA_NAME_TASK), pending)
            self.assertEqual(fake_cognee.search_calls, [])
            self.assertEqual(fake_cognee.recall_calls, [])
        finally:
            pending.cancel()
            try:
                await pending
            except asyncio.CancelledError:
                pass

    async def test_execute_uses_configured_recall_timeout(self):
        fake_cognee = FakeCognee()
        split_calls = []
        module = _load_recall_module(fake_cognee, split_calls)

        cfg = sys.modules["helpers.plugins"].get_plugin_config("memory_cognee")
        cfg["memory_recall_timeout_seconds"] = 123
        timeouts = []
        original_wait_for = module.asyncio.wait_for

        async def fake_wait_for(awaitable, timeout=None):
            timeouts.append(timeout)
            return await awaitable

        module.asyncio.wait_for = fake_wait_for
        try:
            extension = module.RecallMemories()
            extension.agent = FakeAgent()

            await extension.execute(FakeLoopData())
            task = extension.agent.get_data(module.DATA_NAME_TASK)
            await task

            self.assertEqual(timeouts, [123.0])
            self.assertEqual(len(fake_cognee.search_calls), 1)
        finally:
            module.asyncio.wait_for = original_wait_for


if __name__ == "__main__":
    unittest.main()
