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


class FakeAgent:
    def __init__(self):
        self.history = FakeHistory()
        self.context = types.SimpleNamespace(
            id="ctx-1",
            log=types.SimpleNamespace(log=lambda **kwargs: FakeLogItem()),
        )
        self._data = {}

    def parse_prompt(self, name, **kwargs):
        return f"{name}:{kwargs}"

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
        self.search_error = search_error

    async def search(self, **kwargs):
        self.search_calls.append(kwargs)
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
        types.SimpleNamespace(CHUNKS="CHUNKS"),
    )

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

    async def test_recall_uses_one_combined_search_then_splits_results(self):
        fake_cognee = FakeCognee()
        split_calls = []
        module = _load_recall_module(fake_cognee, split_calls)

        extension = module.RecallMemories()
        extension.agent = FakeAgent()
        log_item = FakeLogItem()
        loop_data = FakeLoopData()

        await extension.search_memories(log_item, loop_data)

        self.assertEqual(len(fake_cognee.search_calls), 1)
        call = fake_cognee.search_calls[0]
        self.assertEqual(call["query_type"], "CHUNKS")
        self.assertEqual(call["top_k"], 20)
        self.assertEqual(call["datasets"], ["default"])
        self.assertNotIn("node_type", call)
        self.assertNotIn("node_name", call)
        self.assertIs(call["only_context"], False)
        self.assertIs(call["verbose"], False)
        self.assertIn("debug memory search", call["query_text"])
        self.assertNotIn("previous raw user message", call["query_text"])
        self.assertNotIn("system_message", call["query_text"])
        self.assertNotIn("attachments", call["query_text"])
        self.assertEqual(
            split_calls,
            [
                (["combined-results"], 12, 8),
            ],
        )
        self.assertIn("memories", loop_data.extras_persistent)
        self.assertIn("solutions", loop_data.extras_persistent)
        self.assertEqual(log_item.fields["heading"], "1 memories and 1 relevant solutions found")

    async def test_recall_query_uses_raw_user_message_not_prompt_wrapper(self):
        fake_cognee = FakeCognee()
        split_calls = []
        module = _load_recall_module(fake_cognee, split_calls)

        extension = module.RecallMemories()
        extension.agent = FakeAgent()
        await extension.search_memories(FakeLogItem(), FakeLoopData())

        query = fake_cognee.search_calls[0]["query_text"]
        self.assertEqual(query, "debug memory search")
        self.assertNotIn('"user_message"', query)

    async def test_recall_query_uses_history_only_when_current_message_missing(self):
        fake_cognee = FakeCognee()
        split_calls = []
        module = _load_recall_module(fake_cognee, split_calls)

        extension = module.RecallMemories()
        extension.agent = FakeAgent()
        await extension.search_memories(FakeLogItem(), FakeLoopData(user_message=None))

        query = fake_cognee.search_calls[0]["query_text"]
        self.assertIn("previous raw user message", query)
        self.assertIn("previous assistant response", query)

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

    async def test_recall_search_exception_is_not_reported_as_empty_memory(self):
        fake_cognee = FakeCognee(RuntimeError("search backend down"))
        split_calls = []
        module = _load_recall_module(fake_cognee, split_calls)

        extension = module.RecallMemories()
        extension.agent = FakeAgent()
        log_item = FakeLogItem()
        loop_data = FakeLoopData()

        await extension.search_memories(log_item, loop_data)

        self.assertEqual(log_item.fields["heading"], "Memory recall failed")
        self.assertIn("search backend down", log_item.fields["content"])
        self.assertEqual(loop_data.extras_persistent, {})

    async def test_recall_uses_raw_ranked_chunks_result_shape(self):
        fake_cognee = FakeCognee()
        split_calls = []
        module = _load_recall_module(fake_cognee, split_calls)

        extension = module.RecallMemories()
        extension.agent = FakeAgent()
        await extension.search_memories(FakeLogItem(), FakeLoopData())

        self.assertEqual(len(fake_cognee.search_calls), 1)
        self.assertEqual(fake_cognee.search_calls[0]["query_type"], "CHUNKS")
        self.assertIs(fake_cognee.search_calls[0]["verbose"], False)

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
        finally:
            pending.cancel()
            try:
                await pending
            except asyncio.CancelledError:
                pass


if __name__ == "__main__":
    unittest.main()
