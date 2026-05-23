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
    def output_text(self):
        return "previous conversation about cognee recall"


class FakeMessage:
    def output_text(self):
        return "debug memory search"


class FakeAgent:
    def __init__(self):
        self.history = FakeHistory()
        self.context = types.SimpleNamespace(id="ctx-1")

    def parse_prompt(self, name, **kwargs):
        return f"{name}:{kwargs}"


class FakeLoopData:
    def __init__(self):
        self.extras_persistent = {}
        self.user_message = FakeMessage()
        self.iteration = 0


class FakeCognee:
    def __init__(self):
        self.search_calls = []

    async def search(self, **kwargs):
        self.search_calls.append(kwargs)
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
        package.__path__ = []
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
    cognee_init.get_cognee = lambda: (fake_cognee, None)

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
        self.assertEqual(call["top_k"], 20)
        self.assertEqual(call["datasets"], ["default"])
        self.assertEqual(call["node_name"], ["main", "fragments", "solutions"])
        self.assertIs(call["verbose"], True)
        self.assertEqual(split_calls, [(["combined-results"], 12, 8)])
        self.assertIn("memories", loop_data.extras_persistent)
        self.assertIn("solutions", loop_data.extras_persistent)
        self.assertEqual(log_item.fields["heading"], "1 memories and 1 relevant solutions found")

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
            "Memory rebuild in progress; skipping recall",
        )
        self.assertIn("Cognee memory graph rebuild pending", log_item.fields["content"])
        self.assertEqual(loop_data.extras_persistent, {})

    async def test_recall_keeps_verbose_result_shape_for_metadata(self):
        fake_cognee = FakeCognee()
        split_calls = []
        module = _load_recall_module(fake_cognee, split_calls)

        extension = module.RecallMemories()
        extension.agent = FakeAgent()
        await extension.search_memories(FakeLogItem(), FakeLoopData())

        self.assertEqual(len(fake_cognee.search_calls), 1)
        self.assertIs(fake_cognee.search_calls[0]["verbose"], True)


if __name__ == "__main__":
    unittest.main()
