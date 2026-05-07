import importlib.util
import sys
import types
import unittest
from pathlib import Path

from helpers import llm_json


REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeLogItem:
    def __init__(self):
        self.fields = {}

    def update(self, **kwargs):
        self.fields.update(kwargs)


class FakeAgent:
    history = []

    def read_prompt(self, name):
        return name

    def concat_messages(self, history):
        return ""

    async def call_utility_model(self, **kwargs):
        return '["one", "two"]["one", "two"]'


class FakeConsolidator:
    def __init__(self):
        self.index = 0

    async def process_new_memory(self, **kwargs):
        self.index += 1
        return {"success": True, "memory_ids": [f"mem-{self.index}"]}


def _install_stubs():
    helpers = types.ModuleType("helpers")

    plugins = types.ModuleType("helpers.plugins")
    errors = types.ModuleType("helpers.errors")
    extension = types.ModuleType("helpers.extension")
    dirty_json = types.ModuleType("helpers.dirty_json")
    log = types.ModuleType("helpers.log")
    defer = types.ModuleType("helpers.defer")

    errors.format_error = str
    extension.Extension = object

    class DirtyJson:
        @staticmethod
        def parse_string(value):
            raise AssertionError("strict parser should handle concatenated arrays")

    dirty_json.DirtyJson = DirtyJson
    log.LogItem = FakeLogItem
    defer.DeferredTask = object
    defer.THREAD_BACKGROUND = "background"

    package_names = [
        "usr",
        "usr.plugins",
        "usr.plugins.memory_cognee",
        "usr.plugins.memory_cognee.helpers",
    ]
    for name in package_names:
        package = types.ModuleType(name)
        package.__path__ = []
        sys.modules[name] = package

    memory = types.ModuleType("usr.plugins.memory_cognee.helpers.memory")

    class AreaValue:
        value = "fragments"

    class Area:
        FRAGMENTS = AreaValue()

    class Memory:
        pass

    Memory.Area = Area

    memory.Memory = Memory
    memory.insert_with_simple_dedup = None

    consolidation = types.ModuleType("usr.plugins.memory_cognee.helpers.memory_consolidation")
    consolidation.create_memory_consolidator = lambda *args, **kwargs: FakeConsolidator()

    sys.modules.update(
        {
            "helpers": helpers,
            "helpers.plugins": plugins,
            "helpers.errors": errors,
            "helpers.extension": extension,
            "helpers.dirty_json": dirty_json,
            "helpers.log": log,
            "helpers.defer": defer,
            "agent": types.SimpleNamespace(LoopData=dict),
            "usr.plugins.memory_cognee.helpers.memory": memory,
            "usr.plugins.memory_cognee.helpers.llm_json": llm_json,
            "usr.plugins.memory_cognee.helpers.memory_consolidation": consolidation,
        }
    )


def _load_memorize_memories_module():
    _install_stubs()
    module_path = (
        REPO_ROOT / "extensions" / "python" / "monologue_end" / "_50_memorize_fragments.py"
    )
    spec = importlib.util.spec_from_file_location("memory_cognee_memorize_fragments", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MemorizeLoggingTest(unittest.IsolatedAsyncioTestCase):
    async def test_logs_normalized_json_and_all_memory_ids(self):
        module = _load_memorize_memories_module()
        extension = module.MemorizeMemories()
        extension.agent = FakeAgent()
        log_item = FakeLogItem()

        await extension.memorize(
            {},
            log_item,
            db=None,
            cfg={
                "memory_memorize_consolidation": True,
                "memory_memorize_replace_threshold": 0.9,
                "memory_recall_similarity_threshold": 0.7,
            },
        )

        self.assertEqual(log_item.fields["content"], '[\n  "one",\n  "two"\n]')
        self.assertEqual(log_item.fields["memory_ids"], ["mem-1", "mem-2"])


if __name__ == "__main__":
    unittest.main()
