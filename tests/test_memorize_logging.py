import importlib.util
import sys
import types
import unittest
from pathlib import Path

from helpers import llm_json


REPO_ROOT = Path(__file__).resolve().parents[1]
QUEUED_JOBS = []


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


def _install_stubs():
    global QUEUED_JOBS
    QUEUED_JOBS = []
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
    memory.insert_with_simple_dedup = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("memorize must enqueue writes, not call Cognee inline")
    )

    memory_write_worker = types.ModuleType("usr.plugins.memory_cognee.helpers.memory_write_worker")

    class MemoryWriteWorker:
        @staticmethod
        def get_instance():
            return MemoryWriteWorker()

        def enqueue(self, **kwargs):
            QUEUED_JOBS.append(kwargs)
            return len(QUEUED_JOBS)

    memory_write_worker.MemoryWriteWorker = MemoryWriteWorker

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
            "usr.plugins.memory_cognee.helpers.memory_write_worker": memory_write_worker,
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
    def tearDown(self):
        global CONSOLIDATION_RESULTS
        CONSOLIDATION_RESULTS = None
        for name in list(sys.modules):
            if (
                name == "helpers"
                or name.startswith("helpers.")
                or name == "agent"
                or name.startswith("usr.plugins.memory_cognee")
            ):
                sys.modules.pop(name, None)

    async def test_logs_normalized_json_and_all_memory_ids(self):
        global QUEUED_JOBS
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
        self.assertEqual(log_item.fields["queued_memory_count"], 2)
        self.assertEqual(log_item.fields["memory_ids"], [])
        self.assertEqual([job["text"] for job in QUEUED_JOBS], ["one", "two"])
        self.assertTrue(all(job["use_consolidation"] for job in QUEUED_JOBS))

    async def test_memorize_reports_queued_not_memorized_before_worker_runs(self):
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

        self.assertEqual(log_item.fields["heading"], "2 entries queued for memory write.")
        self.assertNotEqual(log_item.fields["heading"], "2 entries memorized.")


if __name__ == "__main__":
    unittest.main()
