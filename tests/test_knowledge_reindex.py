import asyncio
import importlib.util
import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeWorker:
    def __init__(self):
        self.dirty: list[str] = []

    def mark_dirty(self, dataset_name: str):
        self.dirty.append(dataset_name)


def _load_reindex_module(worker: FakeWorker):
    helpers_api = types.ModuleType("helpers.api")

    class ApiHandler:
        pass

    helpers_api.ApiHandler = ApiHandler
    helpers_api.Request = dict
    helpers_api.Response = dict

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
    cognee_init = types.ModuleType("usr.plugins.memory_cognee.helpers.cognee_init")
    cognee_init.reset_calls = []

    async def reset_cognify_status_for_all_datasets():
        cognee_init.reset_calls.append("all")

    cognee_init.reset_cognify_status_for_all_datasets = reset_cognify_status_for_all_datasets

    class Memory:
        dataset_name = "default"

        @staticmethod
        async def reload(agent):
            return Memory()

    memory.Memory = Memory

    background = types.ModuleType("usr.plugins.memory_cognee.helpers.cognee_background")

    class CogneeBackgroundWorker:
        @staticmethod
        def get_instance():
            return worker

    background.CogneeBackgroundWorker = CogneeBackgroundWorker

    sys.modules.update(
        {
            "helpers.api": helpers_api,
            "usr.plugins.memory_cognee.helpers.memory": memory,
            "usr.plugins.memory_cognee.helpers.cognee_init": cognee_init,
            "usr.plugins.memory_cognee.helpers.cognee_background": background,
        }
    )

    module_path = REPO_ROOT / "api" / "knowledge_reindex.py"
    spec = importlib.util.spec_from_file_location(
        "usr.plugins.memory_cognee.api.knowledge_reindex",
        module_path,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class KnowledgeReindexTest(unittest.TestCase):
    def tearDown(self):
        for name in list(sys.modules):
            if name.startswith("helpers.") or name.startswith("usr.plugins.memory_cognee"):
                sys.modules.pop(name, None)

    def test_reindex_marks_reloaded_dataset_dirty(self):
        worker = FakeWorker()
        module = _load_reindex_module(worker)
        handler = module.ReindexKnowledge()
        context = types.SimpleNamespace(
            agent0=object(),
            log=types.SimpleNamespace(set_initial_progress=lambda: None),
        )
        handler.use_context = lambda ctxid: context

        result = asyncio.run(handler.process({"ctxid": "ctx"}, {}))

        self.assertEqual(result["ok"], True)
        cognee_init = sys.modules["usr.plugins.memory_cognee.helpers.cognee_init"]
        self.assertEqual(cognee_init.reset_calls, ["all"])
        self.assertEqual(worker.dirty, ["default"])


if __name__ == "__main__":
    unittest.main()
