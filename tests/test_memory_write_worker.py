import asyncio
import importlib.util
import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_worker_module():
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

    defer = types.ModuleType("helpers.defer")
    defer.THREAD_BACKGROUND = "background"

    class DeferredTask:
        def __init__(self, thread_name=None):
            self.thread_name = thread_name
            self.started = []

        def start_task(self, fn, *args, **kwargs):
            self.started.append((fn, args, kwargs))

        def is_alive(self):
            return False

    defer.DeferredTask = DeferredTask

    print_style = types.ModuleType("helpers.print_style")

    class PrintStyle:
        @staticmethod
        def standard(*args, **kwargs):
            pass

        @staticmethod
        def warning(*args, **kwargs):
            pass

        @staticmethod
        def error(*args, **kwargs):
            pass

    print_style.PrintStyle = PrintStyle

    memory = types.ModuleType("usr.plugins.memory_cognee.helpers.memory")

    class Memory:
        @staticmethod
        async def get(agent, **kwargs):
            return types.SimpleNamespace(dataset_name="default")

    memory.Memory = Memory

    async def insert_with_simple_dedup(db, text, area, threshold):
        return f"simple:{text}"

    memory.insert_with_simple_dedup = insert_with_simple_dedup

    consolidation = types.ModuleType("usr.plugins.memory_cognee.helpers.memory_consolidation")
    consolidation.calls = []
    consolidation.result = None

    class Consolidator:
        async def process_new_memory(self, **kwargs):
            consolidation.calls.append(kwargs)
            if consolidation.result is not None:
                return consolidation.result
            return {"success": True, "memory_ids": [f"consolidated:{kwargs['new_memory']}"]}

    consolidation.create_memory_consolidator = lambda *args, **kwargs: Consolidator()

    background = types.ModuleType("usr.plugins.memory_cognee.helpers.cognee_background")

    class CogneeBackgroundWorker:
        idle = True

        @staticmethod
        def get_instance():
            return CogneeBackgroundWorker()

        def is_memory_idle(self, idle_seconds):
            return self.idle

    background.CogneeBackgroundWorker = CogneeBackgroundWorker

    sys.modules.update(
        {
            "helpers.defer": defer,
            "helpers.print_style": print_style,
            "usr.plugins.memory_cognee.helpers.memory": memory,
            "usr.plugins.memory_cognee.helpers.memory_consolidation": consolidation,
            "usr.plugins.memory_cognee.helpers.cognee_background": background,
        }
    )

    module_path = REPO_ROOT / "helpers" / "memory_write_worker.py"
    spec = importlib.util.spec_from_file_location(
        "usr.plugins.memory_cognee.helpers.memory_write_worker",
        module_path,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, consolidation, background


class MemoryWriteWorkerTest(unittest.TestCase):
    def tearDown(self):
        for name in list(sys.modules):
            if (
                name == "helpers.defer"
                or name == "helpers.print_style"
                or name.startswith("usr.plugins.memory_cognee")
            ):
                sys.modules.pop(name, None)

    def test_enqueue_starts_background_worker_without_processing_inline(self):
        module, consolidation, _background = _load_worker_module()
        worker = module.MemoryWriteWorker()

        queued = worker.enqueue(
            agent=object(),
            text="queued memory",
            area="fragments",
            metadata={"area": "fragments"},
            cfg={
                "memory_consolidation_idle_seconds": 0,
                "memory_consolidation_retry_seconds": 0,
            },
            use_consolidation=True,
            replace_threshold=0.9,
            similarity_threshold=0.7,
        )

        self.assertEqual(queued, 1)
        self.assertEqual(consolidation.calls, [])
        self.assertTrue(worker._task.started)

    def test_worker_processes_consolidation_job_from_queue(self):
        module, consolidation, _background = _load_worker_module()
        worker = module.MemoryWriteWorker()
        worker.enqueue(
            agent=object(),
            text="queued memory",
            area="fragments",
            metadata={"area": "fragments"},
            cfg={
                "memory_consolidation_idle_seconds": 0,
                "memory_consolidation_retry_seconds": 0,
            },
            use_consolidation=True,
            replace_threshold=0.9,
            similarity_threshold=0.7,
        )

        asyncio.run(worker.run_loop())

        self.assertEqual(len(consolidation.calls), 1)
        self.assertEqual(consolidation.calls[0]["new_memory"], "queued memory")

    def test_worker_waits_until_memory_is_idle(self):
        module, consolidation, background = _load_worker_module()
        background.CogneeBackgroundWorker.idle = False
        sleeps = []

        async def fake_sleep(delay):
            sleeps.append(delay)
            background.CogneeBackgroundWorker.idle = True

        original_sleep = module.asyncio.sleep
        module.asyncio.sleep = fake_sleep
        try:
            worker = module.MemoryWriteWorker()
            worker.enqueue(
                agent=object(),
                text="queued memory",
                area="fragments",
                metadata={"area": "fragments"},
                cfg={"memory_consolidation_idle_seconds": 60},
                use_consolidation=True,
                replace_threshold=0.9,
                similarity_threshold=0.7,
            )

            asyncio.run(worker.run_loop())
        finally:
            module.asyncio.sleep = original_sleep

        self.assertTrue(sleeps)
        self.assertEqual(len(consolidation.calls), 1)

    def test_worker_keeps_failed_consolidation_job_for_retry(self):
        module, consolidation, _background = _load_worker_module()
        consolidation.result = {
            "success": False,
            "memory_ids": [],
            "search_unavailable": True,
            "reason": "Cognee memory graph rebuild pending",
        }
        worker = module.MemoryWriteWorker()
        worker.enqueue(
            agent=object(),
            text="queued memory",
            area="fragments",
            metadata={"area": "fragments"},
            cfg={
                "memory_consolidation_idle_seconds": 0,
                "memory_consolidation_retry_seconds": 0,
            },
            use_consolidation=True,
            replace_threshold=0.9,
            similarity_threshold=0.7,
        )

        asyncio.run(worker.run_loop())
        status = worker.get_status()

        self.assertEqual(status["processed"], 0)
        self.assertEqual(status["queued"], 1)
        self.assertIn("Cognee memory graph rebuild pending", status["last_error"])


if __name__ == "__main__":
    unittest.main()
