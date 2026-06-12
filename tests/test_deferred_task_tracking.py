import asyncio
import importlib.util
import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_tracking_module():
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

    module_path = REPO_ROOT / "helpers" / "deferred_tasks.py"
    spec = importlib.util.spec_from_file_location(
        "usr.plugins.memory_cognee.helpers.deferred_tasks",
        module_path,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_monologue_extension(filename: str, tracked: list):
    package_names = [
        "usr",
        "usr.plugins",
        "usr.plugins.memory_cognee",
        "usr.plugins.memory_cognee.helpers",
        "usr.plugins.memory_cognee.extensions",
        "usr.plugins.memory_cognee.extensions.python",
        "usr.plugins.memory_cognee.extensions.python.monologue_end",
    ]
    for name in package_names:
        package = types.ModuleType(name)
        package.__path__ = []
        sys.modules[name] = package

    plugins = types.ModuleType("helpers.plugins")
    plugins.get_plugin_config = lambda plugin_name, agent=None: {
        "memory_memorize_enabled": True,
        "memory_session_enabled": True,
        "memory_memorize_consolidation": False,
        "memory_memorize_replace_threshold": 0.9,
        "memory_recall_similarity_threshold": 0.7,
    }

    errors = types.ModuleType("helpers.errors")
    errors.format_error = lambda e: str(e)

    extension = types.ModuleType("helpers.extension")

    class Extension:
        def __init__(self, agent=None, **kwargs):
            self.agent = agent

    extension.Extension = Extension

    defer = types.ModuleType("helpers.defer")
    defer.THREAD_BACKGROUND = "background"

    class DeferredTask:
        def __init__(self, thread_name=None):
            self.thread_name = thread_name
            self.started = []

        def start_task(self, fn, *args, **kwargs):
            self.started.append((fn, args, kwargs))
            return self

        def is_alive(self):
            return True

    defer.DeferredTask = DeferredTask

    agent_module = types.ModuleType("agent")
    agent_module.LoopData = object

    log_module = types.ModuleType("helpers.log")
    log_module.LogItem = object

    dirty_json = types.ModuleType("helpers.dirty_json")
    dirty_json.DirtyJson = types.SimpleNamespace(parse_string=lambda value: [])

    memory = types.ModuleType("usr.plugins.memory_cognee.helpers.memory")

    class Memory:
        Area = types.SimpleNamespace(
            FRAGMENTS=types.SimpleNamespace(value="fragments"),
            SOLUTIONS=types.SimpleNamespace(value="solutions"),
        )

    memory.Memory = Memory

    llm_json = types.ModuleType("usr.plugins.memory_cognee.helpers.llm_json")
    llm_json.format_llm_json_for_log = lambda value: str(value)
    llm_json.parse_llm_json_response = lambda value, parser: []

    memory_write_worker = types.ModuleType(
        "usr.plugins.memory_cognee.helpers.memory_write_worker"
    )
    memory_write_worker.MemoryWriteWorker = types.SimpleNamespace(
        get_instance=lambda: types.SimpleNamespace(enqueue=lambda **kwargs: 1)
    )

    session_memory = types.ModuleType("usr.plugins.memory_cognee.helpers.session_memory")

    async def safe_remember_session_turn(agent):
        return None

    session_memory.safe_remember_session_turn = safe_remember_session_turn

    deferred_tasks = types.ModuleType("usr.plugins.memory_cognee.helpers.deferred_tasks")

    def track_deferred_task(task):
        tracked.append(task)
        return task

    deferred_tasks.track_deferred_task = track_deferred_task

    helpers = types.ModuleType("helpers")
    helpers.plugins = plugins
    helpers.errors = errors

    sys.modules.update(
        {
            "helpers": helpers,
            "helpers.plugins": plugins,
            "helpers.errors": errors,
            "helpers.extension": extension,
            "helpers.defer": defer,
            "helpers.log": log_module,
            "helpers.dirty_json": dirty_json,
            "agent": agent_module,
            "usr.plugins.memory_cognee.helpers.memory": memory,
            "usr.plugins.memory_cognee.helpers.llm_json": llm_json,
            "usr.plugins.memory_cognee.helpers.memory_write_worker": memory_write_worker,
            "usr.plugins.memory_cognee.helpers.session_memory": session_memory,
            "usr.plugins.memory_cognee.helpers.deferred_tasks": deferred_tasks,
        }
    )

    module_path = REPO_ROOT / "extensions" / "python" / "monologue_end" / filename
    module_name = filename.removesuffix(".py")
    spec = importlib.util.spec_from_file_location(
        f"usr.plugins.memory_cognee.extensions.python.monologue_end.{module_name}",
        module_path,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class DeferredTaskTrackingTest(unittest.TestCase):
    def tearDown(self):
        for name in list(sys.modules):
            if (
                name == "agent"
                or name == "helpers"
                or name.startswith("helpers.")
                or name.startswith("usr.plugins.memory_cognee")
            ):
                sys.modules.pop(name, None)

    def test_track_deferred_task_keeps_alive_refs_and_prunes_finished(self):
        module = _load_tracking_module()

        class FakeTask:
            def __init__(self, alive):
                self.alive = alive

            def is_alive(self):
                return self.alive

        first = FakeTask(True)
        second = FakeTask(False)

        self.assertIs(module.track_deferred_task(first), first)
        self.assertEqual(module.tracked_deferred_task_count(), 1)

        module.track_deferred_task(second)
        self.assertEqual(module.tracked_deferred_task_count(), 2)

        first.alive = False
        module.track_deferred_task(FakeTask(True))
        self.assertEqual(module.tracked_deferred_task_count(), 1)

    def test_monologue_end_extensions_track_deferred_tasks(self):
        cases = [
            ("_40_remember_session_turn.py", "RememberSessionTurn"),
            ("_50_memorize_fragments.py", "MemorizeMemories"),
            ("_60_memorize_solutions.py", "MemorizeSolutions"),
        ]
        for filename, class_name in cases:
            with self.subTest(filename=filename):
                tracked = []
                module = _load_monologue_extension(filename, tracked)
                agent = types.SimpleNamespace(
                    context=types.SimpleNamespace(
                        log=types.SimpleNamespace(log=lambda **kwargs: object())
                    ),
                    read_prompt=lambda *args, **kwargs: "",
                    concat_messages=lambda history: "",
                    history=[],
                    call_utility_model=lambda **kwargs: "[]",
                )
                extension = getattr(module, class_name)(agent=agent)

                result = asyncio.run(extension.execute())

                self.assertIs(result, tracked[0])
                self.assertEqual(len(tracked), 1)
                self.assertTrue(tracked[0].started)


if __name__ == "__main__":
    unittest.main()
