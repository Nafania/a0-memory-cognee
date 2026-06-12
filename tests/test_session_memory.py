import importlib.util
import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeHistory:
    def output(self):
        return [
            {"ai": False, "content": {"user_message": "first user", "attachments": []}},
            {
                "ai": True,
                "content": (
                    '{"thoughts":["hidden"],"tool_name":"memory_load",'
                    '"tool_args":{"query":"ignore"}}'
                ),
            },
            {"ai": False, "content": "how are family names stored?"},
            {
                "ai": True,
                "content": (
                    '{"thoughts":["hidden"],"tool_name":"response",'
                    '"tool_args":{"text":"Family names are stored in session."}}'
                ),
            },
        ]


class FakeAgent:
    def __init__(self):
        self.history = FakeHistory()
        self.context = types.SimpleNamespace(id="ctx-session")
        self._data = {}

    def get_data(self, name):
        return self._data.get(name)

    def set_data(self, name, value):
        self._data[name] = value


class FakeMemory:
    dataset_name = "default"
    get_calls = []

    @staticmethod
    async def get(agent, **kwargs):
        FakeMemory.get_calls.append(kwargs)
        return FakeMemory()


class FakeCognee:
    def __init__(self):
        self.remember_calls = []
        self.operation_calls = []

    async def remember(self, *args, **kwargs):
        self.remember_calls.append((args, kwargs))
        return types.SimpleNamespace(status="session_stored")


def _load_session_memory(fake_cognee, *, plugin_config=None, background_worker=None):
    helpers = types.ModuleType("helpers")
    plugins = types.ModuleType("helpers.plugins")
    print_style = types.ModuleType("helpers.print_style")

    class PrintStyle:
        @staticmethod
        def warning(*args, **kwargs):
            pass

    config = {
        "memory_session_enabled": True,
        "cognee_operation_timeout_seconds": 777,
        "memory_session_idle_seconds": 0,
    }
    if plugin_config:
        config.update(plugin_config)
    plugins.get_plugin_config = lambda name, agent=None: dict(config)
    print_style.PrintStyle = PrintStyle

    package_names = [
        "usr",
        "usr.plugins",
        "usr.plugins.memory_cognee",
        "usr.plugins.memory_cognee.helpers",
        "cognee",
    ]
    for name in package_names:
        package = types.ModuleType(name)
        package.__path__ = [str(REPO_ROOT / "helpers")] if name.endswith(".helpers") else []
        sys.modules[name] = package

    cognee_init = types.ModuleType("usr.plugins.memory_cognee.helpers.cognee_init")
    cognee_init.get_cognee = lambda: (fake_cognee, types.SimpleNamespace())

    cognee_ops = types.ModuleType("usr.plugins.memory_cognee.helpers.cognee_ops")

    async def run_cognee_operation(name, fn, *args, **kwargs):
        fake_cognee.operation_calls.append((name, args, dict(kwargs)))
        operation_kwargs = dict(kwargs)
        operation_kwargs.pop("timeout", None)
        operation_kwargs.pop("operation_timeout", None)
        return await fn(*args, **operation_kwargs)

    cognee_ops.run_cognee_operation = run_cognee_operation

    memory = types.ModuleType("usr.plugins.memory_cognee.helpers.memory")
    memory.Memory = FakeMemory
    cognee_background = types.ModuleType("usr.plugins.memory_cognee.helpers.cognee_background")

    class CogneeBackgroundWorker:
        @staticmethod
        def get_instance():
            return background_worker

    cognee_background.CogneeBackgroundWorker = CogneeBackgroundWorker

    cognee_memory = types.ModuleType("cognee.memory")

    class QAEntry:
        def __init__(self, *, question, answer, context=""):
            self.question = question
            self.answer = answer
            self.context = context

    cognee_memory.QAEntry = QAEntry

    sys.modules.update(
        {
            "helpers": helpers,
            "helpers.plugins": plugins,
            "helpers.print_style": print_style,
            "usr.plugins.memory_cognee.helpers.cognee_init": cognee_init,
            "usr.plugins.memory_cognee.helpers.cognee_ops": cognee_ops,
            "usr.plugins.memory_cognee.helpers.memory": memory,
            "usr.plugins.memory_cognee.helpers.cognee_background": cognee_background,
            "cognee.memory": cognee_memory,
        }
    )

    module_path = REPO_ROOT / "helpers" / "session_memory.py"
    spec = importlib.util.spec_from_file_location(
        "usr.plugins.memory_cognee.helpers.session_memory",
        module_path,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class SessionMemoryTest(unittest.IsolatedAsyncioTestCase):
    def tearDown(self):
        for name in list(sys.modules):
            if (
                name == "helpers"
                or name.startswith("helpers.")
                or name == "cognee"
                or name.startswith("cognee.")
                or name.startswith("usr.plugins.memory_cognee")
            ):
                sys.modules.pop(name, None)

    async def test_remember_session_turn_stores_latest_response_qa(self):
        fake_cognee = FakeCognee()
        module = _load_session_memory(fake_cognee)
        agent = FakeAgent()
        FakeMemory.get_calls = []

        ok = await module.remember_session_turn(agent)

        self.assertTrue(ok)
        self.assertEqual(len(fake_cognee.remember_calls), 1)
        args, kwargs = fake_cognee.remember_calls[0]
        entry = args[0]
        self.assertEqual(entry.question, "how are family names stored?")
        self.assertEqual(entry.answer, "Family names are stored in session.")
        self.assertEqual(kwargs["dataset_name"], "default")
        self.assertEqual(kwargs["session_id"], "ctx-session")
        self.assertIs(kwargs["self_improvement"], False)
        operation_name, _operation_args, operation_kwargs = fake_cognee.operation_calls[0]
        self.assertEqual(operation_name, "cognee.remember session")
        self.assertEqual(operation_kwargs["timeout"], 777.0)
        self.assertEqual(operation_kwargs["operation_timeout"], 777.0)
        self.assertEqual(operation_kwargs["priority"], "background")
        self.assertEqual(FakeMemory.get_calls[0], {"preload_knowledge": False})
        self.assertIn(module.DATA_NAME_LAST_SESSION_QA, agent._data)

    async def test_remember_session_turn_skips_duplicate_turn(self):
        fake_cognee = FakeCognee()
        module = _load_session_memory(fake_cognee)
        agent = FakeAgent()

        self.assertTrue(await module.remember_session_turn(agent))
        self.assertFalse(await module.remember_session_turn(agent))

        self.assertEqual(len(fake_cognee.remember_calls), 1)

    async def test_extract_latest_qa_skips_tool_payloads(self):
        module = _load_session_memory(FakeCognee())

        question, answer = module.extract_latest_qa(FakeHistory())

        self.assertEqual(question, "how are family names stored?")
        self.assertEqual(answer, "Family names are stored in session.")

    async def test_safe_remember_waits_until_memory_idle(self):
        class FakeBackgroundWorker:
            def __init__(self):
                self.checks = 0

            def is_memory_idle(self, idle_seconds):
                self.checks += 1
                return self.checks >= 2

        fake_worker = FakeBackgroundWorker()
        fake_cognee = FakeCognee()
        module = _load_session_memory(
            fake_cognee,
            plugin_config={"memory_session_idle_seconds": 10},
            background_worker=fake_worker,
        )
        sleeps = []

        async def fake_sleep(delay):
            sleeps.append(delay)

        original_sleep = module.asyncio.sleep
        module.asyncio.sleep = fake_sleep

        try:
            await module.safe_remember_session_turn(FakeAgent())
        finally:
            module.asyncio.sleep = original_sleep

        self.assertEqual(fake_worker.checks, 2)
        self.assertEqual(sleeps, [2.5])
        self.assertEqual(len(fake_cognee.remember_calls), 1)


if __name__ == "__main__":
    unittest.main()
