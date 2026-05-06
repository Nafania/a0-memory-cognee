import asyncio
import importlib.util
import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _install_stubs(cognee_module):
    helpers = types.ModuleType("helpers")

    defer = types.ModuleType("helpers.defer")
    defer.THREAD_BACKGROUND = "background"

    class DeferredTask:
        def __init__(self, *args, **kwargs):
            pass

        def start_task(self, *args, **kwargs):
            pass

    defer.DeferredTask = DeferredTask

    print_style = types.ModuleType("helpers.print_style")

    class PrintStyle:
        messages: list[tuple[str, tuple]] = []

        @classmethod
        def standard(cls, *args, **kwargs):
            cls.messages.append(("standard", args))

        @classmethod
        def warning(cls, *args, **kwargs):
            cls.messages.append(("warning", args))

        @classmethod
        def error(cls, *args, **kwargs):
            cls.messages.append(("error", args))

    print_style.PrintStyle = PrintStyle

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

    cognee_init = types.ModuleType("usr.plugins.memory_cognee.helpers.cognee_init")
    cognee_init.get_cognee_setting = lambda key, default=None: default

    sys.modules.update(
        {
            "helpers": helpers,
            "helpers.defer": defer,
            "helpers.print_style": print_style,
            "usr.plugins.memory_cognee.helpers.cognee_init": cognee_init,
            "cognee": cognee_module,
        }
    )
    return PrintStyle


def _load_background_module(cognee_module):
    _install_stubs(cognee_module)
    module_path = REPO_ROOT / "helpers" / "cognee_background.py"
    spec = importlib.util.spec_from_file_location(
        "usr.plugins.memory_cognee.helpers.cognee_background",
        module_path,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class CogneeBackgroundTest(unittest.TestCase):
    def tearDown(self):
        for name in list(sys.modules):
            if name == "cognee" or name.startswith("usr.plugins.memory_cognee"):
                sys.modules.pop(name, None)

    def test_empty_graph_improve_error_is_non_fatal(self):
        class FakeCognee(types.ModuleType):
            def __init__(self):
                super().__init__("cognee")
                self.cognified = []
                self.improved = []

            async def cognify(self, *, datasets, temporal_cognify):
                self.cognified.append((datasets, temporal_cognify))

            async def improve(self, *, dataset):
                self.improved.append(dataset)
                raise RuntimeError(
                    "EntityNotFoundError: Empty graph projected from the database."
                )

        fake_cognee = FakeCognee()
        background = _load_background_module(fake_cognee)
        worker = background.CogneeBackgroundWorker()
        worker.mark_dirty("default")

        asyncio.run(worker.run_pipeline())

        status = worker.get_status()
        self.assertTrue(status["last_run_success"])
        self.assertIsNone(status["last_error"])
        self.assertEqual(status["dirty_datasets"], [])
        self.assertEqual(fake_cognee.improved, ["default"])


if __name__ == "__main__":
    unittest.main()
