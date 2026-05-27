import importlib.util
import asyncio
import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_cognee_init_module(
    order: list[str],
    dirty_datasets: list[str] | None = None,
):
    helpers = types.ModuleType("helpers")
    dotenv = types.ModuleType("helpers.dotenv")
    files = types.ModuleType("helpers.files")
    settings = types.ModuleType("helpers.settings")
    plugins = types.ModuleType("helpers.plugins")
    print_style = types.ModuleType("helpers.print_style")

    dotenv.load_dotenv = lambda: None
    dotenv.get_dotenv_value = lambda key, default=None: default
    files.get_abs_path = lambda *parts: "/tmp/" + "/".join(parts)
    settings.get_settings = lambda: {"api_keys": {}}
    plugins.get_enabled_plugins = lambda agent=None: ["_memory", "memory_cognee"]
    plugins.toggle_plugin = lambda plugin_name, enabled, **kwargs: order.append(
        f"toggle:{plugin_name}:{enabled}"
    )

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

    package_names = [
        "usr",
        "usr.plugins",
        "usr.plugins.memory_cognee",
        "usr.plugins.memory_cognee.helpers",
    ]
    for name in package_names:
        package = types.ModuleType(name)
        package.__path__ = [str(REPO_ROOT / "helpers")] if name.endswith(".helpers") else []
        sys.modules[name] = package

    background = types.ModuleType("usr.plugins.memory_cognee.helpers.cognee_background")

    dirty_datasets = dirty_datasets or []

    class CogneeBackgroundWorker:
        @staticmethod
        def get_instance():
            async def run_pipeline():
                order.append("rebuild")
                dirty_datasets.clear()

            return types.SimpleNamespace(
                get_status=lambda: {"dirty_datasets": list(dirty_datasets)},
                run_pipeline=run_pipeline,
                mark_dirty=lambda dataset_name, **kwargs: order.append(f"dirty:{dataset_name}"),
                start=lambda: order.append("start"),
            )

    background.CogneeBackgroundWorker = CogneeBackgroundWorker

    faiss_migration = types.ModuleType("usr.plugins.memory_cognee.helpers.faiss_migration")

    async def run_migration():
        order.append("migrate")
        return True

    faiss_migration.run_migration = run_migration

    sys.modules.update(
        {
            "helpers": helpers,
            "helpers.dotenv": dotenv,
            "helpers.files": files,
            "helpers.settings": settings,
            "helpers.plugins": plugins,
            "helpers.print_style": print_style,
            "usr.plugins.memory_cognee.helpers.cognee_background": background,
            "usr.plugins.memory_cognee.helpers.faiss_migration": faiss_migration,
        }
    )

    module_path = REPO_ROOT / "helpers" / "cognee_init.py"
    spec = importlib.util.spec_from_file_location(
        "usr.plugins.memory_cognee.helpers.cognee_init",
        module_path,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class StartupFaissMigrationTest(unittest.TestCase):
    def tearDown(self):
        for name in list(sys.modules):
            if (
                name == "helpers"
                or name.startswith("helpers.")
                or name.startswith("usr.plugins.memory_cognee")
            ):
                sys.modules.pop(name, None)

    def test_init_a0_runs_faiss_migration_after_cognee_init_before_worker(self):
        order: list[str] = []
        module = _load_cognee_init_module(order)

        module.configure_cognee = lambda: order.append("configure")

        async def init_cognee():
            order.append("init")
            module._cognee_module = object()
            module._init_done = True

        module.init_cognee = init_cognee

        module.run_memory_cognee_init_a0_extension()

        self.assertEqual(order, ["toggle:_memory:False", "configure", "init", "migrate"])
        module.run_memory_cognee_start_worker_extension()
        self.assertEqual(
            order, ["toggle:_memory:False", "configure", "init", "migrate", "start"]
        )

    def test_init_a0_starts_worker_without_synchronous_rebuild(self):
        order: list[str] = []
        module = _load_cognee_init_module(order, dirty_datasets=["default"])

        module.configure_cognee = lambda: order.append("configure")

        async def init_cognee():
            order.append("init")
            module._cognee_module = object()
            module._init_done = True

        module.init_cognee = init_cognee

        module.run_memory_cognee_init_a0_extension()

        self.assertEqual(order, ["toggle:_memory:False", "configure", "init", "migrate"])
        module.run_memory_cognee_start_worker_extension()
        self.assertEqual(
            order, ["toggle:_memory:False", "configure", "init", "migrate", "start"]
        )

    def test_init_a0_skips_builtin_memory_toggle_when_already_disabled(self):
        order: list[str] = []
        module = _load_cognee_init_module(order)

        from helpers import plugins

        plugins.get_enabled_plugins = lambda agent=None: ["memory_cognee"]
        module.configure_cognee = lambda: order.append("configure")

        async def init_cognee():
            order.append("init")
            module._cognee_module = object()
            module._init_done = True

        module.init_cognee = init_cognee

        module.run_memory_cognee_init_a0_extension()

        self.assertEqual(order, ["configure", "init", "migrate"])
        module.run_memory_cognee_start_worker_extension()
        self.assertEqual(order, ["configure", "init", "migrate", "start"])

    def test_init_extension_runs_before_agent_zero_components(self):
        base = REPO_ROOT / "extensions" / "python" / "_functions"

        self.assertTrue(
            (base / "__main__" / "init_a0" / "start" / "_20_init_cognee.py").exists()
        )
        self.assertTrue(
            (base / "run_ui" / "init_a0" / "start" / "_20_init_cognee.py").exists()
        )
        self.assertFalse(
            (base / "__main__" / "init_a0" / "end" / "_20_init_cognee.py").exists()
        )
        self.assertFalse(
            (base / "run_ui" / "init_a0" / "end" / "_20_init_cognee.py").exists()
        )
        self.assertTrue(
            (base / "__main__" / "init_a0" / "end" / "_90_start_cognee_worker.py").exists()
        )
        self.assertTrue(
            (base / "run_ui" / "init_a0" / "end" / "_90_start_cognee_worker.py").exists()
        )

    def test_reset_cognify_status_marks_dirty_even_without_pipeline_rows(self):
        order: list[str] = []
        module = _load_cognee_init_module(order)

        fake_cognee = types.ModuleType("cognee")

        class Datasets:
            async def list_datasets(self):
                return [types.SimpleNamespace(id="dataset-id", name="default")]

        fake_cognee.datasets = Datasets()
        async def delete_pipeline_runs(dataset_ids):
            return 0

        module._delete_pipeline_runs_for_dataset_ids = delete_pipeline_runs
        old_cognee = sys.modules.get("cognee")
        sys.modules["cognee"] = fake_cognee
        try:
            asyncio.run(module._reset_cognify_status_for_datasets({"dataset-id"}))
        finally:
            if old_cognee is None:
                sys.modules.pop("cognee", None)
            else:
                sys.modules["cognee"] = old_cognee

        self.assertIn("dirty:default", order)


if __name__ == "__main__":
    unittest.main()
