import importlib.util
import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_cognee_init_module(order: list[str]):
    helpers = types.ModuleType("helpers")
    dotenv = types.ModuleType("helpers.dotenv")
    files = types.ModuleType("helpers.files")
    settings = types.ModuleType("helpers.settings")
    print_style = types.ModuleType("helpers.print_style")

    dotenv.load_dotenv = lambda: None
    dotenv.get_dotenv_value = lambda key, default=None: default
    files.get_abs_path = lambda *parts: "/tmp/" + "/".join(parts)
    settings.get_settings = lambda: {"api_keys": {}}

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

    class CogneeBackgroundWorker:
        @staticmethod
        def get_instance():
            return types.SimpleNamespace(start=lambda: order.append("start"))

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

        module.init_cognee = init_cognee

        module.run_memory_cognee_init_a0_extension()

        self.assertEqual(order, ["configure", "init", "migrate", "start"])


if __name__ == "__main__":
    unittest.main()
