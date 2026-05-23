import unittest
import importlib.util
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class InstallDependenciesTest(unittest.TestCase):
    def test_requirements_do_not_pin_litellm_below_cognee_requirement(self):
        requirements = (REPO_ROOT / "requirements.txt").read_text()

        self.assertNotIn("litellm==", requirements)

    def test_requirements_pin_latest_stable_cognee(self):
        requirements = (REPO_ROOT / "requirements.txt").read_text()

        self.assertIn("cognee[fastembed]==1.1.0", requirements)

    def test_cognee_debug_logs_are_disabled_by_default(self):
        default_config = (REPO_ROOT / "default_config.yaml").read_text()

        self.assertIn("cognee_debug_enabled: false", default_config)

    def test_install_allows_openai_two_for_cognee_litellm(self):
        hooks = (REPO_ROOT / "hooks.py").read_text()

        self.assertIn('pinned_openai = "openai<3"', hooks)
        self.assertNotIn("openai=={ver}", hooks)

    def test_install_runs_cognee_init_in_fresh_python_process(self):
        module_path = REPO_ROOT / "hooks.py"
        spec = importlib.util.spec_from_file_location(
            "memory_cognee_hooks_under_test",
            module_path,
        )
        assert spec and spec.loader
        hooks = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hooks)

        calls = []

        def fake_run(*args, **kwargs):
            return types.SimpleNamespace(stdout="Version: 1.99.5\n")

        def fake_check_call(cmd):
            calls.append(cmd)
            return 0

        hooks.subprocess.run = fake_run
        hooks.subprocess.check_call = fake_check_call

        helpers = types.ModuleType("helpers")
        plugins = types.ModuleType("helpers.plugins")
        plugins.toggle_plugin = lambda *args, **kwargs: None
        plugins.after_plugin_change = lambda *args, **kwargs: None

        print_style = types.ModuleType("helpers.print_style")

        class PrintStyle:
            @staticmethod
            def standard(*args, **kwargs):
                pass

            @staticmethod
            def warning(*args, **kwargs):
                pass

        print_style.PrintStyle = PrintStyle
        helpers.plugins = plugins
        helpers.print_style = print_style

        cognee_init = types.ModuleType("usr.plugins.memory_cognee.helpers.cognee_init")
        cognee_init.ensure_tables_sync = lambda: (_ for _ in ()).throw(
            AssertionError("Cognee init must run in a fresh Python subprocess")
        )
        faiss_migration = types.ModuleType(
            "usr.plugins.memory_cognee.helpers.faiss_migration"
        )
        faiss_migration.migrate = lambda: (_ for _ in ()).throw(
            AssertionError("FAISS migration must run in a fresh Python subprocess")
        )

        module_stubs = {
            "helpers": helpers,
            "helpers.plugins": plugins,
            "helpers.print_style": print_style,
            "usr.plugins.memory_cognee.helpers.cognee_init": cognee_init,
            "usr.plugins.memory_cognee.helpers.faiss_migration": faiss_migration,
        }
        old_modules = {name: sys.modules.get(name) for name in module_stubs}
        sys.modules.update(module_stubs)
        try:
            hooks.install()
        finally:
            for name, old_module in old_modules.items():
                if old_module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = old_module

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][0:4], [sys.executable, "-m", "pip", "install"])
        self.assertIn("openai<3", calls[0])
        self.assertEqual(calls[1][0:2], [sys.executable, "-c"])
        self.assertIn("ensure_tables_sync", calls[1][2])
        self.assertIn("faiss_migration", calls[1][2])


if __name__ == "__main__":
    unittest.main()
