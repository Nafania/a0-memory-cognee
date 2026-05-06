import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _install_agent_zero_stubs() -> None:
    helpers = types.ModuleType("helpers")

    dotenv = types.ModuleType("helpers.dotenv")
    dotenv.load_dotenv = lambda *args, **kwargs: None
    dotenv.get_dotenv_value = lambda *args, **kwargs: None

    files = types.ModuleType("helpers.files")
    files.get_abs_path = lambda path: path

    settings = types.ModuleType("helpers.settings")
    settings.get_settings = lambda: {}

    print_style = types.ModuleType("helpers.print_style")

    class PrintStyle:
        @staticmethod
        def warning(*args, **kwargs):
            pass

        @staticmethod
        def error(*args, **kwargs):
            pass

        @staticmethod
        def standard(*args, **kwargs):
            pass

    print_style.PrintStyle = PrintStyle

    helpers.dotenv = dotenv
    helpers.files = files

    sys.modules.update(
        {
            "helpers": helpers,
            "helpers.dotenv": dotenv,
            "helpers.files": files,
            "helpers.settings": settings,
            "helpers.print_style": print_style,
        }
    )


def _load_cognee_init_module():
    _install_agent_zero_stubs()
    module_path = REPO_ROOT / "helpers" / "cognee_init.py"
    spec = importlib.util.spec_from_file_location("memory_cognee_cognee_init_prompt", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CogneeTemporalPromptTest(unittest.TestCase):
    def setUp(self):
        self.old_prompt = os.environ.get("TEMPORAL_GRAPH_PROMPT_PATH")
        os.environ.pop("TEMPORAL_GRAPH_PROMPT_PATH", None)

    def tearDown(self):
        if self.old_prompt is None:
            os.environ.pop("TEMPORAL_GRAPH_PROMPT_PATH", None)
        else:
            os.environ["TEMPORAL_GRAPH_PROMPT_PATH"] = self.old_prompt

    def test_configures_temporal_prompt_matching_event_list_schema(self):
        cognee_init = _load_cognee_init_module()

        cognee_init._configure_temporal_graph_prompt()

        prompt_path = Path(os.environ["TEMPORAL_GRAPH_PROMPT_PATH"])
        prompt = prompt_path.read_text()
        self.assertEqual(prompt_path.name, "cognee.generate_event_graph_prompt.txt")
        self.assertIn('"events"', prompt)
        self.assertIn("Do not return a bare JSON array", prompt)

    def test_keeps_user_configured_temporal_prompt(self):
        custom_path = "/tmp/custom-temporal-prompt.txt"
        os.environ["TEMPORAL_GRAPH_PROMPT_PATH"] = custom_path
        cognee_init = _load_cognee_init_module()

        cognee_init._configure_temporal_graph_prompt()

        self.assertEqual(os.environ["TEMPORAL_GRAPH_PROMPT_PATH"], custom_path)

    def test_keeps_optional_memify_enabled_by_default(self):
        cognee_init = _load_cognee_init_module()

        self.assertTrue(cognee_init.get_cognee_setting("cognee_memify_enabled", False))


if __name__ == "__main__":
    unittest.main()
