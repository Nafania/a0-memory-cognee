import importlib.util
import logging
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
        self.old_log_level = os.environ.get("LOG_LEVEL")
        os.environ.pop("TEMPORAL_GRAPH_PROMPT_PATH", None)
        os.environ.pop("LOG_LEVEL", None)

    def tearDown(self):
        if self.old_prompt is None:
            os.environ.pop("TEMPORAL_GRAPH_PROMPT_PATH", None)
        else:
            os.environ["TEMPORAL_GRAPH_PROMPT_PATH"] = self.old_prompt
        if self.old_log_level is None:
            os.environ.pop("LOG_LEVEL", None)
        else:
            os.environ["LOG_LEVEL"] = self.old_log_level

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

    def test_get_cognee_setting_reads_plugin_config(self):
        cognee_init = _load_cognee_init_module()
        plugins = types.ModuleType("helpers.plugins")
        plugins.get_plugin_config = lambda name, agent=None: {
            "cognee_debug_enabled": True,
        }
        sys.modules["helpers.plugins"] = plugins

        self.assertTrue(cognee_init.get_cognee_setting("cognee_debug_enabled", False))

    def test_get_cognee_setting_reads_env_override(self):
        cognee_init = _load_cognee_init_module()
        cognee_init.dotenv.get_dotenv_value = (
            lambda key, default=None: "true"
            if key == "A0_SET_cognee_debug_enabled"
            else default
        )

        self.assertTrue(cognee_init.get_cognee_setting("cognee_debug_enabled", False))

    def test_cognee_logging_defaults_to_warning(self):
        cognee_init = _load_cognee_init_module()

        cognee_init._configure_cognee_logging()

        self.assertEqual(os.environ["LOG_LEVEL"], "WARNING")

    def test_cognee_debug_mode_enables_debug_logging(self):
        cognee_init = _load_cognee_init_module()
        cognee_init.get_cognee_setting = (
            lambda name, default=None: True
            if name == "cognee_debug_enabled"
            else default
        )

        cognee_init._configure_cognee_logging()

        self.assertEqual(os.environ["LOG_LEVEL"], "DEBUG")

    def test_cognee_debug_mode_keeps_raw_request_loggers_at_warning(self):
        cognee_init = _load_cognee_init_module()
        cognee_init.get_cognee_setting = (
            lambda name, default=None: True
            if name == "cognee_debug_enabled"
            else default
        )

        cognee_init._configure_cognee_logging()

        self.assertEqual(logging.getLogger("cognee").level, logging.DEBUG)
        self.assertEqual(os.environ["LITELLM_LOG"], "ERROR")
        self.assertEqual(os.environ["LITELLM_SET_VERBOSE"], "False")
        self.assertEqual(logging.getLogger("litellm").level, logging.WARNING)
        self.assertEqual(logging.getLogger("httpx").level, logging.WARNING)
        self.assertEqual(logging.getLogger("aiosqlite").level, logging.WARNING)
        self.assertEqual(logging.getLogger("sqlalchemy.engine").level, logging.WARNING)
        self.assertEqual(logging.getLogger("watchdog").level, logging.WARNING)
        self.assertEqual(logging.getLogger("DatasetQueue").level, logging.WARNING)
        self.assertEqual(logging.getLogger("ChunksRetriever").level, logging.WARNING)
        self.assertEqual(logging.getLogger("instructor").level, logging.WARNING)
        self.assertEqual(
            logging.getLogger("cognee.shared.logging_utils").level,
            logging.WARNING,
        )

    def test_cognee_log_redaction_masks_api_keys(self):
        cognee_init = _load_cognee_init_module()

        redacted = cognee_init._redact_log_text(
            "new_kwargs={'api_key': '" + "sk-" + "proj-secret_123', "
            "'llm_api_key': '" + "sk-" + "othersecret'}"
        )

        self.assertNotIn("sk-" + "proj-secret_123", redacted)
        self.assertNotIn("sk-" + "othersecret", redacted)
        self.assertIn("'api_key': '***", redacted)
        self.assertIn("'llm_api_key': '***", redacted)

    def test_cognee_log_redaction_preserves_structured_records(self):
        cognee_init = _load_cognee_init_module()
        record = logging.LogRecord(
            name="cognee.shared.logging_utils",
            level=logging.WARNING,
            pathname=__file__,
            lineno=1,
            msg={
                "event": "Empty context was provided to the completion",
                "logger": "GraphCompletionRetriever",
                "api_key": "sk-" + "proj-secret_123",
            },
            args=(),
            exc_info=None,
        )

        cognee_init._SecretRedactionFilter().filter(record)

        self.assertIsInstance(record.msg, dict)
        self.assertEqual(record.msg["event"], "Empty context was provided to the completion")
        self.assertEqual(record.msg["api_key"], "***")


if __name__ == "__main__":
    unittest.main()
