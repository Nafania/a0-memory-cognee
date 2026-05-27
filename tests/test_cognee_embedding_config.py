import importlib.util
import asyncio
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class CogneeEmbeddingConfigTest(unittest.TestCase):
    def setUp(self):
        self._old_env = {
            key: os.environ.get(key)
            for key in (
                "EMBEDDING_PROVIDER",
                "EMBEDDING_MODEL",
                "EMBEDDING_DIMENSIONS",
                "EMBEDDING_API_KEY",
                "EMBEDDING_API_BASE",
            )
        }

    def tearDown(self):
        for key, value in self._old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

        for name in list(sys.modules):
            if (
                name == "helpers"
                or name.startswith("helpers.")
                or name == "models"
                or name == "cognee"
                or name == "watchdog"
                or name.startswith("watchdog.")
                or name.startswith("usr.plugins.memory_cognee")
                or name == "memory_cognee_cognee_init_embedding"
            ):
                sys.modules.pop(name, None)

    def test_cognee_uses_agent_zero_embedding_model_config(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            module = self._load_module(tmp_dir)

            module.configure_cognee()

        self.assertEqual(os.environ["EMBEDDING_PROVIDER"], "fastembed")
        self.assertEqual(
            os.environ["EMBEDDING_MODEL"],
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
        self.assertEqual(os.environ["EMBEDDING_DIMENSIONS"], "384")

    def test_first_seen_legacy_embedding_config_does_not_rebuild(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            module = self._load_module(tmp_dir)
            reset_calls = []

            async def reset_all():
                reset_calls.append("reset")

            module.reset_cognify_status_for_all_datasets = reset_all

            asyncio.run(
                module._ensure_embedding_config_state(
                    {
                        "provider": "fastembed",
                        "model": "sentence-transformers/all-MiniLM-L6-v2",
                        "dimensions": "384",
                        "api_base": "",
                    }
                )
            )

        self.assertEqual(reset_calls, [])

    def test_first_seen_nonlegacy_embedding_config_rebuilds_once(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            module = self._load_module(tmp_dir)
            reset_calls = []
            current = {
                "provider": "fastembed",
                "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "dimensions": "384",
                "api_base": "",
            }

            async def reset_all():
                reset_calls.append("reset")

            module.reset_cognify_status_for_all_datasets = reset_all

            asyncio.run(module._ensure_embedding_config_state(current))
            asyncio.run(module._ensure_embedding_config_state(current))

            self.assertIsNone(module._load_embedding_config_state())
            self.assertEqual(module._load_pending_embedding_config_state(), current)
            module._mark_embedding_config_rebuild_completed()
            self.assertEqual(module._load_embedding_config_state(), current)
            self.assertEqual(reset_calls, ["reset"])

    def test_changed_embedding_config_rebuilds_and_persists_current_state(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            module = self._load_module(tmp_dir)
            old = {
                "provider": "fastembed",
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "dimensions": "384",
                "api_base": "",
            }
            current = {
                "provider": "fastembed",
                "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "dimensions": "384",
                "api_base": "",
            }
            module._save_embedding_config_state(old)
            reset_calls = []

            async def reset_all():
                reset_calls.append("reset")

            module.reset_cognify_status_for_all_datasets = reset_all

            asyncio.run(module._ensure_embedding_config_state(current))

            self.assertEqual(module._load_pending_embedding_config_state(), current)
            module._mark_embedding_config_rebuild_completed()
            self.assertEqual(module._load_embedding_config_state(), current)

        self.assertEqual(reset_calls, ["reset"])

    def test_pending_embedding_rebuild_resumes_without_resetting_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            module = self._load_module(tmp_dir)
            old = {
                "provider": "fastembed",
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "dimensions": "384",
                "api_base": "",
            }
            current = {
                "provider": "fastembed",
                "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "dimensions": "384",
                "api_base": "",
            }
            module._save_embedding_config_state(old)
            module._save_pending_embedding_config_state(current)
            module._embedding_rebuild_scheduled = False
            reset_calls = []
            resume_calls = []

            async def reset_all():
                reset_calls.append("reset")

            async def resume_all(reason):
                resume_calls.append(reason)

            module.reset_cognify_status_for_all_datasets = reset_all
            module.mark_all_datasets_dirty_for_rebuild = resume_all

            asyncio.run(module._ensure_embedding_config_state(current))

        self.assertEqual(reset_calls, [])
        self.assertEqual(resume_calls, ["pending embedding config rebuild"])

    def test_unchanged_embedding_config_does_not_rebuild(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            module = self._load_module(tmp_dir)
            current = {
                "provider": "fastembed",
                "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "dimensions": "384",
                "api_base": "",
            }
            module._save_embedding_config_state(current)
            reset_calls = []

            async def reset_all():
                reset_calls.append("reset")

            module.reset_cognify_status_for_all_datasets = reset_all

            asyncio.run(module._ensure_embedding_config_state(current))

        self.assertEqual(reset_calls, [])

    def test_embedding_rebuild_needed_when_pending_matches_current(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            module = self._load_module(tmp_dir)
            current = {
                "provider": "fastembed",
                "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "dimensions": "384",
                "api_base": "",
            }
            module._save_pending_embedding_config_state(current)

            self.assertTrue(module._embedding_config_rebuild_needed(current))

    def test_watchdog_patch_excludes_cognee_dirs_from_recursive_inotify(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            module = self._load_module(tmp_dir)
            root = Path(tmp_dir) / "usr"
            cognee_dir = root / "cognee"
            cognee_nested = cognee_dir / "data_storage" / "table"
            state_dir = root / "cognee_state"
            legacy_memory_dir = root / "memory"
            project_memory_dir = root / "projects" / "personal" / ".a0proj" / "memory"
            git_dir = root / "plugins" / "memory_cognee" / ".git" / "objects"
            node_modules_dir = root / "plugins" / "memory_cognee" / "node_modules" / "pkg"
            mcp_dir = root / "mcp" / "fli-pkg" / "pkg"
            lib_dir = root / "lib" / "python3.12" / "site-packages"
            npm_dir = root / ".npm" / "_npx" / "pkg"
            keep_dir = root / "projects"
            project_keep_dir = root / "projects" / "personal" / "docs"
            for path in (
                cognee_nested,
                state_dir,
                legacy_memory_dir / "default",
                project_memory_dir,
                git_dir,
                node_modules_dir,
                mcp_dir,
                lib_dir,
                npm_dir,
                project_keep_dir,
            ):
                path.mkdir(parents=True, exist_ok=True)
            fake_inotify = types.ModuleType("watchdog.observers.inotify_c")

            class Inotify:
                def __init__(self):
                    self.watched = []

                def _add_watch(self, path, mask):
                    self.watched.append(os.path.abspath(os.fsdecode(path)))

            fake_inotify.Inotify = Inotify
            sys.modules.update(
                {
                    "watchdog": types.ModuleType("watchdog"),
                    "watchdog.observers": types.ModuleType("watchdog.observers"),
                    "watchdog.observers.inotify_c": fake_inotify,
                }
            )

            module._patch_watchdog_inotify_excludes(
                [
                    str(cognee_dir),
                    str(state_dir),
                    str(legacy_memory_dir),
                    str(root / "mcp"),
                    str(root / "lib"),
                    str(root / ".npm"),
                ]
            )
            watcher = fake_inotify.Inotify()
            watcher._add_dir_watch(os.fsencode(root), 1, recursive=True)

            self.assertIn(str(root), watcher.watched)
            self.assertIn(str(keep_dir), watcher.watched)
            self.assertIn(str(project_keep_dir), watcher.watched)
            self.assertNotIn(str(cognee_dir), watcher.watched)
            self.assertNotIn(str(cognee_nested), watcher.watched)
            self.assertNotIn(str(state_dir), watcher.watched)
            self.assertNotIn(str(legacy_memory_dir), watcher.watched)
            self.assertNotIn(str(project_memory_dir), watcher.watched)
            self.assertNotIn(str(git_dir), watcher.watched)
            self.assertNotIn(str(node_modules_dir), watcher.watched)
            self.assertNotIn(str(mcp_dir), watcher.watched)
            self.assertNotIn(str(lib_dir), watcher.watched)
            self.assertNotIn(str(npm_dir), watcher.watched)

            direct = fake_inotify.Inotify()
            direct._add_dir_watch(os.fsencode(cognee_dir), 1, recursive=True)
            self.assertEqual(direct.watched, [])

    def _load_module(self, tmp_dir: str):
        helpers = types.ModuleType("helpers")
        dotenv = types.ModuleType("helpers.dotenv")
        files = types.ModuleType("helpers.files")
        settings = types.ModuleType("helpers.settings")
        plugins = types.ModuleType("helpers.plugins")
        print_style = types.ModuleType("helpers.print_style")

        dotenv.load_dotenv = lambda *args, **kwargs: None
        dotenv.get_dotenv_value = lambda key, default=None: default
        files.get_abs_path = lambda *parts: os.path.join(tmp_dir, *parts)
        settings.get_settings = lambda: {"api_keys": {}}

        def get_plugin_config(name, *args, **kwargs):
            if name == "memory_cognee":
                return {}
            if name == "_model_config":
                return {
                    "utility_model": {
                        "provider": "openai",
                        "name": "gpt-4.1-mini",
                    },
                    "embedding_model": {
                        "provider": "huggingface",
                        "name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                    },
                }
            return {}

        plugins.get_plugin_config = get_plugin_config

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

        providers = types.ModuleType("helpers.providers")
        providers.get_provider_config = lambda *args, **kwargs: {}

        models = types.ModuleType("models")
        models.get_api_key = lambda *args, **kwargs: ""

        cognee = types.ModuleType("cognee")

        class Config:
            def set_llm_config(self, *args, **kwargs):
                pass

            def set_llm_endpoint(self, *args, **kwargs):
                pass

            def set_chunk_size(self, *args, **kwargs):
                pass

            def set_chunk_overlap(self, *args, **kwargs):
                pass

            def data_root_directory(self, *args, **kwargs):
                pass

            def system_root_directory(self, *args, **kwargs):
                pass

        cognee.config = Config()
        cognee.SearchType = types.SimpleNamespace(GRAPH_COMPLETION="GRAPH_COMPLETION")

        sys.modules.update(
            {
                "helpers": helpers,
                "helpers.dotenv": dotenv,
                "helpers.files": files,
                "helpers.settings": settings,
                "helpers.plugins": plugins,
                "helpers.print_style": print_style,
                "helpers.providers": providers,
                "models": models,
                "cognee": cognee,
            }
        )

        module_path = REPO_ROOT / "helpers" / "cognee_init.py"
        spec = importlib.util.spec_from_file_location(
            "memory_cognee_cognee_init_embedding",
            module_path,
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module


if __name__ == "__main__":
    unittest.main()
