import importlib.util
import os
import struct
import sys
import tempfile
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
    spec = importlib.util.spec_from_file_location("memory_cognee_cognee_init", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_catalog(graph_dir: Path, version_code: int) -> None:
    graph_dir.mkdir(parents=True)
    with (graph_dir / "catalog.kz").open("wb") as f:
        f.write(b"KUZ\x00")
        f.write(struct.pack("<Q", version_code))


class StaleGraphDbPurgeTest(unittest.TestCase):
    def test_purges_stale_graph_dirs_without_pkl_suffix(self):
        cognee_init = _load_cognee_init_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            system_root = Path(tmp_dir) / "cognee_system"
            databases_dir = system_root / "databases"

            global_legacy_graph = databases_dir / "cognee_graph_kuzu"
            nested_legacy_graph = databases_dir / "owner-1" / "cognee_graph_ladybug"
            valid_graph = databases_dir / "owner-2" / "valid_graph"

            _write_catalog(global_legacy_graph, 999)
            _write_catalog(nested_legacy_graph, 999)
            _write_catalog(valid_graph, 37)

            old_system_root = os.environ.get("SYSTEM_ROOT_DIRECTORY")
            os.environ["SYSTEM_ROOT_DIRECTORY"] = str(system_root)
            try:
                affected = cognee_init._purge_stale_graph_dbs()
            finally:
                if old_system_root is None:
                    os.environ.pop("SYSTEM_ROOT_DIRECTORY", None)
                else:
                    os.environ["SYSTEM_ROOT_DIRECTORY"] = old_system_root

            self.assertTrue(affected)
            self.assertFalse(global_legacy_graph.exists())
            self.assertFalse(nested_legacy_graph.exists())
            self.assertTrue(valid_graph.exists())


if __name__ == "__main__":
    unittest.main()
