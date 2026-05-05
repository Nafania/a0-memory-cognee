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


def _write_graph_file(graph_file: Path, version_code: int, magic: bytes = b"KUZ\x00") -> None:
    graph_file.parent.mkdir(parents=True, exist_ok=True)
    with graph_file.open("wb") as f:
        f.write(magic)
        f.write(struct.pack("<Q", version_code))


def _run_purge_with_system_root(cognee_init, system_root: Path) -> set[str]:
    old_system_root = os.environ.get("SYSTEM_ROOT_DIRECTORY")
    os.environ["SYSTEM_ROOT_DIRECTORY"] = str(system_root)
    try:
        return cognee_init._purge_stale_graph_dbs()
    finally:
        if old_system_root is None:
            os.environ.pop("SYSTEM_ROOT_DIRECTORY", None)
        else:
            os.environ["SYSTEM_ROOT_DIRECTORY"] = old_system_root


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

            affected = _run_purge_with_system_root(cognee_init, system_root)

            self.assertTrue(affected)
            self.assertFalse(global_legacy_graph.exists())
            self.assertFalse(nested_legacy_graph.exists())
            self.assertTrue(valid_graph.exists())

    def test_purges_stale_file_based_graph_dbs(self):
        cognee_init = _load_cognee_init_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            system_root = Path(tmp_dir) / "cognee_system"
            databases_dir = system_root / "databases"

            stale_graph_file = databases_dir / "cognee_graph_kuzu"
            valid_graph_file = databases_dir / "valid_graph_kuzu"
            _write_graph_file(stale_graph_file, 999)
            _write_graph_file(valid_graph_file, 37)
            (Path(str(stale_graph_file) + ".wal")).write_text("wal")
            (Path(str(stale_graph_file) + ".lock")).write_text("lock")

            affected = _run_purge_with_system_root(cognee_init, system_root)

            self.assertTrue(affected)
            self.assertFalse(stale_graph_file.exists())
            self.assertFalse(Path(str(stale_graph_file) + ".wal").exists())
            self.assertFalse(Path(str(stale_graph_file) + ".lock").exists())
            self.assertTrue(valid_graph_file.exists())

    def test_keeps_unknown_version_graph_if_current_ladybug_can_open_it(self):
        cognee_init = _load_cognee_init_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            system_root = Path(tmp_dir) / "cognee_system"
            databases_dir = system_root / "databases"
            current_graph = databases_dir / "cognee_graph_ladybug"
            _write_catalog(current_graph, 999)

            original = cognee_init._is_graph_readable_by_current_ladybug
            cognee_init._is_graph_readable_by_current_ladybug = lambda path: True
            try:
                affected = _run_purge_with_system_root(cognee_init, system_root)
            finally:
                cognee_init._is_graph_readable_by_current_ladybug = original

            self.assertFalse(affected)
            self.assertTrue(current_graph.exists())

    def test_purges_unreadable_current_lbug_file_graph_dbs(self):
        cognee_init = _load_cognee_init_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            system_root = Path(tmp_dir) / "cognee_system"
            graph_file = system_root / "databases" / "cognee_graph_ladybug"
            _write_graph_file(graph_file, 40, magic=b"LBUG")

            affected = _run_purge_with_system_root(cognee_init, system_root)

            self.assertTrue(affected)
            self.assertFalse(graph_file.exists())

    def test_keeps_current_lbug_file_graph_if_current_ladybug_can_open_it(self):
        cognee_init = _load_cognee_init_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            system_root = Path(tmp_dir) / "cognee_system"
            graph_file = system_root / "databases" / "cognee_graph_ladybug"
            _write_graph_file(graph_file, 40, magic=b"LBUG")

            original = cognee_init._is_graph_readable_by_current_ladybug
            cognee_init._is_graph_readable_by_current_ladybug = lambda path: True
            try:
                affected = _run_purge_with_system_root(cognee_init, system_root)
            finally:
                cognee_init._is_graph_readable_by_current_ladybug = original

            self.assertFalse(affected)
            self.assertTrue(graph_file.exists())

    def test_keeps_real_current_ladybug_graph_file(self):
        try:
            import ladybug
        except Exception as exc:
            self.skipTest(f"ladybug not installed: {exc}")

        cognee_init = _load_cognee_init_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            system_root = Path(tmp_dir) / "cognee_system"
            graph_file = system_root / "databases" / "cognee_graph_ladybug"
            graph_file.parent.mkdir(parents=True, exist_ok=True)
            db = ladybug.Database(str(graph_file))
            db.init_database()
            close = getattr(db, "close", None)
            if callable(close):
                close()

            affected = _run_purge_with_system_root(cognee_init, system_root)

            self.assertFalse(affected)
            self.assertTrue(graph_file.exists())


if __name__ == "__main__":
    unittest.main()
