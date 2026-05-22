import importlib.util
import asyncio
import os
import sqlite3
import struct
import sys
import tempfile
import types
import unittest
from contextlib import closing
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


def _write_graph_file(graph_file: Path, version_code: int, magic: bytes = b"KUZ\x00") -> None:
    graph_file.parent.mkdir(parents=True, exist_ok=True)
    with graph_file.open("wb") as f:
        f.write(magic)
        f.write(struct.pack("<Q", version_code))


class CogneeInitStartupTest(unittest.TestCase):
    def tearDown(self):
        for name in list(sys.modules):
            if name.startswith("cognee.infrastructure.databases.vector.lancedb"):
                sys.modules.pop(name, None)

    def test_rewrites_legacy_data_storage_locations_to_current_data_root(self):
        cognee_init = _load_cognee_init_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            system_root = tmp / "cognee_system"
            data_root = tmp / "data_storage"
            db_dir = system_root / "databases"
            db_dir.mkdir(parents=True)

            data_id = "e0af5892-5ab8-4de4-9418-2f223a330b12"
            data_dir = data_root / data_id
            data_dir.mkdir(parents=True)
            (data_dir / "text_abc.txt").write_text("memory")

            db_path = db_dir / "cognee_db"
            with closing(sqlite3.connect(db_path)) as conn:
                conn.execute(
                    "CREATE TABLE data ("
                    "id TEXT PRIMARY KEY, "
                    "raw_data_location TEXT, "
                    "original_data_location TEXT"
                    ")"
                )
                conn.execute(
                    "INSERT INTO data VALUES (?, ?, ?)",
                    (
                        data_id,
                        f"file:///old/agent-zero/usr/cognee/data_storage/{data_id}",
                        f"file:///old/agent-zero/usr/cognee/data_storage/{data_id}/text_abc.txt",
                    ),
                )
                conn.commit()

            old_system_root = os.environ.get("SYSTEM_ROOT_DIRECTORY")
            old_data_root = os.environ.get("DATA_ROOT_DIRECTORY")
            os.environ["SYSTEM_ROOT_DIRECTORY"] = str(system_root)
            os.environ["DATA_ROOT_DIRECTORY"] = str(data_root)
            try:
                self.assertEqual(cognee_init._rewrite_legacy_data_storage_locations(), 2)
            finally:
                if old_system_root is None:
                    os.environ.pop("SYSTEM_ROOT_DIRECTORY", None)
                else:
                    os.environ["SYSTEM_ROOT_DIRECTORY"] = old_system_root
                if old_data_root is None:
                    os.environ.pop("DATA_ROOT_DIRECTORY", None)
                else:
                    os.environ["DATA_ROOT_DIRECTORY"] = old_data_root

            with closing(sqlite3.connect(db_path)) as conn:
                raw, original = conn.execute(
                    "SELECT raw_data_location, original_data_location FROM data WHERE id = ?",
                    (data_id,),
                ).fetchone()

            self.assertEqual(raw, f"file://{data_dir}")
            self.assertEqual(original, f"file://{data_dir / 'text_abc.txt'}")

    def test_does_not_rewrite_legacy_data_storage_location_when_file_is_missing(self):
        cognee_init = _load_cognee_init_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            data_root = Path(tmp_dir) / "data_storage"
            missing_uri = "file:///old/agent-zero/usr/cognee/data_storage/missing-id"

            self.assertIsNone(
                cognee_init._rewrite_data_storage_uri(missing_uri, str(data_root))
            )

    def test_quarantines_data_rows_with_missing_source_files_without_deleting_data(self):
        cognee_init = _load_cognee_init_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            system_root = tmp / "cognee_system"
            db_dir = system_root / "databases"
            db_dir.mkdir(parents=True)

            data_id = "dc45cc7ecd36550991797812df968b77"
            dataset_id = "2ca04e1d-b576-5b33-9d10-626e04003639"
            missing_file = tmp / "data_storage" / "owner" / "text_missing.txt"
            missing_file.parent.mkdir(parents=True)

            db_path = db_dir / "cognee_db"
            with closing(sqlite3.connect(db_path)) as conn:
                conn.execute(
                    "CREATE TABLE data ("
                    "id TEXT PRIMARY KEY, "
                    "name TEXT, "
                    "extension TEXT, "
                    "raw_data_location TEXT"
                    ")"
                )
                conn.execute(
                    "CREATE TABLE dataset_data (dataset_id TEXT, data_id TEXT)"
                )
                conn.execute(
                    "INSERT INTO data VALUES (?, ?, ?, ?)",
                    (
                        data_id,
                        "text_missing",
                        "txt",
                        f"file://{missing_file}",
                    ),
                )
                conn.execute(
                    "INSERT INTO dataset_data VALUES (?, ?)",
                    (dataset_id, data_id),
                )
                conn.commit()

            old_system_root = os.environ.get("SYSTEM_ROOT_DIRECTORY")
            os.environ["SYSTEM_ROOT_DIRECTORY"] = str(system_root)
            try:
                self.assertEqual(cognee_init._quarantine_missing_data_files(), 1)
            finally:
                if old_system_root is None:
                    os.environ.pop("SYSTEM_ROOT_DIRECTORY", None)
                else:
                    os.environ["SYSTEM_ROOT_DIRECTORY"] = old_system_root

            with closing(sqlite3.connect(db_path)) as conn:
                data_count = conn.execute(
                    "SELECT COUNT(*) FROM data WHERE id = ?",
                    (data_id,),
                ).fetchone()[0]
                association_count = conn.execute(
                    "SELECT COUNT(*) FROM dataset_data WHERE data_id = ?",
                    (data_id,),
                ).fetchone()[0]
                quarantine_count = conn.execute(
                    "SELECT COUNT(*) FROM a0_cognee_quarantined_data WHERE data_id = ?",
                    (data_id,),
                ).fetchone()[0]

            self.assertEqual(data_count, 1)
            self.assertEqual(association_count, 0)
            self.assertEqual(quarantine_count, 1)

    def test_keeps_data_row_when_expected_source_file_exists(self):
        cognee_init = _load_cognee_init_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            source_dir = Path(tmp_dir) / "data_storage" / "owner"
            source_dir.mkdir(parents=True)
            (source_dir / "text_exists.txt").write_text("memory")

            self.assertIsNone(
                cognee_init._missing_data_source_path(
                    f"file://{source_dir}",
                    "text_exists",
                    "txt",
                )
            )

    def test_schema_sync_disposes_sqlalchemy_engine(self):
        cognee_init = _load_cognee_init_module()

        fake_engine = types.SimpleNamespace(
            dialect=types.SimpleNamespace(name="sqlite"),
            disposed=False,
        )
        fake_engine.dispose = lambda: setattr(fake_engine, "disposed", True)

        sqlalchemy = types.ModuleType("sqlalchemy")
        sqlalchemy.create_engine = lambda *args, **kwargs: fake_engine
        sqlalchemy.inspect = lambda engine: types.SimpleNamespace()
        sqlalchemy.text = lambda ddl: ddl

        model_base = types.ModuleType("cognee.infrastructure.databases.relational.ModelBase")
        model_base.Base = types.SimpleNamespace(
            metadata=types.SimpleNamespace(tables={})
        )

        old_modules = {
            name: sys.modules.get(name)
            for name in (
                "sqlalchemy",
                "cognee.infrastructure.databases.relational.ModelBase",
            )
        }
        sys.modules["sqlalchemy"] = sqlalchemy
        sys.modules["cognee.infrastructure.databases.relational.ModelBase"] = model_base

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            system_root = tmp / "cognee_system"
            db_dir = system_root / "databases"
            db_dir.mkdir(parents=True)
            (db_dir / "cognee_db").touch()
            os.environ["SYSTEM_ROOT_DIRECTORY"] = str(system_root)
            os.environ["DB_NAME"] = "cognee_db"
            try:
                cognee_init._sync_missing_columns()
            finally:
                os.environ.pop("SYSTEM_ROOT_DIRECTORY", None)
                os.environ.pop("DB_NAME", None)
                for name, old_module in old_modules.items():
                    if old_module is None:
                        sys.modules.pop(name, None)
                    else:
                        sys.modules[name] = old_module

        self.assertTrue(fake_engine.disposed)

    def test_detects_dataset_with_data_but_empty_graph(self):
        cognee_init = _load_cognee_init_module()

        dataset_id = "00afc710-2c0c-5d61-957e-c452672842ae"
        fake_cognee = types.ModuleType("cognee")

        class Datasets:
            async def list_datasets(self):
                return [types.SimpleNamespace(id=dataset_id, name="default")]

            async def list_data(self, ds_id):
                return [types.SimpleNamespace(id="data-1")]

        fake_cognee.datasets = Datasets()
        fake_graph_module = types.ModuleType(
            "usr.plugins.memory_cognee.helpers.cognee_graph"
        )

        async def read_dataset_graphs(cognee, dataset_names=None, **kwargs):
            return [
                types.SimpleNamespace(
                    dataset_id=dataset_id,
                    dataset_name="default",
                    data_count=3034,
                    nodes=[],
                    edges=[],
                    graph_empty=True,
                    error=None,
                )
            ]

        fake_graph_module.read_dataset_graphs = read_dataset_graphs
        old_cognee = sys.modules.get("cognee")
        old_graph = sys.modules.get("usr.plugins.memory_cognee.helpers.cognee_graph")
        sys.modules["cognee"] = fake_cognee
        sys.modules["usr.plugins.memory_cognee.helpers.cognee_graph"] = fake_graph_module
        try:
            unready = asyncio.run(cognee_init._detect_datasets_with_unready_graphs())
        finally:
            if old_cognee is None:
                sys.modules.pop("cognee", None)
            else:
                sys.modules["cognee"] = old_cognee
            if old_graph is None:
                sys.modules.pop("usr.plugins.memory_cognee.helpers.cognee_graph", None)
            else:
                sys.modules[
                    "usr.plugins.memory_cognee.helpers.cognee_graph"
                ] = old_graph

        self.assertEqual(unready, {dataset_id})

    def test_reset_cognify_status_targets_only_affected_dataset_ids(self):
        cognee_init = _load_cognee_init_module()

        affected_id = "00afc710-2c0c-5d61-957e-c452672842ae"
        clean_id = "eba921c6-3310-528d-8dca-0ec58cfe603b"
        fake_cognee = types.ModuleType("cognee")

        class Datasets:
            async def list_datasets(self):
                return [
                    types.SimpleNamespace(id=affected_id, name="default"),
                    types.SimpleNamespace(id=clean_id, name="projects_alpha"),
                ]

        fake_cognee.datasets = Datasets()
        deleted_ids = []

        async def delete_pipeline_runs(dataset_ids):
            deleted_ids.extend(str(dataset_id) for dataset_id in dataset_ids)
            return len(dataset_ids)

        cognee_init._delete_pipeline_runs_for_dataset_ids = delete_pipeline_runs
        old_cognee = sys.modules.get("cognee")
        sys.modules["cognee"] = fake_cognee
        try:
            asyncio.run(cognee_init._reset_cognify_status_for_datasets({affected_id}))
        finally:
            if old_cognee is None:
                sys.modules.pop("cognee", None)
            else:
                sys.modules["cognee"] = old_cognee

        self.assertEqual(deleted_ids, [affected_id])

    def test_patches_lancedb_migration_defaults_for_source_fields(self):
        cognee_init = _load_cognee_init_module()

        module_name = "cognee.infrastructure.databases.vector.lancedb.LanceDBAdapter"
        fake_module = types.ModuleType(module_name)

        class PayloadSchema:
            model_fields = {
                "id": object(),
                "source_pipeline": object(),
                "source_task": object(),
                "source_node_set": object(),
                "source_user": object(),
                "source_content_hash": object(),
                "metadata": object(),
                "text": object(),
            }

        class LanceDBAdapter:
            def _get_payload_defaults(self, payload_schema):
                return {"id": "", "text": ""}

            def get_data_point_schema(self, payload_schema):
                return PayloadSchema

        fake_module.LanceDBAdapter = LanceDBAdapter
        sys.modules[module_name] = fake_module

        cognee_init._patch_lancedb_migration_defaults()

        defaults = LanceDBAdapter()._get_payload_defaults(object())
        self.assertEqual(defaults["source_pipeline"], None)
        self.assertEqual(defaults["source_task"], None)
        self.assertEqual(defaults["source_node_set"], None)
        self.assertEqual(defaults["source_user"], None)
        self.assertEqual(defaults["source_content_hash"], None)
        self.assertEqual(defaults["metadata"], {})
        self.assertEqual(defaults["text"], "")

    def test_startup_does_not_purge_ladybug_graph_files(self):
        cognee_init = _load_cognee_init_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            system_root = Path(tmp_dir) / "cognee_system"
            graph_file = system_root / "databases" / "cognee_graph_ladybug"
            _write_graph_file(graph_file, 999, magic=b"LBUG")

            run_migrations_module = types.ModuleType("cognee.run_migrations")

            async def run_startup_migrations():
                return None

            run_migrations_module.run_startup_migrations = run_startup_migrations
            old_run_migrations = sys.modules.get("cognee.run_migrations")
            sys.modules["cognee.run_migrations"] = run_migrations_module

            old_system_root = os.environ.get("SYSTEM_ROOT_DIRECTORY")
            os.environ["SYSTEM_ROOT_DIRECTORY"] = str(system_root)
            original_lancedb = cognee_init._patch_lancedb_migration_defaults
            original_sync = cognee_init._sync_missing_columns
            original_rewrite = cognee_init._rewrite_legacy_data_storage_locations
            original_quarantine = cognee_init._quarantine_missing_data_files
            original_unready = cognee_init._detect_datasets_with_unready_graphs

            async def no_datasets():
                return set()

            cognee_init._patch_lancedb_migration_defaults = lambda: None
            cognee_init._sync_missing_columns = lambda: None
            cognee_init._rewrite_legacy_data_storage_locations = lambda: None
            cognee_init._quarantine_missing_data_files = lambda: None
            cognee_init._detect_datasets_with_unready_graphs = no_datasets
            try:
                asyncio.run(cognee_init._create_db_tables())
            finally:
                cognee_init._patch_lancedb_migration_defaults = original_lancedb
                cognee_init._sync_missing_columns = original_sync
                cognee_init._rewrite_legacy_data_storage_locations = original_rewrite
                cognee_init._quarantine_missing_data_files = original_quarantine
                cognee_init._detect_datasets_with_unready_graphs = original_unready
                if old_system_root is None:
                    os.environ.pop("SYSTEM_ROOT_DIRECTORY", None)
                else:
                    os.environ["SYSTEM_ROOT_DIRECTORY"] = old_system_root
                if old_run_migrations is None:
                    sys.modules.pop("cognee.run_migrations", None)
                else:
                    sys.modules["cognee.run_migrations"] = old_run_migrations

            self.assertTrue(graph_file.exists())

if __name__ == "__main__":
    unittest.main()
