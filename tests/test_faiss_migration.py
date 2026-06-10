import asyncio
import importlib.util
import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_faiss_migration_module():
    helpers = types.ModuleType("helpers")
    files = types.ModuleType("helpers.files")
    files.get_abs_path = lambda *parts: "/tmp/" + "/".join(parts)

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

    cognee_init = types.ModuleType("usr.plugins.memory_cognee.helpers.cognee_init")
    cognee_init.configure_cognee = lambda: None
    cognee_init.get_cognee_setting = lambda key, default=None: default

    cognee_ops = types.ModuleType("usr.plugins.memory_cognee.helpers.cognee_ops")

    async def run_cognee_operation(label, operation, *args, **kwargs):
        result = operation(*args, **kwargs)
        if hasattr(result, "__await__"):
            return await result
        return result

    cognee_ops.run_cognee_operation = run_cognee_operation

    sys.modules.update(
        {
            "helpers": helpers,
            "helpers.files": files,
            "usr.plugins.memory_cognee.helpers.cognee_init": cognee_init,
            "usr.plugins.memory_cognee.helpers.cognee_ops": cognee_ops,
        }
    )

    module_path = REPO_ROOT / "helpers" / "faiss_migration.py"
    spec = importlib.util.spec_from_file_location(
        "usr.plugins.memory_cognee.helpers.faiss_migration",
        module_path,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FaissMigrationTest(unittest.TestCase):
    def tearDown(self):
        for name in list(sys.modules):
            if (
                name == "helpers"
                or name.startswith("helpers.")
                or name.startswith("usr.plugins.memory_cognee")
            ):
                sys.modules.pop(name, None)

    def test_migrate_index_reports_error_status_when_faiss_load_fails(self):
        module = _load_faiss_migration_module()
        sys.modules["cognee"] = types.ModuleType("cognee")
        state = {"version": 1, "indices": {}, "completed": False}
        index = {
            "type": "global",
            "memory_subdir": "default",
            "db_dir": "/tmp/usr/memory/default",
            "index_path": "/tmp/usr/memory/default/index.faiss",
        }
        saved = []
        module.load_faiss_db = lambda db_dir: None
        module.save_state = lambda base_dir, new_state: saved.append(dict(new_state))

        result = asyncio.run(module.migrate_index(index, state, "/tmp"))

        self.assertEqual(result["status"], "error")
        self.assertEqual(state["indices"]["global:default"]["status"], "error")
        self.assertTrue(saved)

    def test_run_cognify_uses_non_temporal_pipeline_by_default(self):
        module = _load_faiss_migration_module()
        calls = []

        class FakeDatasets:
            async def list_datasets(self):
                return [types.SimpleNamespace(name="default")]

        fake_cognee = types.ModuleType("cognee")
        fake_cognee.datasets = FakeDatasets()

        async def cognify(**kwargs):
            calls.append(kwargs)

        fake_cognee.cognify = cognify
        sys.modules["cognee"] = fake_cognee

        asyncio.run(module.run_cognify([{"memory_subdir": "default"}]))

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["datasets"], ["default"])
        self.assertFalse(calls[0]["temporal_cognify"])

    def test_completed_cleanup_reruns_migration_in_same_startup(self):
        module = _load_faiss_migration_module()
        state = {
            "version": 1,
            "completed": True,
            "data_dir": "/tmp/usr/cognee/data_storage",
            "indices": {"global:default": {"status": "complete", "total": 1}},
        }
        calls = []
        index = {
            "type": "global",
            "memory_subdir": "default",
            "db_dir": "/tmp/usr/memory/default",
            "index_path": "/tmp/usr/memory/default/index.faiss",
        }

        module.load_state = lambda base_dir: state
        module.save_state = lambda base_dir, new_state: state.update(new_state)
        module._current_data_dir = lambda: "/tmp/usr/cognee/data_storage"
        module.find_faiss_indices = lambda base_dir: [index]

        async def cleanup(base_dir, *, delete=True):
            if delete:
                state["cleanup_v2_done"] = True
                state["cleanup_v2_verified_done"] = True
                state.pop("cleanup_v2_pending", None)
                state.pop("cleanup_v2_reimported", None)
            else:
                state["completed"] = False
                state["cleanup_v2_pending"] = True
            return True

        async def migrate_index(index_info, migration_state, base_dir, dry_run=False):
            calls.append(index_info["memory_subdir"])
            migration_state.setdefault("indices", {})["global:default"] = {
                "status": "complete"
            }
            return {"subdir": "default", "total": 1, "migrated": 1, "skipped": False}

        module.cleanup_backup_datasets = cleanup
        module.migrate_index = migrate_index
        module.backup_completed_indices = lambda indices, migration_state: None

        completed = asyncio.run(module.run_migration())

        self.assertTrue(completed)
        self.assertEqual(calls, ["default"])

    def test_completed_cleanup_does_not_delete_without_reimport_source(self):
        module = _load_faiss_migration_module()
        state = {
            "version": 1,
            "completed": True,
            "data_dir": "/tmp/usr/cognee/data_storage",
            "indices": {"global:default": {"status": "complete", "total": 1}},
        }
        cleanup_calls = []

        module.load_state = lambda base_dir: state
        module.save_state = lambda base_dir, new_state: state.update(new_state)
        module._current_data_dir = lambda: "/tmp/usr/cognee/data_storage"
        module.find_faiss_indices = lambda base_dir: []

        async def cleanup(base_dir, *, delete=True):
            cleanup_calls.append(delete)
            self.assertFalse(delete)
            state["completed"] = False
            state["cleanup_v2_pending"] = True
            return True

        module.cleanup_backup_datasets = cleanup

        completed = asyncio.run(module.run_migration())

        self.assertFalse(completed)
        self.assertEqual(cleanup_calls, [False])
        self.assertTrue(state["completed"])
        self.assertTrue(state["cleanup_v2_pending"])

    def test_cleanup_prepare_does_not_forget_before_reimport(self):
        module = _load_faiss_migration_module()
        state = {
            "version": 1,
            "completed": True,
            "indices": {"global:default": {"status": "complete", "total": 1}},
        }
        forget_calls = []

        class FakeDatasets:
            async def list_datasets(self):
                return [types.SimpleNamespace(id="old-id", name="default_main")]

        fake_cognee = types.ModuleType("cognee")
        fake_cognee.datasets = FakeDatasets()

        async def forget(*, dataset):
            forget_calls.append(dataset)

        fake_cognee.forget = forget
        sys.modules["cognee"] = fake_cognee
        module.load_state = lambda base_dir: state
        module.save_state = lambda base_dir, new_state: state.update(new_state)

        changed = asyncio.run(module.cleanup_backup_datasets("/base", delete=False))

        self.assertTrue(changed)
        self.assertEqual(forget_calls, [])
        self.assertFalse(state["completed"])
        self.assertTrue(state["cleanup_v2_pending"])
        self.assertEqual(state["indices"]["global:default"]["status"], "pending")

    def test_legacy_cleanup_done_is_revalidated_when_old_datasets_remain(self):
        module = _load_faiss_migration_module()
        state = {
            "version": 1,
            "completed": True,
            "cleanup_v2_done": True,
            "indices": {"global:default": {"status": "complete", "total": 1}},
        }

        class FakeDatasets:
            async def list_datasets(self):
                return [types.SimpleNamespace(id="old-id", name="default_main")]

        fake_cognee = types.ModuleType("cognee")
        fake_cognee.datasets = FakeDatasets()
        sys.modules["cognee"] = fake_cognee
        module.load_state = lambda base_dir: state
        module.save_state = lambda base_dir, new_state: state.update(new_state)

        changed = asyncio.run(module.cleanup_backup_datasets("/base", delete=False))

        self.assertTrue(changed)
        self.assertFalse(state["completed"])
        self.assertTrue(state["cleanup_v2_pending"])
        self.assertNotIn("cleanup_v2_done", state)

    def test_verified_cleanup_done_skips_dataset_listing(self):
        module = _load_faiss_migration_module()
        state = {
            "version": 1,
            "completed": True,
            "cleanup_v2_done": True,
            "cleanup_v2_verified_done": True,
            "indices": {"global:default": {"status": "complete", "total": 1}},
        }

        module.load_state = lambda base_dir: state

        def fail_configure():
            raise AssertionError("verified cleanup should not configure Cognee")

        module.configure_cognee = fail_configure

        changed = asyncio.run(module.cleanup_backup_datasets("/base", delete=False))

        self.assertFalse(changed)

    def test_cleanup_delete_failure_is_retried(self):
        module = _load_faiss_migration_module()
        state = {
            "version": 1,
            "completed": True,
            "cleanup_v2_pending": True,
            "indices": {"global:default": {"status": "complete", "total": 1}},
        }

        class FakeDatasets:
            async def list_datasets(self):
                return [types.SimpleNamespace(id="old-id", name="default_main")]

        fake_cognee = types.ModuleType("cognee")
        fake_cognee.datasets = FakeDatasets()

        async def forget(*, dataset):
            raise RuntimeError("delete failed")

        fake_cognee.forget = forget
        sys.modules["cognee"] = fake_cognee
        module.load_state = lambda base_dir: state
        module.save_state = lambda base_dir, new_state: state.update(new_state)

        changed = asyncio.run(module.cleanup_backup_datasets("/base"))

        self.assertFalse(changed)
        self.assertTrue(state["cleanup_v2_pending"])
        self.assertNotIn("cleanup_v2_done", state)

    def test_pending_cleanup_after_reimport_retries_delete_without_reimport(self):
        module = _load_faiss_migration_module()
        state = {
            "version": 1,
            "completed": True,
            "data_dir": "/tmp/usr/cognee/data_storage",
            "cleanup_v2_pending": True,
            "cleanup_v2_reimported": True,
            "indices": {"global:default": {"status": "complete", "total": 1}},
        }
        cleanup_calls = []
        migrate_calls = []

        module.load_state = lambda base_dir: state
        module.save_state = lambda base_dir, new_state: state.update(new_state)
        module._current_data_dir = lambda: "/tmp/usr/cognee/data_storage"
        module.find_faiss_indices = lambda base_dir: [
            {
                "type": "global",
                "memory_subdir": "default",
                "db_dir": "/tmp/usr/memory/default",
                "index_path": "/tmp/usr/memory/default/index.faiss",
            }
        ]

        async def cleanup(base_dir, *, delete=True):
            cleanup_calls.append(delete)
            state["cleanup_v2_done"] = True
            state["cleanup_v2_verified_done"] = True
            state.pop("cleanup_v2_pending", None)
            state.pop("cleanup_v2_reimported", None)
            return True

        async def migrate_index(index_info, migration_state, base_dir, dry_run=False):
            migrate_calls.append(index_info["memory_subdir"])
            return {"subdir": "default", "total": 1, "migrated": 1, "skipped": False}

        module.cleanup_backup_datasets = cleanup
        module.migrate_index = migrate_index

        completed = asyncio.run(module.run_migration())

        self.assertTrue(completed)
        self.assertEqual(cleanup_calls, [True])
        self.assertEqual(migrate_calls, [])

    def test_pending_cleanup_prepare_failure_returns_incomplete(self):
        module = _load_faiss_migration_module()
        state = {
            "version": 1,
            "completed": True,
            "data_dir": "/tmp/usr/cognee/data_storage",
            "cleanup_v2_pending": True,
            "cleanup_v2_reimported": False,
            "indices": {"global:default": {"status": "complete", "total": 1}},
        }

        module.load_state = lambda base_dir: state
        module.save_state = lambda base_dir, new_state: state.update(new_state)
        module._current_data_dir = lambda: "/tmp/usr/cognee/data_storage"
        module.find_faiss_indices = lambda base_dir: []

        async def cleanup(base_dir, *, delete=True):
            self.assertFalse(delete)
            return False

        module.cleanup_backup_datasets = cleanup

        completed = asyncio.run(module.run_migration())

        self.assertFalse(completed)
        self.assertTrue(state["cleanup_v2_pending"])

    def test_legacy_cleanup_done_prepare_failure_returns_incomplete(self):
        module = _load_faiss_migration_module()
        state = {
            "version": 1,
            "completed": True,
            "data_dir": "/tmp/usr/cognee/data_storage",
            "cleanup_v2_done": True,
            "indices": {"global:default": {"status": "complete", "total": 1}},
        }

        module.load_state = lambda base_dir: state
        module.save_state = lambda base_dir, new_state: state.update(new_state)
        module._current_data_dir = lambda: "/tmp/usr/cognee/data_storage"

        async def cleanup(base_dir, *, delete=True):
            self.assertFalse(delete)
            return False

        module.cleanup_backup_datasets = cleanup

        completed = asyncio.run(module.run_migration())

        self.assertFalse(completed)
        self.assertTrue(state["cleanup_v2_done"])
        self.assertNotIn("cleanup_v2_verified_done", state)

    def test_fresh_completed_migration_marks_cleanup_reimported_before_delete(self):
        module = _load_faiss_migration_module()
        state = {
            "version": 1,
            "completed": False,
            "data_dir": "/tmp/usr/cognee/data_storage",
            "indices": {},
        }
        index = {
            "type": "global",
            "memory_subdir": "default",
            "db_dir": "/tmp/usr/memory/default",
            "index_path": "/tmp/usr/memory/default/index.faiss",
        }

        module.load_state = lambda base_dir: state
        module.save_state = lambda base_dir, new_state: state.update(new_state)
        module._current_data_dir = lambda: "/tmp/usr/cognee/data_storage"
        module.find_faiss_indices = lambda base_dir: [index]
        module.backup_completed_indices = lambda indices, migration_state: None

        async def migrate_index(index_info, migration_state, base_dir, dry_run=False):
            migration_state.setdefault("indices", {})["global:default"] = {
                "status": "complete"
            }
            return {"subdir": "default", "total": 1, "migrated": 1, "skipped": False}

        async def cleanup(base_dir, *, delete=True):
            self.assertTrue(delete)
            self.assertTrue(state["cleanup_v2_pending"])
            self.assertTrue(state["cleanup_v2_reimported"])
            return False

        module.migrate_index = migrate_index
        module.cleanup_backup_datasets = cleanup

        completed = asyncio.run(module.run_migration())

        self.assertFalse(completed)
        self.assertTrue(state["cleanup_v2_pending"])
        self.assertTrue(state["cleanup_v2_reimported"])

    def test_pending_reimported_cleanup_failure_requires_verified_done(self):
        module = _load_faiss_migration_module()
        state = {
            "version": 1,
            "completed": True,
            "data_dir": "/tmp/usr/cognee/data_storage",
            "cleanup_v2_pending": True,
            "cleanup_v2_reimported": True,
            "indices": {"global:default": {"status": "complete", "total": 1}},
        }

        module.load_state = lambda base_dir: state
        module.save_state = lambda base_dir, new_state: state.update(new_state)
        module._current_data_dir = lambda: "/tmp/usr/cognee/data_storage"

        async def cleanup(base_dir, *, delete=True):
            self.assertTrue(delete)
            return False

        module.cleanup_backup_datasets = cleanup

        completed = asyncio.run(module.run_migration())

        self.assertFalse(completed)
        self.assertTrue(state["cleanup_v2_pending"])

    def test_dry_run_does_not_prepare_cleanup_for_completed_migration(self):
        module = _load_faiss_migration_module()
        state = {
            "version": 1,
            "completed": True,
            "data_dir": "/tmp/usr/cognee/data_storage",
            "indices": {"global:default": {"status": "complete", "total": 1}},
        }
        saved = []

        module.load_state = lambda base_dir: state
        module.save_state = lambda base_dir, new_state: saved.append(dict(new_state))
        module._current_data_dir = lambda: "/tmp/usr/cognee/data_storage"

        async def cleanup(base_dir, *, delete=True):
            raise AssertionError("dry run must not run cleanup")

        module.cleanup_backup_datasets = cleanup

        completed = asyncio.run(module.run_migration(dry_run=True))

        self.assertTrue(completed)
        self.assertEqual(saved, [])


if __name__ == "__main__":
    unittest.main()
