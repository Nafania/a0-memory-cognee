import asyncio
import builtins
import importlib.util
import os
import re
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _install_stubs(cognee_module):
    helpers = types.ModuleType("helpers")

    defer = types.ModuleType("helpers.defer")
    defer.THREAD_BACKGROUND = "background"

    class DeferredTask:
        start_count = 0

        def __init__(self, *args, **kwargs):
            self.alive = False
            self.event_loop_thread = types.SimpleNamespace(loop=None)

        def start_task(self, *args, **kwargs):
            type(self).start_count += 1
            self.alive = True
            return self

        def is_alive(self):
            return self.alive

    defer.DeferredTask = DeferredTask

    print_style = types.ModuleType("helpers.print_style")

    class PrintStyle:
        messages: list[tuple[str, tuple]] = []

        @classmethod
        def standard(cls, *args, **kwargs):
            cls.messages.append(("standard", args))

        @classmethod
        def warning(cls, *args, **kwargs):
            cls.messages.append(("warning", args))

        @classmethod
        def error(cls, *args, **kwargs):
            cls.messages.append(("error", args))

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

    cognee_init = types.ModuleType("usr.plugins.memory_cognee.helpers.cognee_init")
    cognee_init.get_cognee_setting = lambda key, default=None: default

    sys.modules.update(
        {
            "helpers": helpers,
            "helpers.defer": defer,
            "helpers.print_style": print_style,
            "usr.plugins.memory_cognee.helpers.cognee_init": cognee_init,
            "cognee": cognee_module,
        }
    )
    return PrintStyle


def _load_background_module(cognee_module):
    _install_stubs(cognee_module)
    module_path = REPO_ROOT / "helpers" / "cognee_background.py"
    spec = importlib.util.spec_from_file_location(
        "usr.plugins.memory_cognee.helpers.cognee_background",
        module_path,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _install_graph_engine_stub(*, is_empty: bool, graph_data=None):
    graph_module = types.ModuleType("cognee.infrastructure.databases.graph")
    graph_engine_module = types.ModuleType(
        "cognee.infrastructure.databases.graph.get_graph_engine"
    )
    context_module = types.ModuleType("cognee.context_global_variables")
    users_methods_module = types.ModuleType("cognee.modules.users.methods")

    class FakeDatabaseContext:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

    class FakeGraphEngine:
        async def is_empty(self):
            return is_empty

        async def get_graph_data(self):
            if graph_data is None:
                return [("node-id", {"name": "node"})], []
            return graph_data

    async def get_graph_engine():
        return FakeGraphEngine()

    def _create_graph_engine():
        return None

    _create_graph_engine.cache_clear = lambda: None

    graph_module.get_graph_engine = get_graph_engine
    graph_engine_module._create_graph_engine = _create_graph_engine
    context_module.set_database_global_context_variables = (
        lambda dataset_id, owner_id: FakeDatabaseContext()
    )
    context_module.graph_db_config = types.SimpleNamespace(get=lambda: {})
    users_methods_module.get_default_user = (
        lambda: _async_value(types.SimpleNamespace(id="owner-id"))
    )
    sys.modules["cognee.infrastructure"] = types.ModuleType("cognee.infrastructure")
    sys.modules["cognee.infrastructure.databases"] = types.ModuleType(
        "cognee.infrastructure.databases"
    )
    sys.modules["cognee.modules"] = types.ModuleType("cognee.modules")
    sys.modules["cognee.modules.users"] = types.ModuleType("cognee.modules.users")
    sys.modules["cognee.modules.users.methods"] = users_methods_module
    sys.modules["cognee.context_global_variables"] = context_module
    sys.modules["cognee.infrastructure.databases.graph"] = graph_module
    sys.modules[
        "cognee.infrastructure.databases.graph.get_graph_engine"
    ] = graph_engine_module


async def _async_value(value):
    return value


class CogneeBackgroundTest(unittest.TestCase):
    def tearDown(self):
        for name in list(sys.modules):
            if (
                name == "cognee"
                or name.startswith("usr.plugins.memory_cognee")
                or name.startswith("cognee.infrastructure")
                or name.startswith("cognee.context_global_variables")
                or name.startswith("cognee.modules")
            ):
                sys.modules.pop(name, None)

    def test_empty_graph_improve_error_is_non_fatal(self):
        class FakeDatasets:
            async def list_datasets(self):
                return [types.SimpleNamespace(id="dataset-id", name="default")]

            async def list_data(self, dataset_id):
                return [types.SimpleNamespace(id="data-id")]

        class FakeCognee(types.ModuleType):
            def __init__(self):
                super().__init__("cognee")
                self.cognified = []
                self.improved = []
                self.datasets = FakeDatasets()

            async def cognify(self, *, datasets, temporal_cognify, **kwargs):
                self.cognified.append((datasets, temporal_cognify))

            async def improve(self, *, dataset):
                self.improved.append(dataset)
                raise RuntimeError(
                    "EntityNotFoundError: Empty graph projected from the database."
                )

        fake_cognee = FakeCognee()
        _install_graph_engine_stub(is_empty=False)
        background = _load_background_module(fake_cognee)
        worker = background.CogneeBackgroundWorker()
        worker.mark_dirty("default")

        asyncio.run(worker.run_pipeline())

        status = worker.get_status()
        self.assertTrue(status["last_run_success"])
        self.assertIsNone(status["last_error"])
        self.assertEqual(status["dirty_datasets"], [])
        self.assertEqual(fake_cognee.improved, ["default"])

    def test_dirty_mark_added_during_cognify_is_not_cleared(self):
        class FakeDatasets:
            async def list_datasets(self):
                return [types.SimpleNamespace(id="dataset-id", name="default")]

            async def list_data(self, dataset_id):
                return [types.SimpleNamespace(id="data-id")]

        fake_cognee = types.ModuleType("cognee")
        fake_cognee.datasets = FakeDatasets()
        background = _load_background_module(fake_cognee)
        worker = background.CogneeBackgroundWorker()

        async def cognify(*, datasets, temporal_cognify, **kwargs):
            worker.mark_dirty("default")

        async def improve(*, dataset):
            return None

        fake_cognee.cognify = cognify
        fake_cognee.improve = improve
        _install_graph_engine_stub(is_empty=False)

        worker.mark_dirty("default")
        asyncio.run(worker.run_pipeline())

        status = worker.get_status()
        self.assertTrue(status["last_run_success"])
        self.assertEqual(status["dirty_datasets"], ["default"])

    def test_multiple_dirty_datasets_are_cognified_one_at_a_time(self):
        class FakeDatasets:
            async def list_datasets(self):
                return [
                    types.SimpleNamespace(id="default-id", name="default"),
                    types.SimpleNamespace(id="project-id", name="projects_alpha"),
                ]

            async def list_data(self, dataset_id):
                return [types.SimpleNamespace(id=f"data-{dataset_id}")]

        fake_cognee = types.ModuleType("cognee")
        fake_cognee.datasets = FakeDatasets()
        cognify_calls = []

        async def cognify(*, datasets, temporal_cognify, **kwargs):
            cognify_calls.append(list(datasets))

        async def improve(*, dataset):
            return None

        fake_cognee.cognify = cognify
        fake_cognee.improve = improve
        _install_graph_engine_stub(is_empty=False)

        background = _load_background_module(fake_cognee)
        cleanup_labels = []
        background._cleanup_cognee_child_processes = cleanup_labels.append
        worker = background.CogneeBackgroundWorker()
        worker.mark_dirty("default")
        worker.mark_dirty("projects_alpha")

        asyncio.run(worker.run_pipeline())

        self.assertEqual(cognify_calls, [["default"], ["projects_alpha"]])
        self.assertEqual(cleanup_labels, ["default", "projects_alpha"])
        status = worker.get_status()
        self.assertTrue(status["last_run_success"])
        self.assertEqual(status["dirty_datasets"], [])

    def test_only_current_dataset_is_marked_rebuilding(self):
        class FakeDatasets:
            async def list_datasets(self):
                return [
                    types.SimpleNamespace(id="default-id", name="default"),
                    types.SimpleNamespace(id="project-id", name="projects_alpha"),
                ]

            async def list_data(self, dataset_id):
                return [types.SimpleNamespace(id=f"data-{dataset_id}")]

        fake_cognee = types.ModuleType("cognee")
        fake_cognee.datasets = FakeDatasets()
        observed_states = []
        worker_ref = {}

        async def cognify(*, datasets, temporal_cognify, **kwargs):
            status = worker_ref["worker"].get_status()
            observed_states.append(
                (
                    datasets[0],
                    status["dataset_readiness"]["default"]["state"],
                    status["dataset_readiness"]["projects_alpha"]["state"],
                )
            )

        async def improve(*, dataset):
            return None

        fake_cognee.cognify = cognify
        fake_cognee.improve = improve
        _install_graph_engine_stub(is_empty=False)

        background = _load_background_module(fake_cognee)
        worker = background.CogneeBackgroundWorker()
        worker_ref["worker"] = worker
        worker.mark_dirty("default")
        worker.mark_dirty("projects_alpha")

        asyncio.run(worker.run_pipeline())

        self.assertEqual(
            observed_states,
            [
                ("default", "rebuilding", "dirty"),
                ("projects_alpha", "ready", "rebuilding"),
            ],
        )

    def test_running_rebuild_blocks_ready_dataset_search(self):
        fake_cognee = types.ModuleType("cognee")
        background = _load_background_module(fake_cognee)
        worker = background.CogneeBackgroundWorker()
        with worker._state_lock:
            worker._running = True
            worker._set_dataset_state_locked(
                "default",
                "ready",
                "Cognee memory graph rebuild completed",
            )
            worker._set_dataset_state_locked(
                "projects_alpha",
                "rebuilding",
                "Cognee memory graph rebuild running",
            )

        self.assertEqual(
            worker.get_search_block_reason(["default"]),
            "Cognee memory graph rebuild running for dataset(s): ['projects_alpha']",
        )

    def test_background_rebuild_passes_bounded_cognify_batches(self):
        class FakeDatasets:
            async def list_datasets(self):
                return [types.SimpleNamespace(id="dataset-id", name="default")]

            async def list_data(self, dataset_id):
                return [types.SimpleNamespace(id="data-id")]

        fake_cognee = types.ModuleType("cognee")
        fake_cognee.datasets = FakeDatasets()
        cognify_kwargs = []

        async def cognify(**kwargs):
            cognify_kwargs.append(kwargs)

        async def improve(*, dataset):
            return None

        fake_cognee.cognify = cognify
        fake_cognee.improve = improve
        _install_graph_engine_stub(is_empty=False)

        background = _load_background_module(fake_cognee)

        def setting(key, default=None):
            if key == "cognee_rebuild_data_per_batch":
                return 1
            if key == "cognee_rebuild_chunks_per_batch":
                return 1
            return default

        background.get_cognee_setting = setting
        worker = background.CogneeBackgroundWorker()
        worker.mark_dirty("default")

        asyncio.run(worker.run_pipeline())

        self.assertEqual(len(cognify_kwargs), 1)
        self.assertEqual(cognify_kwargs[0]["datasets"], ["default"])
        self.assertTrue(cognify_kwargs[0]["temporal_cognify"])
        self.assertEqual(cognify_kwargs[0]["data_per_batch"], 1)
        self.assertEqual(cognify_kwargs[0]["chunks_per_batch"], 1)

    def test_embedding_rebuild_uses_bounded_cognify_batches_by_default(self):
        class FakeDatasets:
            async def list_datasets(self):
                return [types.SimpleNamespace(id="dataset-id", name="default")]

            async def list_data(self, dataset_id):
                return [types.SimpleNamespace(id="data-id")]

        fake_cognee = types.ModuleType("cognee")
        fake_cognee.datasets = FakeDatasets()
        cognify_kwargs = []

        async def cognify(**kwargs):
            cognify_kwargs.append(kwargs)

        async def improve(*, dataset):
            return None

        fake_cognee.cognify = cognify
        fake_cognee.improve = improve
        _install_graph_engine_stub(is_empty=False)

        background = _load_background_module(fake_cognee)
        background._embedding_config_rebuild_needed = lambda: True
        background._reset_pipeline_status_for_rebuild = lambda dataset: _async_value(None)
        background._purge_vector_store_for_rebuild = lambda dataset: _async_value(None)
        worker = background.CogneeBackgroundWorker()
        worker.mark_dirty("default")

        asyncio.run(worker.run_pipeline())

        self.assertEqual(cognify_kwargs[0]["data_per_batch"], 1)
        self.assertEqual(cognify_kwargs[0]["chunks_per_batch"], 1)

    def test_cognify_repairs_corrupt_kuzu_wal_and_retries_once(self):
        class FakeDatasets:
            async def list_datasets(self):
                return [types.SimpleNamespace(id="dataset-id", name="default")]

            async def list_data(self, dataset_id):
                return [types.SimpleNamespace(id="data-id")]

        fake_cognee = types.ModuleType("cognee")
        fake_cognee.datasets = FakeDatasets()
        calls = []

        async def cognify(**kwargs):
            calls.append(("cognify", kwargs))
            if len([call for call in calls if call[0] == "cognify"]) == 1:
                raise RuntimeError(
                    "Runtime exception: Corrupted wal file. "
                    "Read out invalid WAL record type."
                )

        async def improve(*, dataset):
            calls.append(("improve", dataset))

        async def reset(dataset):
            calls.append(("reset", dataset))

        fake_cognee.cognify = cognify
        fake_cognee.improve = improve
        _install_graph_engine_stub(is_empty=False)

        background = _load_background_module(fake_cognee)
        background._repair_corrupt_kuzu_wal = lambda error: True
        background._reset_pipeline_status_for_rebuild = reset
        worker = background.CogneeBackgroundWorker()
        worker.mark_dirty("default")

        asyncio.run(worker.run_pipeline())

        status = worker.get_status()
        self.assertTrue(status["last_run_success"])
        self.assertEqual(status["dirty_datasets"], [])
        self.assertEqual([call[0] for call in calls], ["cognify", "reset", "cognify", "improve"])
        self.assertTrue(calls[0][1]["temporal_cognify"])

    def test_cognify_preflights_graph_store_before_pipeline(self):
        class FakeDatasets:
            async def list_datasets(self):
                return [types.SimpleNamespace(id="dataset-id", name="default")]

            async def list_data(self, dataset_id):
                return [types.SimpleNamespace(id="data-id")]

        fake_cognee = types.ModuleType("cognee")
        fake_cognee.datasets = FakeDatasets()
        calls = []

        async def cognify(**kwargs):
            calls.append("cognify")

        async def improve(*, dataset):
            calls.append("improve")

        async def read_graphs(*args, **kwargs):
            calls.append("preflight")
            return [
                types.SimpleNamespace(
                    dataset_name="default",
                    data_count=1,
                    graph_empty=False,
                    error=None,
                )
            ]

        fake_cognee.cognify = cognify
        fake_cognee.improve = improve
        _install_graph_engine_stub(is_empty=False)

        background = _load_background_module(fake_cognee)
        background.read_dataset_graphs = read_graphs
        worker = background.CogneeBackgroundWorker()
        worker.mark_dirty("default")

        asyncio.run(worker.run_pipeline())

        self.assertEqual(calls[:2], ["preflight", "cognify"])
        self.assertTrue(worker.get_status()["last_run_success"])

    def test_unrepaired_corrupt_wal_preflight_blocks_cognify(self):
        class FakeDatasets:
            async def list_datasets(self):
                return [types.SimpleNamespace(id="dataset-id", name="default")]

            async def list_data(self, dataset_id):
                return [types.SimpleNamespace(id="data-id")]

        fake_cognee = types.ModuleType("cognee")
        fake_cognee.datasets = FakeDatasets()

        async def cognify(**kwargs):
            raise AssertionError("cognify should not run with unrepaired corrupt WAL")

        async def read_graphs(*args, **kwargs):
            return [
                types.SimpleNamespace(
                    dataset_name="default",
                    error=(
                        "Runtime exception: Corrupted wal file. "
                        "Read out invalid WAL record type."
                    ),
                )
            ]

        fake_cognee.cognify = cognify
        fake_cognee.improve = lambda *args, **kwargs: None
        _install_graph_engine_stub(is_empty=False)

        background = _load_background_module(fake_cognee)
        background.read_dataset_graphs = read_graphs
        worker = background.CogneeBackgroundWorker()
        worker._schedule_run_soon = lambda delay=None: None
        worker.mark_dirty("default")

        asyncio.run(worker.run_pipeline())

        status = worker.get_status()
        self.assertFalse(status["last_run_success"])
        self.assertEqual(status["dirty_datasets"], ["default"])
        self.assertIn("WAL repair failed before rebuild", status["last_error"])

    def test_corrupt_wal_preflight_retries_after_child_cleanup(self):
        class FakeDatasets:
            async def list_datasets(self):
                return [types.SimpleNamespace(id="dataset-id", name="default")]

            async def list_data(self, dataset_id):
                return [types.SimpleNamespace(id="data-id")]

        fake_cognee = types.ModuleType("cognee")
        fake_cognee.datasets = FakeDatasets()
        calls = []

        async def cognify(**kwargs):
            calls.append("cognify")

        async def improve(*, dataset):
            calls.append("improve")

        read_attempts = {"count": 0}

        async def read_graphs(*args, **kwargs):
            read_attempts["count"] += 1
            calls.append(f"preflight-{read_attempts['count']}")
            if read_attempts["count"] == 1:
                return [
                    types.SimpleNamespace(
                        dataset_name="default",
                        error=(
                            "Runtime exception: Corrupted wal file. "
                            "Read out invalid WAL record type."
                        ),
                    )
                ]
            return [
                types.SimpleNamespace(
                    dataset_name="default",
                    data_count=1,
                    graph_empty=False,
                    error=None,
                )
            ]

        fake_cognee.cognify = cognify
        fake_cognee.improve = improve
        _install_graph_engine_stub(is_empty=False)

        background = _load_background_module(fake_cognee)
        background.read_dataset_graphs = read_graphs
        background._cleanup_cognee_child_processes = (
            lambda label: calls.append(f"cleanup:{label}")
        )
        worker = background.CogneeBackgroundWorker()
        worker.mark_dirty("default")

        asyncio.run(worker.run_pipeline())

        self.assertEqual(
            calls[:4],
            ["preflight-1", "cleanup:default-graph-repair-preflight", "preflight-2", "cognify"],
        )
        self.assertTrue(worker.get_status()["last_run_success"])

    def test_embedding_rebuild_reindexes_vectors_without_cognify_when_graph_exists(self):
        class FakeDatasets:
            async def list_datasets(self):
                return [types.SimpleNamespace(id="dataset-id", name="default")]

            async def list_data(self, dataset_id):
                return [types.SimpleNamespace(id="data-id")]

        fake_cognee = types.ModuleType("cognee")
        fake_cognee.datasets = FakeDatasets()
        calls = []

        async def cognify(**kwargs):
            calls.append(("cognify", kwargs))

        async def improve(*, dataset):
            calls.append(("improve", dataset))

        async def purge(dataset):
            calls.append(("purge", dataset))

        async def vector_rebuild(dataset, **kwargs):
            calls.append(("vector_rebuild", dataset))
            self.assertIn("progress_callback", kwargs)
            self.assertTrue(callable(kwargs["progress_callback"]))
            return {"nodes": 2, "skipped_nodes": 0, "edge_types": 1}

        fake_cognee.cognify = cognify
        fake_cognee.improve = improve
        _install_graph_engine_stub(is_empty=False)

        background = _load_background_module(fake_cognee)
        background._embedding_config_rebuild_needed = lambda: True
        background._dataset_has_existing_graph = lambda dataset: _async_value(True)
        background._purge_vector_store_for_rebuild = purge
        background._rebuild_embedding_vectors_from_existing_graph = vector_rebuild
        background._reset_pipeline_status_for_rebuild = (
            lambda dataset: calls.append(("reset", dataset))
        )

        worker = background.CogneeBackgroundWorker()
        worker.mark_dirty("default")

        asyncio.run(worker.run_pipeline())

        status = worker.get_status()
        self.assertTrue(status["last_run_success"])
        self.assertEqual(status["dirty_datasets"], [])
        self.assertIn(("purge", "default"), calls)
        self.assertIn(("vector_rebuild", "default"), calls)
        self.assertNotIn(("reset", "default"), calls)
        self.assertFalse(any(call[0] == "cognify" for call in calls))
        self.assertFalse(any(call[0] == "improve" for call in calls))

    def test_vector_rebuild_orchestrates_isolated_process_chunks(self):
        fake_cognee = types.ModuleType("cognee")
        background = _load_background_module(fake_cognee)
        calls = []

        async def node_count(dataset):
            self.assertEqual(dataset, "default")
            return 12001

        async def chunk(dataset, *, offset, limit, batch_size, include_edges, seen_path):
            calls.append((dataset, offset, limit, batch_size, include_edges, bool(seen_path)))
            if include_edges:
                return {"rows": 0, "nodes": 0, "skipped_nodes": 0, "edge_types": 3}
            return {
                "rows": limit,
                "nodes": limit - 1,
                "skipped_nodes": 1,
                "edge_types": 0,
            }

        background._get_existing_graph_dataset_node_count = node_count
        background._run_vector_rebuild_chunk_subprocess = chunk
        background._prepare_vector_rebuild_manifest = (
            lambda dataset, seen_path: asyncio.sleep(0, result=0)
        )
        progress_events = []

        counts = asyncio.run(
            background._rebuild_embedding_vectors_from_existing_graph(
                "default",
                batch_size=64,
                process_chunk_size=5000,
                progress_callback=lambda event: progress_events.append(dict(event)),
            )
        )

        self.assertEqual(
            calls,
            [
                ("default", 0, 5000, 64, False, True),
                ("default", 5000, 5000, 64, False, True),
                ("default", 10000, 2001, 64, False, True),
                ("default", 0, 0, 64, True, True),
            ],
        )
        self.assertEqual(counts, {"nodes": 11998, "skipped_nodes": 3, "edge_types": 3})
        self.assertEqual(
            progress_events,
            [
                {
                    "phase": "manifest",
                    "rows_done": 0,
                    "rows_total": 12001,
                    "indexed_vectors": 0,
                    "skipped_rows": 0,
                },
                {
                    "phase": "nodes",
                    "rows_done": 5000,
                    "rows_total": 12001,
                    "indexed_vectors": 4999,
                    "skipped_rows": 1,
                },
                {
                    "phase": "nodes",
                    "rows_done": 10000,
                    "rows_total": 12001,
                    "indexed_vectors": 9998,
                    "skipped_rows": 2,
                },
                {
                    "phase": "nodes",
                    "rows_done": 12001,
                    "rows_total": 12001,
                    "indexed_vectors": 11998,
                    "skipped_rows": 3,
                },
                {
                    "phase": "edges",
                    "rows_done": 12001,
                    "rows_total": 12001,
                    "indexed_vectors": 11998,
                    "skipped_rows": 3,
                    "edge_types": 3,
                },
            ],
        )

    def test_seen_manifest_persists_belongs_to_sets_by_collection_and_id(self):
        fake_cognee = types.ModuleType("cognee")
        background = _load_background_module(fake_cognee)
        fd, seen_path = tempfile.mkstemp()
        os.close(fd)
        try:
            conn = background._open_seen_manifest(seen_path)
            background._write_seen_manifest(
                conn,
                "Entity_name",
                [("same", ["one", "two"])],
            )
            self.assertEqual(
                background._read_seen_manifest(conn, "Entity_name", ["same", "missing"]),
                {"same": ["one", "two"]},
            )
            self.assertEqual(
                background._read_seen_manifest(conn, "Other_name", ["same"]),
                {},
            )
            conn.close()
        finally:
            for suffix in ("", "-wal", "-shm"):
                try:
                    os.remove(f"{seen_path}{suffix}")
                except FileNotFoundError:
                    pass

    def test_vector_rebuild_chunk_cli_accepts_seen_manifest_path(self):
        fake_cognee = types.ModuleType("cognee")
        background = _load_background_module(fake_cognee)
        cognee_init = sys.modules["usr.plugins.memory_cognee.helpers.cognee_init"]
        cognee_init.configure_cognee = lambda: None
        calls = []

        async def rebuild_chunk(dataset, **kwargs):
            calls.append({"dataset": dataset, **kwargs})
            return {"rows": 1, "nodes": 1, "skipped_nodes": 0, "edge_types": 0}

        background._rebuild_embedding_vectors_chunk_in_current_process = rebuild_chunk
        fd, result_path = tempfile.mkstemp()
        os.close(fd)
        try:
            code = background._run_vector_rebuild_chunk_cli(
                [
                    "cognee_background.py",
                    "vector-rebuild-chunk",
                    "default",
                    "10",
                    "20",
                    "30",
                    "0",
                    result_path,
                    "/tmp/seen.sqlite3",
                ]
            )
            self.assertEqual(code, 0)
            self.assertEqual(calls[0]["seen_path"], "/tmp/seen.sqlite3")
            self.assertEqual(calls[0]["offset"], 10)
        finally:
            try:
                os.remove(result_path)
            except FileNotFoundError:
                pass

    def test_lancedb_vector_rebuild_uses_delete_add_upsert_without_merge_insert(self):
        from pydantic import BaseModel, Field, create_model

        fake_cognee = types.ModuleType("cognee")
        background = _load_background_module(fake_cognee)

        vector_module = types.ModuleType("cognee.infrastructure.databases.vector")
        lancedb_package = types.ModuleType("cognee.infrastructure.databases.vector.lancedb")
        adapter_module = types.ModuleType(
            "cognee.infrastructure.databases.vector.lancedb.LanceDBAdapter"
        )

        class IndexSchema(BaseModel):
            id: str
            text: str
            metadata: dict = {"index_fields": ["text"]}
            belongs_to_set: list[str] = []

        adapter_module.IndexSchema = IndexSchema
        sys.modules["cognee.infrastructure.databases.vector"] = vector_module
        sys.modules["cognee.infrastructure.databases.vector.lancedb"] = lancedb_package
        sys.modules[
            "cognee.infrastructure.databases.vector.lancedb.LanceDBAdapter"
        ] = adapter_module

        class SourcePoint(BaseModel):
            id: str
            name: str
            belongs_to_set: list[str] = []
            metadata: dict = Field(default_factory=lambda: {"index_fields": ["name"]})

        class FakeQuery:
            def __init__(self, collection):
                self.collection = collection
                self.where_clause = ""

            def where(self, where_clause):
                self.where_clause = where_clause
                return self

            async def to_list(self):
                ids = [
                    match.replace("''", "'")
                    for match in re.findall(r"'((?:''|[^'])*)'", self.where_clause)
                ]
                return [
                    self.collection.rows[row_id]
                    for row_id in ids
                    if row_id in self.collection.rows
                ]

        class FakeCollection:
            def __init__(self):
                self.rows = {
                    "same": {
                        "id": "same",
                        "payload": {"belongs_to_set": ["old"], "text": "old"},
                    }
                }
                self.deleted = []
                self.added_batches = []
                self.merge_insert_called = False

            def query(self):
                return FakeQuery(self)

            async def delete(self, where_clause):
                self.deleted.append(where_clause)
                for row_id in re.findall(r"'((?:''|[^'])*)'", where_clause):
                    self.rows.pop(row_id.replace("''", "'"), None)

            async def add(self, records):
                self.added_batches.append(records)
                for record in records:
                    payload = record.payload
                    if hasattr(payload, "model_dump"):
                        payload = payload.model_dump()
                    self.rows[record.id] = {
                        "id": record.id,
                        "payload": payload,
                        "vector": record.vector,
                    }

            def merge_insert(self, *_args, **_kwargs):
                self.merge_insert_called = True
                raise AssertionError("merge_insert must not be used")

        class FakeEmbeddingEngine:
            def get_batch_size(self):
                return 16

            def get_vector_size(self):
                return 2

        class FakeVectorEngine:
            name = "LanceDB"

            def __init__(self):
                self.embedding_engine = FakeEmbeddingEngine()
                self.collection = FakeCollection()
                self.created = []
                self.embedded = []

            async def create_vector_index(self, type_name, field_name):
                self.created.append((type_name, field_name))

            async def get_collection(self, collection_name):
                self.collection_name = collection_name
                return self.collection

            async def embed_data(self, texts):
                self.embedded.append(list(texts))
                return [[float(len(text)), 1.0] for text in texts]

            def get_data_point_schema(self, model_type):
                return model_type

            def _make_lance_datapoint_cls(self, payload_schema, vector_size):
                return create_model(
                    "FakeLanceDataPoint",
                    id=(str, ...),
                    vector=(list[float], ...),
                    payload=(payload_schema, ...),
                )

            def _records_for_write(self, records):
                return records

        vector_engine = FakeVectorEngine()
        points = [
            SourcePoint(id="same", name="first", belongs_to_set=["first-tag"]),
            SourcePoint(id="same", name="second", belongs_to_set=["second-tag"]),
            SourcePoint(id="new", name="new text", belongs_to_set=["new-tag"]),
        ]

        asyncio.run(
            background._index_data_points_for_vector_rebuild(
                points,
                vector_engine=vector_engine,
                max_batch_size=16,
            )
        )

        self.assertEqual(vector_engine.created, [("SourcePoint", "name")])
        self.assertEqual(vector_engine.collection_name, "SourcePoint_name")
        self.assertFalse(vector_engine.collection.merge_insert_called)
        self.assertEqual(vector_engine.embedded, [["second", "new text"]])
        self.assertEqual(len(vector_engine.collection.added_batches), 1)
        same = vector_engine.collection.rows["same"]["payload"]
        self.assertEqual(same["text"], "second")
        self.assertEqual(same["belongs_to_set"], ["old", "first-tag", "second-tag"])

    def test_lancedb_vector_rebuild_claims_prepared_manifest_once(self):
        from pydantic import BaseModel, Field, create_model

        fake_cognee = types.ModuleType("cognee")
        background = _load_background_module(fake_cognee)

        adapter_module = types.ModuleType(
            "cognee.infrastructure.databases.vector.lancedb.LanceDBAdapter"
        )

        class IndexSchema(BaseModel):
            id: str
            text: str
            metadata: dict = {"index_fields": ["text"]}
            belongs_to_set: list[str] = []

        adapter_module.IndexSchema = IndexSchema
        sys.modules[
            "cognee.infrastructure.databases.vector.lancedb.LanceDBAdapter"
        ] = adapter_module

        class SourcePoint(BaseModel):
            id: str
            name: str
            belongs_to_set: list[str] = []
            metadata: dict = Field(default_factory=lambda: {"index_fields": ["name"]})

        class FakeCollection:
            def __init__(self):
                self.rows = {}
                self.deleted = []
                self.added_batches = []

            async def delete(self, where_clause):
                self.deleted.append(where_clause)

            async def add(self, records):
                self.added_batches.append(records)
                for record in records:
                    payload = record.payload
                    if hasattr(payload, "model_dump"):
                        payload = payload.model_dump()
                    self.rows[record.id] = {"payload": payload, "vector": record.vector}

        class FakeEmbeddingEngine:
            def get_batch_size(self):
                return 16

            def get_vector_size(self):
                return 2

        class FakeVectorEngine:
            name = "LanceDB"

            def __init__(self):
                self.embedding_engine = FakeEmbeddingEngine()
                self.collection = FakeCollection()
                self.embedded = []

            async def create_vector_index(self, type_name, field_name):
                return None

            async def get_collection(self, collection_name):
                return self.collection

            async def embed_data(self, texts):
                self.embedded.append(list(texts))
                return [[float(len(text)), 1.0] for text in texts]

            def get_data_point_schema(self, model_type):
                return model_type

            def _make_lance_datapoint_cls(self, payload_schema, vector_size):
                return create_model(
                    "FakeLanceDataPointPrepared",
                    id=(str, ...),
                    vector=(list[float], ...),
                    payload=(payload_schema, ...),
                )

            def _records_for_write(self, records):
                return records

        fd, seen_path = tempfile.mkstemp()
        os.close(fd)
        try:
            conn = background._open_seen_manifest(seen_path)
            background._upsert_manifest_entries(
                conn,
                "SourcePoint_name",
                [
                    ("same", "final text", ["old", "new"]),
                    ("new", "new final", ["fresh"]),
                ],
            )
            background._set_seen_manifest_prepared(conn)
            conn.close()

            vector_engine = FakeVectorEngine()
            points = [
                SourcePoint(id="same", name="ignored first", belongs_to_set=["old"]),
                SourcePoint(id="same", name="ignored second", belongs_to_set=["new"]),
                SourcePoint(id="new", name="ignored new", belongs_to_set=["fresh"]),
            ]

            asyncio.run(
                background._index_data_points_for_vector_rebuild(
                    points,
                    vector_engine=vector_engine,
                    max_batch_size=16,
                    seen_path=seen_path,
                )
            )

            self.assertEqual(vector_engine.collection.deleted, [])
            self.assertEqual(vector_engine.embedded, [["final text", "new final"]])
            self.assertEqual(
                vector_engine.collection.rows["same"]["payload"]["belongs_to_set"],
                ["old", "new"],
            )

            asyncio.run(
                background._index_data_points_for_vector_rebuild(
                    points,
                    vector_engine=vector_engine,
                    max_batch_size=16,
                    seen_path=seen_path,
                )
            )
            self.assertEqual(vector_engine.embedded, [["final text", "new final"]])
        finally:
            for suffix in ("", "-wal", "-shm"):
                try:
                    os.remove(f"{seen_path}{suffix}")
                except FileNotFoundError:
                    pass

    def test_raw_graph_node_fetch_decodes_json_without_uuid_processors(self):
        fake_cognee = types.ModuleType("cognee")
        background = _load_background_module(fake_cognee)
        statements = []

        class FakeResult:
            def mappings(self):
                return self

            def all(self):
                return [
                    {
                        "id": 12.5,
                        "slug": "slug-id",
                        "type": "Entity",
                        "indexed_fields": '["name"]',
                        "attributes": '{"id":"node-id","name":"Ivan"}',
                    }
                ]

        class FakeSession:
            async def execute(self, statement, params):
                statements.append((str(statement), params))
                return FakeResult()

        rows = asyncio.run(
            background._fetch_graph_node_rows(
                FakeSession(),
                "dataset-id",
                offset=10,
                limit=5,
            )
        )

        self.assertIn("FROM nodes", statements[0][0])
        self.assertIn("CAST(dataset_id AS TEXT)", statements[0][0])
        self.assertEqual(statements[0][1], {"dataset_id": "dataset-id", "offset": 10, "limit": 5})
        self.assertEqual(rows[0].id, "12.5")
        self.assertEqual(rows[0].slug, "slug-id")
        self.assertEqual(rows[0].indexed_fields, ["name"])
        self.assertEqual(rows[0].attributes, {"id": "node-id", "name": "Ivan"})

    def test_existing_graph_count_uses_raw_dataset_id_text(self):
        fake_cognee = types.ModuleType("cognee")
        background = _load_background_module(fake_cognee)
        seen_node_params = []

        class FakeResult:
            def __init__(self, *, row=None, scalar=None):
                self._row = row
                self._scalar = scalar

            def mappings(self):
                return self

            def first(self):
                return self._row

            def scalar_one(self):
                return self._scalar

        class FakeSession:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

            async def execute(self, statement, params=None):
                sql = str(statement)
                if "FROM datasets" in sql:
                    return FakeResult(row={"id": "00afc7102c0c5d61957ec452672842ae"})
                if "FROM nodes" in sql:
                    seen_node_params.append(dict(params or {}))
                    return FakeResult(scalar=117290)
                raise AssertionError(sql)

        class FakeEngine:
            def get_async_session(self):
                return FakeSession()

        relational = types.ModuleType("cognee.infrastructure.databases.relational")
        relational.get_relational_engine = lambda: FakeEngine()
        sys.modules["cognee.infrastructure"] = types.ModuleType("cognee.infrastructure")
        sys.modules["cognee.infrastructure.databases"] = types.ModuleType(
            "cognee.infrastructure.databases"
        )
        sys.modules["cognee.infrastructure.databases.relational"] = relational

        count = asyncio.run(
            background._get_existing_graph_dataset_node_count("default")
        )

        self.assertEqual(count, 117290)
        self.assertEqual(
            seen_node_params,
            [{"dataset_id": "00afc7102c0c5d61957ec452672842ae"}],
        )

    def test_close_cached_vector_engine_evicts_and_closes_current_engine(self):
        fake_cognee = types.ModuleType("cognee")
        background = _load_background_module(fake_cognee)
        calls = []

        config_module = types.ModuleType("cognee.infrastructure.databases.vector.config")
        create_module = types.ModuleType(
            "cognee.infrastructure.databases.vector.create_vector_engine"
        )

        config = {"vector_db_provider": "lancedb", "vector_db_url": "/tmp/db"}
        config_module.get_vectordb_context_config = lambda: config

        class FakeEngine:
            async def close(self):
                calls.append(("close", None))

        def is_cached(**kwargs):
            calls.append(("cached", kwargs))
            return True

        def create(**kwargs):
            calls.append(("create", kwargs))
            return FakeEngine()

        def evict(**kwargs):
            calls.append(("evict", kwargs))
            return True

        create_module.is_vector_engine_cached = is_cached
        create_module.create_vector_engine = create
        create_module.evict_vector_engine = evict
        sys.modules["cognee.infrastructure.databases.vector.config"] = config_module
        sys.modules[
            "cognee.infrastructure.databases.vector.create_vector_engine"
        ] = create_module

        asyncio.run(background._close_cached_vector_engine())

        self.assertEqual(
            calls,
            [
                ("cached", config),
                ("create", config),
                ("evict", config),
                ("close", None),
            ],
        )

    def test_embedding_rebuild_purges_vector_store_before_cognify(self):
        class FakeDatasets:
            async def list_datasets(self):
                return [types.SimpleNamespace(id="dataset-id", name="default")]

            async def list_data(self, dataset_id):
                return [types.SimpleNamespace(id="data-id")]

        fake_cognee = types.ModuleType("cognee")
        fake_cognee.datasets = FakeDatasets()
        calls = []

        async def cognify(*, datasets, temporal_cognify, **kwargs):
            calls.append(("cognify", list(datasets)))

        async def improve(*, dataset):
            calls.append(("improve", dataset))

        fake_cognee.cognify = cognify
        fake_cognee.improve = improve
        _install_graph_engine_stub(is_empty=False)

        background = _load_background_module(fake_cognee)
        background._embedding_config_rebuild_needed = lambda: True

        async def reset(dataset):
            calls.append(("reset", dataset))

        async def purge(dataset):
            calls.append(("purge", dataset))

        background._reset_pipeline_status_for_rebuild = reset
        background._purge_vector_store_for_rebuild = purge
        worker = background.CogneeBackgroundWorker()
        worker.mark_dirty("default")

        asyncio.run(worker.run_pipeline())

        self.assertEqual(
            calls,
            [
                ("reset", "default"),
                ("purge", "default"),
                ("cognify", ["default"]),
                ("improve", "default"),
            ],
        )

    def test_empty_search_does_not_nudge_unknown_dataset(self):
        fake_cognee = types.ModuleType("cognee")
        background = _load_background_module(fake_cognee)
        worker = background.CogneeBackgroundWorker()

        nudged = worker.nudge_rebuild_if_unready(["default"], "empty graph")

        self.assertFalse(nudged)
        self.assertEqual(worker.get_status()["dirty_datasets"], [])

    def test_empty_search_does_not_nudge_ready_dataset_after_other_failure(self):
        fake_cognee = types.ModuleType("cognee")
        background = _load_background_module(fake_cognee)
        worker = background.CogneeBackgroundWorker()
        with worker._state_lock:
            worker._last_run_success = False
            worker._set_dataset_state_locked(
                "default",
                "ready",
                "Cognee memory graph rebuild completed",
            )
            worker._set_dataset_state_locked(
                "projects_alpha",
                "failed",
                "Cognee memory graph rebuild failed",
            )

        nudged = worker.nudge_rebuild_if_unready(["default"], "empty graph")

        self.assertFalse(nudged)
        self.assertEqual(worker.get_status()["dirty_datasets"], [])

    def test_dirty_dataset_blocks_search_until_successful_rebuild(self):
        class FakeDatasets:
            async def list_datasets(self):
                return [types.SimpleNamespace(id="dataset-id", name="default")]

            async def list_data(self, dataset_id):
                return [types.SimpleNamespace(id="data-id")]

        fake_cognee = types.ModuleType("cognee")

        async def cognify(*, datasets, temporal_cognify, **kwargs):
            return None

        async def improve(*, dataset):
            return None

        fake_cognee.cognify = cognify
        fake_cognee.improve = improve
        fake_cognee.datasets = FakeDatasets()
        _install_graph_engine_stub(is_empty=False)

        background = _load_background_module(fake_cognee)
        worker = background.CogneeBackgroundWorker()

        worker.mark_dirty("default")

        self.assertEqual(
            worker.get_search_block_reason(["default"]),
            "Cognee memory graph rebuild pending for dataset(s): ['default']",
        )
        self.assertEqual(
            worker.get_status()["dataset_readiness"]["default"]["state"],
            "dirty",
        )

        asyncio.run(worker.run_pipeline())

        self.assertIsNone(worker.get_search_block_reason(["default"]))
        self.assertEqual(
            worker.get_status()["dataset_readiness"]["default"]["state"],
            "ready",
        )

    def test_failed_rebuild_keeps_dataset_search_blocked(self):
        class FakeDatasets:
            async def list_datasets(self):
                return [types.SimpleNamespace(id="dataset-id", name="default")]

            async def list_data(self, dataset_id):
                return [types.SimpleNamespace(id="data-id")]

        fake_cognee = types.ModuleType("cognee")

        async def cognify(*, datasets, temporal_cognify, **kwargs):
            return None

        async def improve(*, dataset):
            return None

        fake_cognee.cognify = cognify
        fake_cognee.improve = improve
        fake_cognee.datasets = FakeDatasets()
        _install_graph_engine_stub(is_empty=True)

        background = _load_background_module(fake_cognee)
        worker = background.CogneeBackgroundWorker()
        scheduled_runs = []
        worker._schedule_run_soon = lambda delay=None: scheduled_runs.append(delay)
        worker.mark_dirty("default")
        scheduled_runs.clear()

        asyncio.run(worker.run_pipeline())

        block_reason = worker.get_search_block_reason(["default"])
        self.assertIn("Cognee memory graph rebuild failed", block_reason)
        self.assertEqual(
            worker.get_status()["dataset_readiness"]["default"]["state"],
            "failed",
        )
        self.assertEqual(scheduled_runs, [30.0])

    def test_successful_rebuild_logs_ready_summary(self):
        class FakeDatasets:
            async def list_datasets(self):
                return [types.SimpleNamespace(id="dataset-id", name="default")]

            async def list_data(self, dataset_id):
                return [types.SimpleNamespace(id="data-id")]

        fake_cognee = types.ModuleType("cognee")

        async def cognify(*, datasets, temporal_cognify, **kwargs):
            return None

        async def improve(*, dataset):
            return None

        fake_cognee.cognify = cognify
        fake_cognee.improve = improve
        fake_cognee.datasets = FakeDatasets()
        _install_graph_engine_stub(is_empty=False)

        background = _load_background_module(fake_cognee)
        worker = background.CogneeBackgroundWorker()
        worker.mark_dirty("default")

        asyncio.run(worker.run_pipeline())

        messages = sys.modules["helpers.print_style"].PrintStyle.messages
        self.assertTrue(
            any(
                level == "standard"
                and "Cognee rebuild readiness: READY" in " ".join(str(arg) for arg in args)
                for level, args in messages
            )
        )

    def test_failed_rebuild_logs_blocked_summary(self):
        class FakeDatasets:
            async def list_datasets(self):
                return [types.SimpleNamespace(id="dataset-id", name="default")]

            async def list_data(self, dataset_id):
                return [types.SimpleNamespace(id="data-id")]

        fake_cognee = types.ModuleType("cognee")

        async def cognify(*, datasets, temporal_cognify, **kwargs):
            return None

        async def improve(*, dataset):
            return None

        fake_cognee.cognify = cognify
        fake_cognee.improve = improve
        fake_cognee.datasets = FakeDatasets()
        _install_graph_engine_stub(is_empty=True)

        background = _load_background_module(fake_cognee)
        worker = background.CogneeBackgroundWorker()
        worker._schedule_run_soon = lambda delay=None: None
        worker.mark_dirty("default")

        asyncio.run(worker.run_pipeline())

        messages = sys.modules["helpers.print_style"].PrintStyle.messages
        self.assertTrue(
            any(
                level == "warning"
                and "Cognee rebuild readiness: BLOCKED" in " ".join(str(arg) for arg in args)
                and "retry_scheduled=True" in " ".join(str(arg) for arg in args)
                for level, args in messages
            )
        )

    def test_failed_rebuild_resets_pipeline_status_before_retry(self):
        class FakeDatasets:
            async def list_datasets(self):
                return [types.SimpleNamespace(id="dataset-id", name="default")]

            async def list_data(self, dataset_id):
                return [types.SimpleNamespace(id="data-id")]

        fake_cognee = types.ModuleType("cognee")

        async def cognify(*, datasets, temporal_cognify, **kwargs):
            return None

        async def improve(*, dataset):
            return None

        fake_cognee.cognify = cognify
        fake_cognee.improve = improve
        fake_cognee.datasets = FakeDatasets()
        _install_graph_engine_stub(is_empty=True)

        background = _load_background_module(fake_cognee)
        worker = background.CogneeBackgroundWorker()
        worker._schedule_run_soon = lambda delay=None: None
        worker.mark_dirty("default")
        asyncio.run(worker.run_pipeline())

        reset_calls = []

        async def reset_pipeline_status(dataset):
            reset_calls.append(dataset)

        background._reset_pipeline_status_for_rebuild = reset_pipeline_status
        asyncio.run(worker.run_pipeline())

        self.assertEqual(reset_calls, ["default"])

    def test_failed_rebuild_reset_survives_new_dirty_mark(self):
        class FakeDatasets:
            async def list_datasets(self):
                return [types.SimpleNamespace(id="dataset-id", name="default")]

            async def list_data(self, dataset_id):
                return [types.SimpleNamespace(id="data-id")]

        fake_cognee = types.ModuleType("cognee")

        async def cognify(*, datasets, temporal_cognify, **kwargs):
            return None

        async def improve(*, dataset):
            return None

        fake_cognee.cognify = cognify
        fake_cognee.improve = improve
        fake_cognee.datasets = FakeDatasets()
        _install_graph_engine_stub(is_empty=True)

        background = _load_background_module(fake_cognee)
        worker = background.CogneeBackgroundWorker()
        worker._schedule_run_soon = lambda delay=None: None
        worker.mark_dirty("default")
        asyncio.run(worker.run_pipeline())

        worker.mark_dirty("default")
        self.assertEqual(
            worker.get_status()["dataset_readiness"]["default"]["state"],
            "dirty",
        )
        reset_calls = []

        async def reset_pipeline_status(dataset):
            reset_calls.append(dataset)

        background._reset_pipeline_status_for_rebuild = reset_pipeline_status
        asyncio.run(worker.run_pipeline())

        self.assertEqual(reset_calls, ["default"])

    def test_cognee_import_failure_marks_dataset_failed_for_retry(self):
        fake_cognee = types.ModuleType("cognee")
        background = _load_background_module(fake_cognee)
        worker = background.CogneeBackgroundWorker()
        scheduled_runs = []
        worker._schedule_run_soon = lambda delay=None: scheduled_runs.append(delay)
        worker.mark_dirty("default")
        scheduled_runs.clear()
        original_import = builtins.__import__

        def failing_import(name, *args, **kwargs):
            if name == "cognee":
                raise ImportError("forced cognee import failure")
            return original_import(name, *args, **kwargs)

        try:
            builtins.__import__ = failing_import
            asyncio.run(worker.run_pipeline())
        finally:
            builtins.__import__ = original_import

        status = worker.get_status()
        self.assertFalse(status["running"])
        self.assertFalse(status["last_run_success"])
        self.assertEqual(status["dataset_readiness"]["default"]["state"], "failed")
        self.assertIn("Cognee import failed", status["last_error"])
        self.assertIn("default", status["dirty_datasets"])
        self.assertEqual(scheduled_runs, [30.0])

    def test_hung_cognify_times_out_and_unblocks_rebuild_state(self):
        class FakeDatasets:
            async def list_datasets(self):
                return [types.SimpleNamespace(id="dataset-id", name="default")]

            async def list_data(self, dataset_id):
                return [types.SimpleNamespace(id="data-id")]

        fake_cognee = types.ModuleType("cognee")

        async def cognify(*, datasets, temporal_cognify, **kwargs):
            await asyncio.sleep(0.05)

        async def improve(*, dataset):
            return None

        fake_cognee.cognify = cognify
        fake_cognee.improve = improve
        fake_cognee.datasets = FakeDatasets()
        _install_graph_engine_stub(is_empty=False)

        background = _load_background_module(fake_cognee)
        background.get_cognee_setting = (
            lambda key, default=None: 0.01
            if key == "cognee_operation_timeout_seconds"
            else default
        )
        worker = background.CogneeBackgroundWorker()
        scheduled_runs = []
        worker._schedule_run_soon = lambda delay=None: scheduled_runs.append(delay)
        worker.mark_dirty("default")
        scheduled_runs.clear()

        asyncio.run(worker.run_pipeline())

        status = worker.get_status()
        self.assertFalse(status["running"])
        self.assertFalse(status["last_run_success"])
        self.assertEqual(status["dataset_readiness"]["default"]["state"], "failed")
        self.assertIn("timed out", status["last_error"].lower())
        self.assertIn("Cognee memory graph rebuild failed", worker.get_search_block_reason(["default"]))
        self.assertEqual(scheduled_runs, [30.0])

    def test_embedding_rebuild_cognify_is_not_cut_by_normal_operation_timeout(self):
        class FakeDatasets:
            async def list_datasets(self):
                return [types.SimpleNamespace(id="dataset-id", name="default")]

            async def list_data(self, dataset_id):
                return [types.SimpleNamespace(id="data-id")]

        fake_cognee = types.ModuleType("cognee")

        async def cognify(*, datasets, temporal_cognify, **kwargs):
            await asyncio.sleep(0.05)

        async def improve(*, dataset):
            return None

        fake_cognee.cognify = cognify
        fake_cognee.improve = improve
        fake_cognee.datasets = FakeDatasets()
        _install_graph_engine_stub(is_empty=False)

        background = _load_background_module(fake_cognee)
        background.get_cognee_setting = (
            lambda key, default=None: 0.01
            if key == "cognee_operation_timeout_seconds"
            else default
        )
        background._embedding_config_rebuild_needed = lambda: True
        background._reset_pipeline_status_for_rebuild = lambda dataset: _async_value(None)
        background._purge_vector_store_for_rebuild = lambda dataset: _async_value(None)

        worker = background.CogneeBackgroundWorker()
        worker.mark_dirty("default")

        asyncio.run(worker.run_pipeline())

        status = worker.get_status()
        self.assertFalse(status["running"])
        self.assertTrue(status["last_run_success"])
        self.assertIsNone(status["last_error"])
        self.assertEqual(status["dirty_datasets"], [])

    def test_cancelled_rebuild_marks_state_failed_before_exit(self):
        fake_cognee = types.ModuleType("cognee")

        async def cognify(*, datasets, temporal_cognify, **kwargs):
            raise asyncio.CancelledError()

        fake_cognee.cognify = cognify
        fake_cognee.improve = lambda dataset: None

        background = _load_background_module(fake_cognee)
        worker = background.CogneeBackgroundWorker()
        scheduled_runs = []
        worker._schedule_run_soon = lambda delay=None: scheduled_runs.append(delay)
        worker.mark_dirty("default")
        scheduled_runs.clear()

        with self.assertRaises(asyncio.CancelledError):
            asyncio.run(worker.run_pipeline())

        status = worker.get_status()
        self.assertFalse(status["running"])
        self.assertEqual(status["dataset_readiness"]["default"]["state"], "failed")
        self.assertIn(
            "interrupted before readiness update",
            status["dataset_readiness"]["default"]["reason"],
        )
        self.assertIn("default", status["dirty_datasets"])
        self.assertEqual(scheduled_runs, [30.0])

    def test_cancelled_rebuild_does_not_fail_queued_datasets(self):
        fake_cognee = types.ModuleType("cognee")

        async def cognify(*, datasets, temporal_cognify, **kwargs):
            raise asyncio.CancelledError()

        fake_cognee.cognify = cognify
        fake_cognee.improve = lambda dataset: None

        background = _load_background_module(fake_cognee)
        worker = background.CogneeBackgroundWorker()
        worker._schedule_run_soon = lambda delay=None: None
        worker.mark_dirty("default")
        worker.mark_dirty("projects_alpha")

        with self.assertRaises(asyncio.CancelledError):
            asyncio.run(worker.run_pipeline())

        status = worker.get_status()
        self.assertEqual(status["dataset_readiness"]["default"]["state"], "failed")
        self.assertEqual(
            status["dataset_readiness"]["projects_alpha"]["state"],
            "dirty",
        )

    def test_stale_rebuilding_state_does_not_clear_active_run(self):
        fake_cognee = types.ModuleType("cognee")
        background = _load_background_module(fake_cognee)
        background.get_cognee_setting = (
            lambda key, default=None: 1
            if key == "cognee_rebuild_stale_after_seconds"
            else default
        )
        worker = background.CogneeBackgroundWorker()
        scheduled_runs = []
        worker._schedule_run_soon = lambda delay=None: scheduled_runs.append(delay)
        worker.mark_dirty("default")
        scheduled_runs.clear()
        with worker._state_lock:
            worker._running = True
            worker._run_scheduled = True
            worker._set_dataset_state_locked(
                "default",
                "rebuilding",
                "Cognee memory graph rebuild running",
            )
            worker._dataset_readiness["default"]["updated_at"] = time.monotonic() - 5

        reason = worker.get_search_block_reason(["default"])

        status = worker.get_status()
        self.assertIn("Cognee memory graph rebuild running", reason)
        self.assertEqual(status["dataset_readiness"]["default"]["state"], "rebuilding")
        self.assertTrue(status["running"])
        self.assertTrue(worker._run_scheduled)
        self.assertIn("default", status["dirty_datasets"])
        self.assertEqual(status["pipeline_reset_datasets"], [])
        self.assertEqual(scheduled_runs, [])

    def test_stale_rebuilding_state_is_marked_failed_for_retry_when_not_running(self):
        fake_cognee = types.ModuleType("cognee")
        background = _load_background_module(fake_cognee)
        background.get_cognee_setting = (
            lambda key, default=None: 1
            if key == "cognee_rebuild_stale_after_seconds"
            else default
        )
        worker = background.CogneeBackgroundWorker()
        scheduled_runs = []
        worker._schedule_run_soon = lambda delay=None: scheduled_runs.append(delay)
        worker.mark_dirty("default")
        scheduled_runs.clear()
        with worker._state_lock:
            worker._running = False
            worker._run_scheduled = False
            worker._set_dataset_state_locked(
                "default",
                "rebuilding",
                "Cognee memory graph rebuild running",
            )
            worker._dataset_readiness["default"]["updated_at"] = time.monotonic() - 5

        reason = worker.get_search_block_reason(["default"])

        status = worker.get_status()
        self.assertIn("Cognee memory graph rebuild failed", reason)
        self.assertEqual(status["dataset_readiness"]["default"]["state"], "failed")
        self.assertIn("stale", status["dataset_readiness"]["default"]["reason"])
        self.assertFalse(status["running"])
        self.assertFalse(worker._run_scheduled)
        self.assertIn("default", status["dirty_datasets"])
        self.assertEqual(status["pipeline_reset_datasets"], ["default"])
        self.assertEqual(scheduled_runs, [30.0])

    def test_start_is_idempotent(self):
        fake_cognee = types.ModuleType("cognee")
        background = _load_background_module(fake_cognee)
        worker = background.CogneeBackgroundWorker()
        deferred_task = sys.modules["helpers.defer"].DeferredTask

        first = worker.start()
        second = worker.start()

        self.assertIs(first, second)
        self.assertEqual(deferred_task.start_count, 1)

    def test_dirty_mark_before_worker_start_does_not_run_on_current_loop(self):
        fake_cognee = types.ModuleType("cognee")
        cognify_calls = []

        async def cognify(**kwargs):
            cognify_calls.append(kwargs)

        fake_cognee.cognify = cognify
        background = _load_background_module(fake_cognee)
        worker = background.CogneeBackgroundWorker()

        async def mark_inside_loop():
            worker.mark_dirty("default")
            await asyncio.sleep(0.01)

        asyncio.run(mark_inside_loop())

        self.assertEqual(cognify_calls, [])
        self.assertFalse(worker.get_status()["running"])
        self.assertFalse(worker._run_scheduled)
        self.assertEqual(worker.get_status()["dirty_datasets"], ["default"])

    def test_get_instance_is_thread_safe(self):
        fake_cognee = types.ModuleType("cognee")
        background = _load_background_module(fake_cognee)
        worker_cls = background.CogneeBackgroundWorker
        original_init = worker_cls.__init__
        barrier = threading.Barrier(12)
        instances = []

        def slow_init(self):
            time.sleep(0.01)
            original_init(self)

        worker_cls.__init__ = slow_init
        try:
            def get_worker():
                barrier.wait()
                instances.append(worker_cls.get_instance())

            threads = [threading.Thread(target=get_worker) for _ in range(12)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        finally:
            worker_cls.__init__ = original_init
            worker_cls._instance = None

        self.assertEqual(len({id(instance) for instance in instances}), 1)

    def test_empty_graph_with_dataset_data_keeps_dataset_dirty(self):
        class FakeDatasets:
            async def list_datasets(self):
                return [types.SimpleNamespace(id="dataset-id", name="default")]

            async def list_data(self, dataset_id):
                return [types.SimpleNamespace(id="data-id")]

        fake_cognee = types.ModuleType("cognee")

        async def cognify(*, datasets, temporal_cognify, **kwargs):
            return None

        async def improve(*, dataset):
            return None

        fake_cognee.cognify = cognify
        fake_cognee.improve = improve
        fake_cognee.datasets = FakeDatasets()
        _install_graph_engine_stub(is_empty=True)

        background = _load_background_module(fake_cognee)
        worker = background.CogneeBackgroundWorker()
        scheduled_runs = []
        worker._schedule_run_soon = lambda delay=None: scheduled_runs.append(delay)
        worker.mark_dirty("default")
        scheduled_runs.clear()

        asyncio.run(worker.run_pipeline())

        status = worker.get_status()
        self.assertFalse(status["last_run_success"])
        self.assertEqual(status["dirty_datasets"], ["default"])
        self.assertIn("graph is still empty", status["last_error"])
        self.assertEqual(scheduled_runs, [30.0])

    def test_non_empty_graph_without_exported_nodes_marks_ready(self):
        class FakeDatasets:
            async def list_datasets(self):
                return [types.SimpleNamespace(id="dataset-id", name="default")]

            async def list_data(self, dataset_id):
                return [types.SimpleNamespace(id="data-id")]

        fake_cognee = types.ModuleType("cognee")

        async def cognify(*, datasets, temporal_cognify, **kwargs):
            return None

        async def improve(*, dataset):
            return None

        fake_cognee.cognify = cognify
        fake_cognee.improve = improve
        fake_cognee.datasets = FakeDatasets()
        _install_graph_engine_stub(is_empty=False, graph_data=([], []))

        background = _load_background_module(fake_cognee)
        worker = background.CogneeBackgroundWorker()
        worker.mark_dirty("default")

        asyncio.run(worker.run_pipeline())

        status = worker.get_status()
        self.assertTrue(status["last_run_success"])
        self.assertEqual(status["dirty_datasets"], [])
        self.assertIsNone(status["last_error"])

    def test_graph_emptied_after_improve_keeps_dataset_dirty(self):
        class FakeDatasets:
            async def list_datasets(self):
                return [types.SimpleNamespace(id="dataset-id", name="default")]

            async def list_data(self, dataset_id):
                return [types.SimpleNamespace(id="data-id")]

        fake_cognee = types.ModuleType("cognee")
        graph_state = {"graph_data": ([("node-id", {"name": "node"})], [])}

        async def cognify(*, datasets, temporal_cognify, **kwargs):
            graph_state["graph_data"] = ([("node-id", {"name": "node"})], [])

        async def improve(*, dataset):
            graph_state["graph_data"] = ([], [])

        fake_cognee.cognify = cognify
        fake_cognee.improve = improve
        fake_cognee.datasets = FakeDatasets()
        _install_graph_engine_stub(
            is_empty=False,
            graph_data=graph_state["graph_data"],
        )

        background = _load_background_module(fake_cognee)

        async def get_graph_engine():
            class FakeGraphEngine:
                async def is_empty(self):
                    return not graph_state["graph_data"][0]

                async def get_graph_data(self):
                    return graph_state["graph_data"]

            return FakeGraphEngine()

        graph_module = sys.modules["cognee.infrastructure.databases.graph"]
        graph_module.get_graph_engine = get_graph_engine

        worker = background.CogneeBackgroundWorker()
        worker.mark_dirty("default")

        asyncio.run(worker.run_pipeline())

        status = worker.get_status()
        self.assertFalse(status["last_run_success"])
        self.assertEqual(status["dirty_datasets"], ["default"])
        self.assertIn("graph is still empty", status["last_error"])

    def test_readiness_uses_dataset_scoped_graph_context(self):
        class FakeDatasets:
            async def list_datasets(self):
                return [types.SimpleNamespace(id="dataset-id", name="default")]

            async def list_data(self, dataset_id):
                return [types.SimpleNamespace(id="data-id")]

        fake_cognee = types.ModuleType("cognee")

        async def cognify(*, datasets, temporal_cognify, **kwargs):
            return None

        async def improve(*, dataset):
            return None

        fake_cognee.cognify = cognify
        fake_cognee.improve = improve
        fake_cognee.datasets = FakeDatasets()
        _install_graph_engine_stub(is_empty=True)

        background = _load_background_module(fake_cognee)
        active_dataset = {"id": None}

        class DatasetContext:
            def __init__(self, dataset_id):
                self.dataset_id = dataset_id

            async def __aenter__(self):
                active_dataset["id"] = self.dataset_id
                return self

            async def __aexit__(self, exc_type, exc, tb):
                active_dataset["id"] = None
                return None

        async def get_graph_engine():
            class FakeGraphEngine:
                async def is_empty(self):
                    return active_dataset["id"] != "dataset-id"

                async def get_graph_data(self):
                    if active_dataset["id"] == "dataset-id":
                        return [("node-id", {"name": "node"})], []
                    return [], []

            return FakeGraphEngine()

        sys.modules[
            "cognee.context_global_variables"
        ].set_database_global_context_variables = (
            lambda dataset_id, owner_id: DatasetContext(dataset_id)
        )
        sys.modules["cognee.infrastructure.databases.graph"].get_graph_engine = (
            get_graph_engine
        )

        worker = background.CogneeBackgroundWorker()
        worker.mark_dirty("default")

        asyncio.run(worker.run_pipeline())

        status = worker.get_status()
        self.assertTrue(status["last_run_success"])
        self.assertEqual(status["dirty_datasets"], [])

    def test_readiness_does_not_clear_graph_engine_cache(self):
        class FakeDatasets:
            async def list_datasets(self):
                return [types.SimpleNamespace(id="dataset-id", name="default")]

            async def list_data(self, dataset_id):
                return [types.SimpleNamespace(id="data-id")]

        fake_cognee = types.ModuleType("cognee")

        async def cognify(*, datasets, temporal_cognify, **kwargs):
            return None

        async def improve(*, dataset):
            return None

        fake_cognee.cognify = cognify
        fake_cognee.improve = improve
        fake_cognee.datasets = FakeDatasets()
        _install_graph_engine_stub(is_empty=False)

        background = _load_background_module(fake_cognee)
        cache_cleared = {"value": False}

        async def get_graph_engine():
            class FakeGraphEngine:
                async def is_empty(self):
                    return False

                async def get_graph_data(self):
                    return [("node-id", {"name": "node"})], []

            return FakeGraphEngine()

        def _create_graph_engine():
            return None

        def cache_clear():
            cache_cleared["value"] = True

        _create_graph_engine.cache_clear = cache_clear

        graph_module = sys.modules["cognee.infrastructure.databases.graph"]
        graph_module.get_graph_engine = get_graph_engine
        graph_engine_module = sys.modules[
            "cognee.infrastructure.databases.graph.get_graph_engine"
        ]
        graph_engine_module._create_graph_engine = _create_graph_engine

        worker = background.CogneeBackgroundWorker()
        worker.mark_dirty("default")

        asyncio.run(worker.run_pipeline())

        status = worker.get_status()
        self.assertFalse(cache_cleared["value"])
        self.assertTrue(status["last_run_success"])
        self.assertEqual(status["dirty_datasets"], [])
        self.assertIsNone(status["last_error"])

    def test_empty_graph_with_unknown_dataset_data_keeps_dataset_dirty(self):
        class FakeDatasets:
            async def list_datasets(self):
                raise RuntimeError("database temporarily unavailable")

        fake_cognee = types.ModuleType("cognee")

        async def cognify(*, datasets, temporal_cognify, **kwargs):
            return None

        async def improve(*, dataset):
            return None

        fake_cognee.cognify = cognify
        fake_cognee.improve = improve
        fake_cognee.datasets = FakeDatasets()
        _install_graph_engine_stub(is_empty=True)

        background = _load_background_module(fake_cognee)
        worker = background.CogneeBackgroundWorker()
        worker.mark_dirty("default")

        asyncio.run(worker.run_pipeline())

        status = worker.get_status()
        self.assertFalse(status["last_run_success"])
        self.assertEqual(status["dirty_datasets"], ["default"])


if __name__ == "__main__":
    unittest.main()
