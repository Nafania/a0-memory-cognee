import asyncio
import importlib.util
import sys
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

            async def cognify(self, *, datasets, temporal_cognify):
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

        async def cognify(*, datasets, temporal_cognify):
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

        async def cognify(*, datasets, temporal_cognify):
            cognify_calls.append(list(datasets))

        async def improve(*, dataset):
            return None

        fake_cognee.cognify = cognify
        fake_cognee.improve = improve
        _install_graph_engine_stub(is_empty=False)

        background = _load_background_module(fake_cognee)
        worker = background.CogneeBackgroundWorker()
        worker.mark_dirty("default")
        worker.mark_dirty("projects_alpha")

        asyncio.run(worker.run_pipeline())

        self.assertEqual(cognify_calls, [["default"], ["projects_alpha"]])
        status = worker.get_status()
        self.assertTrue(status["last_run_success"])
        self.assertEqual(status["dirty_datasets"], [])

    def test_empty_search_can_nudge_rebuild_before_first_successful_run(self):
        fake_cognee = types.ModuleType("cognee")
        background = _load_background_module(fake_cognee)
        worker = background.CogneeBackgroundWorker()

        nudged = worker.nudge_rebuild_if_unready(["default"], "empty graph")

        self.assertTrue(nudged)
        self.assertEqual(worker.get_status()["dirty_datasets"], ["default"])

    def test_dirty_dataset_blocks_search_until_successful_rebuild(self):
        class FakeDatasets:
            async def list_datasets(self):
                return [types.SimpleNamespace(id="dataset-id", name="default")]

            async def list_data(self, dataset_id):
                return [types.SimpleNamespace(id="data-id")]

        fake_cognee = types.ModuleType("cognee")

        async def cognify(*, datasets, temporal_cognify):
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

        async def cognify(*, datasets, temporal_cognify):
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

    def test_hung_cognify_times_out_and_unblocks_rebuild_state(self):
        class FakeDatasets:
            async def list_datasets(self):
                return [types.SimpleNamespace(id="dataset-id", name="default")]

            async def list_data(self, dataset_id):
                return [types.SimpleNamespace(id="data-id")]

        fake_cognee = types.ModuleType("cognee")

        async def cognify(*, datasets, temporal_cognify):
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

    def test_cancelled_rebuild_marks_state_failed_before_exit(self):
        fake_cognee = types.ModuleType("cognee")

        async def cognify(*, datasets, temporal_cognify):
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

    def test_stale_rebuilding_state_is_marked_failed_for_retry(self):
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
        self.assertEqual(scheduled_runs, [])

    def test_start_is_idempotent(self):
        fake_cognee = types.ModuleType("cognee")
        background = _load_background_module(fake_cognee)
        worker = background.CogneeBackgroundWorker()
        deferred_task = sys.modules["helpers.defer"].DeferredTask

        first = worker.start()
        second = worker.start()

        self.assertIs(first, second)
        self.assertEqual(deferred_task.start_count, 1)

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

        async def cognify(*, datasets, temporal_cognify):
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

    def test_graph_without_nodes_keeps_dataset_dirty(self):
        class FakeDatasets:
            async def list_datasets(self):
                return [types.SimpleNamespace(id="dataset-id", name="default")]

            async def list_data(self, dataset_id):
                return [types.SimpleNamespace(id="data-id")]

        fake_cognee = types.ModuleType("cognee")

        async def cognify(*, datasets, temporal_cognify):
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
        self.assertFalse(status["last_run_success"])
        self.assertEqual(status["dirty_datasets"], ["default"])
        self.assertIn("graph has no readable nodes", status["last_error"])

    def test_graph_emptied_after_improve_keeps_dataset_dirty(self):
        class FakeDatasets:
            async def list_datasets(self):
                return [types.SimpleNamespace(id="dataset-id", name="default")]

            async def list_data(self, dataset_id):
                return [types.SimpleNamespace(id="data-id")]

        fake_cognee = types.ModuleType("cognee")
        graph_state = {"graph_data": ([("node-id", {"name": "node"})], [])}

        async def cognify(*, datasets, temporal_cognify):
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

        async def cognify(*, datasets, temporal_cognify):
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

        async def cognify(*, datasets, temporal_cognify):
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

        async def cognify(*, datasets, temporal_cognify):
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
