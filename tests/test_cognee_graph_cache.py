import asyncio
import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_cognee_graph_module(
    cache_clear_calls: list[str],
    graph_data_calls: list[str] | None = None,
):
    helpers = types.ModuleType("helpers")
    print_style = types.ModuleType("helpers.print_style")

    class PrintStyle:
        @staticmethod
        def warning(*args, **kwargs):
            pass

    print_style.PrintStyle = PrintStyle

    context_module = types.ModuleType("cognee.context_global_variables")
    graph_module = types.ModuleType("cognee.infrastructure.databases.graph")
    graph_engine_module = types.ModuleType(
        "cognee.infrastructure.databases.graph.get_graph_engine"
    )
    users_methods_module = types.ModuleType("cognee.modules.users.methods")

    class FakeDatabaseContext:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

    class FakeGraphEngine:
        async def is_empty(self):
            return False

        async def get_graph_data(self):
            if graph_data_calls is not None:
                graph_data_calls.append("get_graph_data")
            return [("node-id", {"name": "node"})], []

    async def get_graph_engine():
        return FakeGraphEngine()

    def _create_graph_engine():
        return None

    _create_graph_engine.cache_clear = lambda: cache_clear_calls.append("clear")

    context_module.set_database_global_context_variables = (
        lambda dataset_id, owner_id: FakeDatabaseContext()
    )
    graph_module.get_graph_engine = get_graph_engine
    graph_engine_module._create_graph_engine = _create_graph_engine
    users_methods_module.get_default_user = (
        lambda: _async_value(types.SimpleNamespace(id="owner-id"))
    )

    sys.modules.update(
        {
            "helpers": helpers,
            "helpers.print_style": print_style,
            "cognee": types.ModuleType("cognee"),
            "cognee.context_global_variables": context_module,
            "cognee.infrastructure": types.ModuleType("cognee.infrastructure"),
            "cognee.infrastructure.databases": types.ModuleType(
                "cognee.infrastructure.databases"
            ),
            "cognee.infrastructure.databases.graph": graph_module,
            "cognee.infrastructure.databases.graph.get_graph_engine": graph_engine_module,
            "cognee.modules": types.ModuleType("cognee.modules"),
            "cognee.modules.users": types.ModuleType("cognee.modules.users"),
            "cognee.modules.users.methods": users_methods_module,
        }
    )

    module_path = REPO_ROOT / "helpers" / "cognee_graph.py"
    spec = importlib.util.spec_from_file_location(
        "usr.plugins.memory_cognee.helpers.cognee_graph",
        module_path,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


async def _async_value(value):
    return value


class CogneeGraphCacheTest(unittest.TestCase):
    def tearDown(self):
        for name in list(sys.modules):
            if (
                name == "helpers"
                or name.startswith("helpers.")
                or name == "cognee"
                or name.startswith("cognee.")
                or name.startswith("usr.plugins.memory_cognee")
            ):
                sys.modules.pop(name, None)

    def test_dataset_graph_read_does_not_clear_cognee_engine_cache(self):
        cache_clear_calls: list[str] = []
        cognee_graph = _load_cognee_graph_module(cache_clear_calls)

        class FakeDatasets:
            async def list_datasets(self):
                return [types.SimpleNamespace(id="dataset-id", name="default")]

            async def list_data(self, dataset_id):
                return [types.SimpleNamespace(id="data-id")]

        fake_cognee = types.SimpleNamespace(datasets=FakeDatasets())

        results = asyncio.run(
            cognee_graph.read_dataset_graphs(
                fake_cognee,
                ["default"],
                skip_empty_data=False,
            )
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].nodes, [("node-id", {"name": "node"})])
        self.assertEqual(cache_clear_calls, [])

    def test_dataset_graph_read_can_skip_loading_full_graph_data(self):
        cache_clear_calls: list[str] = []
        graph_data_calls: list[str] = []
        cognee_graph = _load_cognee_graph_module(
            cache_clear_calls,
            graph_data_calls,
        )

        class FakeDatasets:
            async def list_datasets(self):
                return [types.SimpleNamespace(id="dataset-id", name="default")]

            async def list_data(self, dataset_id):
                return [types.SimpleNamespace(id="data-id")]

        fake_cognee = types.SimpleNamespace(datasets=FakeDatasets())

        results = asyncio.run(
            cognee_graph.read_dataset_graphs(
                fake_cognee,
                ["default"],
                skip_empty_data=False,
                include_graph_data=False,
            )
        )

        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].graph_empty)
        self.assertEqual(results[0].nodes, [])
        self.assertEqual(results[0].edges, [])
        self.assertEqual(graph_data_calls, [])

    def test_repair_unreadable_graph_renames_corrupt_kuzu_wal_and_retries(self):
        cache_clear_calls: list[str] = []
        cognee_graph = _load_cognee_graph_module(cache_clear_calls)

        with tempfile.TemporaryDirectory() as temp_dir:
            graph_path = os.path.join(temp_dir, "cognee_graph_kuzu")
            wal_path = f"{graph_path}.wal"
            Path(graph_path).write_bytes(b"graph")
            Path(wal_path).write_bytes(b"bad wal")

            config_module = types.ModuleType("cognee.infrastructure.databases.graph.config")

            class FakeGraphConfig:
                def model_dump(self):
                    return {
                        "graph_database_provider": "ladybug",
                        "graph_file_path": graph_path,
                    }

            config_module.get_graph_context_config = lambda: FakeGraphConfig()
            sys.modules["cognee.infrastructure.databases.graph.config"] = config_module

            graph_engine_module = sys.modules[
                "cognee.infrastructure.databases.graph.get_graph_engine"
            ]
            evictions = []
            graph_engine_module.evict_graph_engine = (
                lambda **kwargs: evictions.append(kwargs) or True
            )

            calls = {"count": 0}

            async def get_graph_engine():
                calls["count"] += 1
                if calls["count"] == 1:
                    raise RuntimeError(
                        "Runtime exception: Corrupted wal file. "
                        "Read out invalid WAL record type."
                    )

                class FakeGraphEngine:
                    async def is_empty(self):
                        return False

                    async def get_graph_data(self):
                        return [("node-id", {"name": "node"})], []

                return FakeGraphEngine()

            sys.modules["cognee.infrastructure.databases.graph"].get_graph_engine = (
                get_graph_engine
            )

            class FakeDatasets:
                async def list_datasets(self):
                    return [types.SimpleNamespace(id="dataset-id", name="default")]

                async def list_data(self, dataset_id):
                    return [types.SimpleNamespace(id="data-id")]

            fake_cognee = types.SimpleNamespace(datasets=FakeDatasets())

            results = asyncio.run(
                cognee_graph.read_dataset_graphs(
                    fake_cognee,
                    ["default"],
                    skip_empty_data=False,
                    repair_unreadable=True,
                    include_graph_data=False,
                )
            )

            self.assertEqual(calls["count"], 2)
            self.assertFalse(results[0].graph_empty)
            self.assertIsNone(results[0].error)
            self.assertFalse(os.path.exists(wal_path))
            self.assertEqual(len(list(Path(temp_dir).glob("*.wal.corrupt.*"))), 1)
            self.assertEqual(evictions[0]["graph_file_path"], graph_path)

    def test_repair_unreadable_graph_falls_back_to_global_kuzu_wal(self):
        cache_clear_calls: list[str] = []
        cognee_graph = _load_cognee_graph_module(cache_clear_calls)

        with tempfile.TemporaryDirectory() as temp_dir:
            context_graph_path = os.path.join(temp_dir, "dataset", "graph.pkl")
            global_db_dir = os.path.join(temp_dir, "system", "databases")
            global_graph_path = os.path.join(global_db_dir, "cognee_graph_kuzu")
            global_wal_path = f"{global_graph_path}.wal"
            os.makedirs(os.path.dirname(context_graph_path), exist_ok=True)
            os.makedirs(global_db_dir, exist_ok=True)
            Path(context_graph_path).write_bytes(b"dataset graph")
            Path(global_graph_path).write_bytes(b"global graph")
            Path(global_wal_path).write_bytes(b"bad wal")

            old_system_root = os.environ.get("SYSTEM_ROOT_DIRECTORY")
            os.environ["SYSTEM_ROOT_DIRECTORY"] = os.path.join(temp_dir, "system")
            try:
                config_module = types.ModuleType(
                    "cognee.infrastructure.databases.graph.config"
                )

                class FakeGraphConfig:
                    def model_dump(self):
                        return {
                            "graph_database_provider": "ladybug",
                            "graph_file_path": context_graph_path,
                        }

                config_module.get_graph_context_config = lambda: FakeGraphConfig()
                sys.modules["cognee.infrastructure.databases.graph.config"] = (
                    config_module
                )

                graph_engine_module = sys.modules[
                    "cognee.infrastructure.databases.graph.get_graph_engine"
                ]
                evictions = []
                graph_engine_module.evict_graph_engine = (
                    lambda **kwargs: evictions.append(kwargs) or True
                )

                calls = {"count": 0}

                async def get_graph_engine():
                    calls["count"] += 1
                    if calls["count"] == 1:
                        raise RuntimeError(
                            "Runtime exception: Corrupted wal file. "
                            "Read out invalid WAL record type."
                        )

                    class FakeGraphEngine:
                        async def is_empty(self):
                            return False

                        async def get_graph_data(self):
                            return [("node-id", {"name": "node"})], []

                    return FakeGraphEngine()

                sys.modules["cognee.infrastructure.databases.graph"].get_graph_engine = (
                    get_graph_engine
                )

                class FakeDatasets:
                    async def list_datasets(self):
                        return [types.SimpleNamespace(id="dataset-id", name="default")]

                    async def list_data(self, dataset_id):
                        return [types.SimpleNamespace(id="data-id")]

                fake_cognee = types.SimpleNamespace(datasets=FakeDatasets())

                results = asyncio.run(
                    cognee_graph.read_dataset_graphs(
                        fake_cognee,
                        ["default"],
                        skip_empty_data=False,
                        repair_unreadable=True,
                        include_graph_data=False,
                    )
                )
            finally:
                if old_system_root is None:
                    os.environ.pop("SYSTEM_ROOT_DIRECTORY", None)
                else:
                    os.environ["SYSTEM_ROOT_DIRECTORY"] = old_system_root

            self.assertEqual(calls["count"], 2)
            self.assertFalse(results[0].graph_empty)
            self.assertIsNone(results[0].error)
            self.assertFalse(os.path.exists(global_wal_path))
            self.assertEqual(len(list(Path(global_db_dir).glob("*.wal.corrupt.*"))), 1)
            self.assertEqual(evictions[0]["graph_file_path"], context_graph_path)
            self.assertEqual(evictions[1]["graph_file_path"], global_graph_path)

    def test_repair_unreadable_graph_quarantines_global_graph_when_wal_missing(self):
        cache_clear_calls: list[str] = []
        cognee_graph = _load_cognee_graph_module(cache_clear_calls)

        with tempfile.TemporaryDirectory() as temp_dir:
            context_graph_path = os.path.join(temp_dir, "dataset", "graph.pkl")
            global_db_dir = os.path.join(temp_dir, "system", "databases")
            global_graph_path = os.path.join(global_db_dir, "cognee_graph_kuzu")
            os.makedirs(os.path.dirname(context_graph_path), exist_ok=True)
            os.makedirs(global_db_dir, exist_ok=True)
            Path(context_graph_path).write_bytes(b"dataset graph")
            Path(global_graph_path).write_bytes(b"global graph")

            old_system_root = os.environ.get("SYSTEM_ROOT_DIRECTORY")
            os.environ["SYSTEM_ROOT_DIRECTORY"] = os.path.join(temp_dir, "system")
            try:
                config_module = types.ModuleType(
                    "cognee.infrastructure.databases.graph.config"
                )

                class FakeGraphConfig:
                    def model_dump(self):
                        return {
                            "graph_database_provider": "ladybug",
                            "graph_file_path": context_graph_path,
                        }

                config_module.get_graph_context_config = lambda: FakeGraphConfig()
                sys.modules["cognee.infrastructure.databases.graph.config"] = (
                    config_module
                )

                graph_engine_module = sys.modules[
                    "cognee.infrastructure.databases.graph.get_graph_engine"
                ]
                evictions = []
                graph_engine_module.evict_graph_engine = (
                    lambda **kwargs: evictions.append(kwargs) or True
                )

                calls = {"count": 0}

                async def get_graph_engine():
                    calls["count"] += 1
                    if calls["count"] == 1:
                        raise RuntimeError(
                            "Runtime exception: Corrupted wal file. "
                            "Read out invalid WAL record type."
                        )

                    class FakeGraphEngine:
                        async def is_empty(self):
                            return True

                    return FakeGraphEngine()

                sys.modules["cognee.infrastructure.databases.graph"].get_graph_engine = (
                    get_graph_engine
                )

                class FakeDatasets:
                    async def list_datasets(self):
                        return [types.SimpleNamespace(id="dataset-id", name="default")]

                    async def list_data(self, dataset_id):
                        return [types.SimpleNamespace(id="data-id")]

                fake_cognee = types.SimpleNamespace(datasets=FakeDatasets())

                results = asyncio.run(
                    cognee_graph.read_dataset_graphs(
                        fake_cognee,
                        ["default"],
                        skip_empty_data=False,
                        repair_unreadable=True,
                        include_graph_data=False,
                    )
                )
            finally:
                if old_system_root is None:
                    os.environ.pop("SYSTEM_ROOT_DIRECTORY", None)
                else:
                    os.environ["SYSTEM_ROOT_DIRECTORY"] = old_system_root

            self.assertEqual(calls["count"], 2)
            self.assertTrue(results[0].graph_empty)
            self.assertFalse(os.path.exists(global_graph_path))
            self.assertTrue(os.path.exists(context_graph_path))
            self.assertEqual(len(list(Path(global_db_dir).glob("*.corrupt.*"))), 1)
            self.assertEqual(evictions[0]["graph_file_path"], context_graph_path)
            self.assertEqual(evictions[1]["graph_file_path"], global_graph_path)

    def test_repair_unreadable_graph_quarantines_leftover_kuzu_sidecars(self):
        cache_clear_calls: list[str] = []
        cognee_graph = _load_cognee_graph_module(cache_clear_calls)

        with tempfile.TemporaryDirectory() as temp_dir:
            context_graph_path = os.path.join(temp_dir, "dataset", "graph.pkl")
            os.makedirs(os.path.dirname(context_graph_path), exist_ok=True)
            Path(f"{context_graph_path}.shadow").write_bytes(b"")
            Path(f"{context_graph_path}.wal.checkpoint").write_bytes(b"checkpoint")

            config_module = types.ModuleType("cognee.infrastructure.databases.graph.config")

            class FakeGraphConfig:
                def model_dump(self):
                    return {
                        "graph_database_provider": "ladybug",
                        "graph_file_path": context_graph_path,
                    }

            config_module.get_graph_context_config = lambda: FakeGraphConfig()
            sys.modules["cognee.infrastructure.databases.graph.config"] = config_module

            graph_engine_module = sys.modules[
                "cognee.infrastructure.databases.graph.get_graph_engine"
            ]
            evictions = []
            graph_engine_module.evict_graph_engine = (
                lambda **kwargs: evictions.append(kwargs) or True
            )

            calls = {"count": 0}

            async def get_graph_engine():
                calls["count"] += 1
                if calls["count"] == 1:
                    raise RuntimeError(
                        "Runtime exception: Corrupted wal file. "
                        "Read out invalid WAL record type."
                    )

                class FakeGraphEngine:
                    async def is_empty(self):
                        return True

                return FakeGraphEngine()

            sys.modules["cognee.infrastructure.databases.graph"].get_graph_engine = (
                get_graph_engine
            )

            class FakeDatasets:
                async def list_datasets(self):
                    return [types.SimpleNamespace(id="dataset-id", name="default")]

                async def list_data(self, dataset_id):
                    return [types.SimpleNamespace(id="data-id")]

            fake_cognee = types.SimpleNamespace(datasets=FakeDatasets())

            results = asyncio.run(
                cognee_graph.read_dataset_graphs(
                    fake_cognee,
                    ["default"],
                    skip_empty_data=False,
                    repair_unreadable=True,
                    include_graph_data=False,
                )
            )

            self.assertEqual(calls["count"], 2)
            self.assertTrue(results[0].graph_empty)
            self.assertFalse(os.path.exists(f"{context_graph_path}.shadow"))
            self.assertFalse(os.path.exists(f"{context_graph_path}.wal.checkpoint"))
            self.assertEqual(
                len(list(Path(os.path.dirname(context_graph_path)).glob("*.corrupt.*"))),
                2,
            )
            self.assertEqual(evictions[0]["graph_file_path"], context_graph_path)

    def test_repair_unreadable_graph_quarantines_ladybug_shadow_temp_file(self):
        cache_clear_calls: list[str] = []
        cognee_graph = _load_cognee_graph_module(cache_clear_calls)

        with tempfile.TemporaryDirectory() as temp_dir:
            context_graph_path = os.path.join(temp_dir, "dataset", "eba921c6.pkl")
            os.makedirs(os.path.dirname(context_graph_path), exist_ok=True)
            shadow_path = f"{context_graph_path}.shadow"
            Path(shadow_path).write_bytes(b"stale temp")

            config_module = types.ModuleType("cognee.infrastructure.databases.graph.config")

            class FakeGraphConfig:
                def model_dump(self):
                    return {
                        "graph_database_provider": "ladybug",
                        "graph_file_path": context_graph_path,
                    }

            config_module.get_graph_context_config = lambda: FakeGraphConfig()
            sys.modules["cognee.infrastructure.databases.graph.config"] = config_module

            graph_engine_module = sys.modules[
                "cognee.infrastructure.databases.graph.get_graph_engine"
            ]
            evictions = []
            graph_engine_module.evict_graph_engine = (
                lambda **kwargs: evictions.append(kwargs) or True
            )

            calls = {"count": 0}

            async def get_graph_engine():
                calls["count"] += 1
                if calls["count"] == 1:
                    raise RuntimeError(
                        "Runtime exception: Database ID for temporary file "
                        f"'{shadow_path}' does not match the current database. "
                        "This file may have been left behind from a previous "
                        "database with the same name."
                    )

                class FakeGraphEngine:
                    async def is_empty(self):
                        return True

                return FakeGraphEngine()

            sys.modules["cognee.infrastructure.databases.graph"].get_graph_engine = (
                get_graph_engine
            )

            class FakeDatasets:
                async def list_datasets(self):
                    return [types.SimpleNamespace(id="dataset-id", name="default")]

                async def list_data(self, dataset_id):
                    return [types.SimpleNamespace(id="data-id")]

            fake_cognee = types.SimpleNamespace(datasets=FakeDatasets())

            results = asyncio.run(
                cognee_graph.read_dataset_graphs(
                    fake_cognee,
                    ["default"],
                    skip_empty_data=False,
                    repair_unreadable=True,
                    include_graph_data=False,
                )
            )

            self.assertEqual(calls["count"], 2)
            self.assertTrue(results[0].graph_empty)
            self.assertFalse(os.path.exists(shadow_path))
            self.assertEqual(
                len(list(Path(os.path.dirname(context_graph_path)).glob("*.corrupt.*"))),
                1,
            )
            self.assertEqual(evictions[0]["graph_file_path"], context_graph_path)

    def test_repair_unreadable_graph_quarantines_missing_ladybug_shadow_store(self):
        cache_clear_calls: list[str] = []
        cognee_graph = _load_cognee_graph_module(cache_clear_calls)

        with tempfile.TemporaryDirectory() as temp_dir:
            context_graph_path = os.path.join(temp_dir, "dataset", "eba921c6.pkl")
            os.makedirs(os.path.dirname(context_graph_path), exist_ok=True)
            shadow_path = f"{context_graph_path}.shadow"
            Path(context_graph_path).write_bytes(b"graph")
            Path(f"{context_graph_path}.wal.checkpoint").write_bytes(b"checkpoint")

            config_module = types.ModuleType("cognee.infrastructure.databases.graph.config")

            class FakeGraphConfig:
                def model_dump(self):
                    return {
                        "graph_database_provider": "ladybug",
                        "graph_file_path": context_graph_path,
                    }

            config_module.get_graph_context_config = lambda: FakeGraphConfig()
            sys.modules["cognee.infrastructure.databases.graph.config"] = config_module

            graph_engine_module = sys.modules[
                "cognee.infrastructure.databases.graph.get_graph_engine"
            ]
            evictions = []
            graph_engine_module.evict_graph_engine = (
                lambda **kwargs: evictions.append(kwargs) or True
            )

            calls = {"count": 0}

            async def get_graph_engine():
                calls["count"] += 1
                if calls["count"] == 1:
                    raise RuntimeError(
                        "Runtime exception: IO exception: Cannot open file "
                        f"{shadow_path}: No such file or directory"
                    )

                class FakeGraphEngine:
                    async def is_empty(self):
                        return True

                return FakeGraphEngine()

            sys.modules["cognee.infrastructure.databases.graph"].get_graph_engine = (
                get_graph_engine
            )

            class FakeDatasets:
                async def list_datasets(self):
                    return [types.SimpleNamespace(id="dataset-id", name="default")]

                async def list_data(self, dataset_id):
                    return [types.SimpleNamespace(id="data-id")]

            fake_cognee = types.SimpleNamespace(datasets=FakeDatasets())

            results = asyncio.run(
                cognee_graph.read_dataset_graphs(
                    fake_cognee,
                    ["default"],
                    skip_empty_data=False,
                    repair_unreadable=True,
                    include_graph_data=False,
                )
            )

            self.assertEqual(calls["count"], 2)
            self.assertTrue(results[0].graph_empty)
            self.assertFalse(os.path.exists(context_graph_path))
            self.assertFalse(os.path.exists(f"{context_graph_path}.wal.checkpoint"))
            self.assertFalse(os.path.exists(shadow_path))
            self.assertEqual(
                len(list(Path(os.path.dirname(context_graph_path)).glob("*.corrupt.*"))),
                2,
            )
            self.assertEqual(evictions[0]["graph_file_path"], context_graph_path)


if __name__ == "__main__":
    unittest.main()
