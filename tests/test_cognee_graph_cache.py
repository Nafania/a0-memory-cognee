import asyncio
import importlib.util
import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_cognee_graph_module(cache_clear_calls: list[str]):
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


if __name__ == "__main__":
    unittest.main()
