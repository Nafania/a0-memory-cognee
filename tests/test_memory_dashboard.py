import asyncio
import importlib.util
import sys
import types
import unittest
from pathlib import Path

from langchain_core.documents import Document


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_dashboard_module(captured: dict, *, node_count: int = 2):
    helpers_api = types.ModuleType("helpers.api")

    class ApiHandler:
        pass

    helpers_api.ApiHandler = ApiHandler
    helpers_api.Request = dict
    helpers_api.Response = dict

    files = types.ModuleType("helpers.files")
    print_style = types.ModuleType("helpers.print_style")

    class PrintStyle:
        @staticmethod
        def error(*args, **kwargs):
            pass

    print_style.PrintStyle = PrintStyle

    package_names = [
        "usr",
        "usr.plugins",
        "usr.plugins.memory_cognee",
        "usr.plugins.memory_cognee.helpers",
        "usr.plugins.memory_cognee.api",
    ]
    for name in package_names:
        package = types.ModuleType(name)
        package.__path__ = [str(REPO_ROOT / "helpers")] if name.endswith(".helpers") else []
        sys.modules[name] = package

    memory = types.ModuleType("usr.plugins.memory_cognee.helpers.memory")

    class Memory:
        def __init__(self, dataset_name, memory_subdir="default"):
            self.dataset_name = dataset_name
            self.memory_subdir = memory_subdir

        @staticmethod
        async def get_by_subdir(memory_subdir, preload_knowledge=False):
            return Memory(
                memory_subdir.replace("/", "_").replace(" ", "_").lower(),
                memory_subdir,
            )

        async def search_similarity_threshold(
            self,
            *,
            query,
            limit,
            threshold,
            filter,
            include_default,
        ):
            captured["search_threshold"] = threshold
            return [
                Document(
                    page_content="matched memory",
                    metadata={"id": "memory-id", "area": "main"},
                )
            ]

        async def update_documents(self, docs):
            captured["updated_metadata"] = dict(docs[0].metadata)
            captured["updated_content"] = docs[0].page_content
            return ["new-id"]

    memory.Memory = Memory
    memory.get_existing_memory_subdirs = lambda: ["default", "projects/demo"]
    memory.get_context_memory_subdir = lambda context: "default"
    def read_data_item_content(item):
        raise AssertionError("sync content reader should not be used")

    async def read_data_item_content_async(item):
        captured["async_content_reader_used"] = True
        return "async memory"

    memory.read_data_item_content = read_data_item_content
    memory.read_data_item_content_async = read_data_item_content_async
    memory.content_hash_id = lambda content, dataset_name="": f"hash-{dataset_name}"
    memory.parse_node_set_area = lambda raw: "main"
    memory.hydrate_metadata = (
        lambda memory_subdir, dataset_name, content, metadata: metadata
    )

    graph_module = types.ModuleType("usr.plugins.memory_cognee.helpers.cognee_graph")

    async def read_dataset_graphs(cognee, dataset_names=None, **kwargs):
        captured["dataset_names"] = dataset_names
        nodes = [
            (f"node-{idx}", {"name": f"Node {idx}", "type": "entity"})
            for idx in range(node_count)
        ]
        return [
            types.SimpleNamespace(
                dataset_id="dataset-id",
                dataset_name=(dataset_names or ["default"])[0],
                nodes=nodes,
                edges=[],
                error=None,
            )
        ]

    graph_module.read_dataset_graphs = read_dataset_graphs

    cognee_ops = types.ModuleType("usr.plugins.memory_cognee.helpers.cognee_ops")

    async def run_cognee_operation(label, operation, *args, **kwargs):
        captured.setdefault("operation_labels", []).append(label)
        op_kwargs = dict(kwargs)
        op_kwargs.pop("timeout", None)
        op_kwargs.pop("operation_timeout", None)
        op_kwargs.pop("a0_agent", None)
        result = operation(*args, **op_kwargs)
        if asyncio.iscoroutine(result):
            return await result
        return result

    cognee_ops.run_cognee_operation = run_cognee_operation

    class FakeDatasets:
        async def list_datasets(self):
            return [types.SimpleNamespace(id="dataset-id", name="default")]

        async def list_data(self, dataset_id):
            return [
                types.SimpleNamespace(
                    id="data-id",
                    node_set='["main"]',
                    created_at="2026-05-24 00:00:00",
                )
            ]

    fake_cognee = types.ModuleType("cognee")
    fake_cognee.datasets = FakeDatasets()

    sys.modules.update(
        {
            "helpers.api": helpers_api,
            "helpers.files": files,
            "helpers.print_style": print_style,
            "agent": types.SimpleNamespace(AgentContext=object),
            "cognee": fake_cognee,
            "usr.plugins.memory_cognee.helpers.memory": memory,
            "usr.plugins.memory_cognee.helpers.cognee_graph": graph_module,
            "usr.plugins.memory_cognee.helpers.cognee_ops": cognee_ops,
        }
    )

    module_path = REPO_ROOT / "api" / "memory_dashboard.py"
    spec = importlib.util.spec_from_file_location(
        "usr.plugins.memory_cognee.api.memory_dashboard",
        module_path,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class MemoryDashboardTest(unittest.TestCase):
    def tearDown(self):
        for name in list(sys.modules):
            if (
                name == "agent"
                or name == "cognee"
                or name.startswith("helpers.")
                or name.startswith("usr.plugins.memory_cognee")
            ):
                sys.modules.pop(name, None)

    def test_graph_data_uses_memory_subdir_dataset_not_client_dataset_filter(self):
        captured = {}
        module = _load_dashboard_module(captured)
        handler = module.MemoryDashboard()

        result = asyncio.run(
            handler._get_graph_data(
                {
                    "memory_subdir": "projects/Demo Project",
                    "datasets": ["default", "other"],
                }
            )
        )

        self.assertTrue(result["success"])
        self.assertEqual(captured["dataset_names"], ["projects_demo_project"])

    def test_graph_data_clamps_node_limit_server_side(self):
        captured = {}
        module = _load_dashboard_module(captured, node_count=1005)
        handler = module.MemoryDashboard()

        result = asyncio.run(
            handler._get_graph_data(
                {
                    "memory_subdir": "default",
                    "node_limit": 99999,
                }
            )
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["node_count"], 1000)
        self.assertEqual(result["total_node_count"], 1005)
        self.assertTrue(result["truncated"])

    def test_search_memories_passes_default_threshold(self):
        captured = {}
        module = _load_dashboard_module(captured)
        handler = module.MemoryDashboard()

        result = asyncio.run(
            handler._search_memories(
                {
                    "memory_subdir": "default",
                    "search": "needle",
                    "limit": 5,
                }
            )
        )

        self.assertTrue(result["success"])
        self.assertEqual(captured["search_threshold"], 0.7)
        self.assertEqual(result["memories"][0]["content_full"], "matched memory")

    def test_update_memory_uses_original_id_as_immutable_target(self):
        captured = {}
        module = _load_dashboard_module(captured)
        handler = module.MemoryDashboard()

        result = asyncio.run(
            handler._update_memory(
                {
                    "memory_subdir": "default",
                    "original": {
                        "id": "original-id",
                        "metadata": {"id": "original-id", "area": "main"},
                    },
                    "edited": {
                        "content_full": "edited memory",
                        "metadata": {"id": "attacker-id", "area": "main"},
                    },
                }
            )
        )

        self.assertTrue(result["success"])
        self.assertEqual(captured["updated_content"], "edited memory")
        self.assertEqual(captured["updated_metadata"]["id"], "original-id")

    def test_list_memories_uses_async_content_reader(self):
        captured = {}
        module = _load_dashboard_module(captured)
        handler = module.MemoryDashboard()

        result = asyncio.run(
            handler._search_memories(
                {
                    "memory_subdir": "default",
                    "limit": 10,
                }
            )
        )

        self.assertTrue(result["success"])
        self.assertTrue(captured["async_content_reader_used"])
        self.assertEqual(result["memories"][0]["content_full"], "async memory")


if __name__ == "__main__":
    unittest.main()
