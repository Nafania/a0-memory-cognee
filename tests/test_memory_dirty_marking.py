import asyncio
import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeDirtyWorker:
    def __init__(self):
        self.dirty: list[str] = []
        self.block_reason: str | None = None

    def mark_dirty(self, dataset_name: str):
        self.dirty.append(dataset_name)

    def get_search_block_reason(self, datasets):
        return self.block_reason

    def nudge_rebuild_if_unready(self, datasets, reason=""):
        return False


def _load_memory_module(tmp_dir: str, worker: FakeDirtyWorker):
    helpers = types.ModuleType("helpers")

    files = types.ModuleType("helpers.files")
    files.get_abs_path = lambda *parts: os.path.join(tmp_dir, *parts)

    projects = types.ModuleType("helpers.projects")
    projects.get_context_project_name = lambda context: ""
    projects.load_project_header = lambda project_name: {}

    plugins = types.ModuleType("helpers.plugins")
    plugins.get_plugin_config = lambda name, agent=None: {}

    print_style = types.ModuleType("helpers.print_style")

    class PrintStyle:
        def __init__(self, *args, **kwargs):
            pass

        @staticmethod
        def error(*args, **kwargs):
            pass

        @staticmethod
        def warning(*args, **kwargs):
            pass

        def print(self, *args, **kwargs):
            pass

    print_style.PrintStyle = PrintStyle

    log = types.ModuleType("helpers.log")
    log.Log = object
    log.LogItem = object

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

    knowledge_import = types.ModuleType("usr.plugins.memory_cognee.helpers.knowledge_import")

    def load_knowledge(log_item, path, index, metadata, **kwargs):
        if "doc.md" not in index:
            index["doc.md"] = {
                "state": "changed",
                "documents": [types.SimpleNamespace(page_content="knowledge text")],
                "metadata": metadata,
            }
        return index

    knowledge_import.load_knowledge = load_knowledge
    knowledge_import.KnowledgeImport = dict

    cognee_init = types.ModuleType("usr.plugins.memory_cognee.helpers.cognee_init")
    cognee_init.get_cognee_setting = lambda key, default=None: default
    cognee_init.ensure_cognee_llm_config_current = lambda agent=None: None

    background = types.ModuleType("usr.plugins.memory_cognee.helpers.cognee_background")

    class CogneeBackgroundWorker:
        @staticmethod
        def get_instance():
            return worker

    background.CogneeBackgroundWorker = CogneeBackgroundWorker

    sys.modules.update(
        {
            "helpers": helpers,
            "helpers.files": files,
            "helpers.projects": projects,
            "helpers.plugins": plugins,
            "helpers.print_style": print_style,
            "helpers.log": log,
            "agent": types.SimpleNamespace(Agent=object, AgentContext=object),
            "models": types.ModuleType("models"),
            "usr.plugins.memory_cognee.helpers.knowledge_import": knowledge_import,
            "usr.plugins.memory_cognee.helpers.cognee_init": cognee_init,
            "usr.plugins.memory_cognee.helpers.cognee_background": background,
        }
    )

    module_path = REPO_ROOT / "helpers" / "memory.py"
    spec = importlib.util.spec_from_file_location(
        "usr.plugins.memory_cognee.helpers.memory",
        module_path,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class MemoryDirtyMarkingTest(unittest.TestCase):
    def tearDown(self):
        for name in list(sys.modules):
            if (
                name == "helpers"
                or name.startswith("helpers.")
                or name == "agent"
                or name == "models"
                or name.startswith("usr.plugins.memory_cognee")
            ):
                sys.modules.pop(name, None)

    def test_delete_marks_dataset_dirty(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = FakeDirtyWorker()
            memory_module = _load_memory_module(tmp_dir, worker)
            memory_module._get_cognee = lambda: (types.SimpleNamespace(), None)

            async def find_dataset(dataset_name):
                return types.SimpleNamespace(id="dataset-id", name=dataset_name)

            async def try_delete_direct(cognee, target, data_id, agent=None):
                return True

            memory_module._find_dataset = find_dataset
            memory_module._try_delete_direct = try_delete_direct

            memory = memory_module.Memory("default", "default")
            removed = asyncio.run(memory.delete_documents_by_ids(["data-id"]))

            self.assertEqual(len(removed), 1)
            self.assertEqual(worker.dirty, ["default"])

    def test_preload_knowledge_marks_dataset_dirty_after_import(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = FakeDirtyWorker()
            memory_module = _load_memory_module(tmp_dir, worker)

            class FakeCognee:
                def __init__(self):
                    self.added: list[tuple[str, str, list[str]]] = []
                    self.datasets = types.SimpleNamespace(list_datasets=self.list_datasets)

                async def list_datasets(self):
                    return []

                async def add(self, content, *, dataset_name, node_set):
                    self.added.append((content, dataset_name, node_set))

            fake_cognee = FakeCognee()
            memory_module._get_cognee = lambda: (fake_cognee, None)

            memory = memory_module.Memory("default", "default")
            asyncio.run(memory.preload_knowledge(None, ["default"], "default"))

            self.assertEqual(fake_cognee.added, [("knowledge text", "default", ["main"])])
            self.assertEqual(worker.dirty, ["default"])

    def test_get_can_skip_preload_for_read_only_paths(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = FakeDirtyWorker()
            memory_module = _load_memory_module(tmp_dir, worker)

            memory_module._get_cognee = lambda: (_ for _ in ()).throw(
                AssertionError("read-only Memory.get must not preload knowledge")
            )
            agent = types.SimpleNamespace(
                config=types.SimpleNamespace(knowledge_subdirs=["default"]),
                context=types.SimpleNamespace(
                    log=types.SimpleNamespace(log=lambda **kwargs: object()),
                    streaming_agent=None,
                    agent0=None,
                ),
            )

            mem = asyncio.run(memory_module.Memory.get(agent, preload_knowledge=False))

            self.assertEqual(mem.dataset_name, "default")
            self.assertEqual(worker.dirty, [])
            self.assertEqual(memory_module.Memory._initialized_subdirs, set())

    def test_search_skips_cognee_when_rebuild_not_ready(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = FakeDirtyWorker()
            worker.block_reason = "Cognee memory graph rebuild pending for dataset(s): ['default']"
            memory_module = _load_memory_module(tmp_dir, worker)
            memory_module._get_cognee = lambda: (_ for _ in ()).throw(
                AssertionError("cognee should not be loaded while rebuild blocks search")
            )

            memory = memory_module.Memory("default", "default")
            docs = asyncio.run(
                memory.search_similarity_threshold(
                    query="test",
                    limit=5,
                    threshold=0.7,
                )
            )

            self.assertEqual(docs, [])

    def test_search_can_raise_when_rebuild_not_ready(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = FakeDirtyWorker()
            worker.block_reason = "Cognee memory graph rebuild pending for dataset(s): ['default']"
            memory_module = _load_memory_module(tmp_dir, worker)
            memory_module._get_cognee = lambda: (_ for _ in ()).throw(
                AssertionError("cognee should not be loaded while rebuild blocks search")
            )

            memory = memory_module.Memory("default", "default")
            with self.assertRaisesRegex(
                memory_module.SearchUnavailable,
                "Cognee memory graph rebuild pending",
            ):
                asyncio.run(
                    memory.search_similarity_threshold(
                        query="test",
                        limit=5,
                        threshold=0.7,
                        raise_unavailable=True,
                    )
                )

    def test_reload_uses_public_cognee_init_reset(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = FakeDirtyWorker()
            memory_module = _load_memory_module(tmp_dir, worker)
            cognee_init = sys.modules["usr.plugins.memory_cognee.helpers.cognee_init"]
            calls = []
            cognee_init.reset_cognee_init_state = lambda: calls.append("reset")
            cognee_init.configure_cognee = lambda: calls.append("configure")

            memory_module.Memory._initialized_subdirs.add("default")
            memory_module.Memory._datasets_cache["default"] = ["default"]

            memory_module.reload()

            self.assertEqual(calls, ["reset", "configure"])
            self.assertEqual(memory_module.Memory._initialized_subdirs, set())
            self.assertEqual(memory_module.Memory._datasets_cache, {})

    def test_similarity_search_keeps_verbose_result_shape_for_metadata(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = FakeDirtyWorker()
            memory_module = _load_memory_module(tmp_dir, worker)

            class NodeSet:
                pass

            node_set_module = types.ModuleType("cognee.modules.engine.models.node_set")
            node_set_module.NodeSet = NodeSet
            sys.modules["cognee.modules.engine.models.node_set"] = node_set_module

            class FakeCognee:
                def __init__(self):
                    self.search_calls = []

                async def search(self, **kwargs):
                    self.search_calls.append(kwargs)
                    return [{"search_result": "stored memory", "dataset_name": "default"}]

            fake_cognee = FakeCognee()
            memory_module._get_cognee = lambda: (fake_cognee, None)

            memory = memory_module.Memory("default", "default")
            docs = asyncio.run(
                memory.search_similarity_threshold(
                    query="test",
                    limit=5,
                    threshold=0.7,
                )
            )

            self.assertEqual([doc.page_content for doc in docs], ["stored memory"])
            self.assertEqual(len(fake_cognee.search_calls), 1)
            self.assertIs(fake_cognee.search_calls[0]["verbose"], True)

    def test_similarity_search_uses_short_user_path_timeout(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = FakeDirtyWorker()
            memory_module = _load_memory_module(tmp_dir, worker)

            class NodeSet:
                pass

            node_set_module = types.ModuleType("cognee.modules.engine.models.node_set")
            node_set_module.NodeSet = NodeSet
            sys.modules["cognee.modules.engine.models.node_set"] = node_set_module

            class FakeCognee:
                async def search(self, **kwargs):
                    return [{"search_result": "stored memory", "dataset_name": "default"}]

            captured = {}

            async def run_operation(label, operation, *args, **kwargs):
                captured.update(kwargs)
                op_kwargs = dict(kwargs)
                op_kwargs.pop("timeout", None)
                op_kwargs.pop("operation_timeout", None)
                op_kwargs.pop("a0_agent", None)
                return await operation(*args, **op_kwargs)

            memory_module._get_cognee = lambda: (FakeCognee(), None)
            memory_module.run_cognee_operation = run_operation

            memory = memory_module.Memory("default", "default")
            asyncio.run(
                memory.search_similarity_threshold(
                    query="test",
                    limit=5,
                    threshold=0.7,
                )
            )

            self.assertEqual(captured.get("timeout"), memory_module.Memory.SEARCH_TIMEOUT)
            self.assertEqual(
                captured.get("operation_timeout"),
                memory_module.Memory.SEARCH_TIMEOUT,
            )

    def test_similarity_search_defaults_to_memory_node_sets(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = FakeDirtyWorker()
            memory_module = _load_memory_module(tmp_dir, worker)

            class NodeSet:
                pass

            node_set_module = types.ModuleType("cognee.modules.engine.models.node_set")
            node_set_module.NodeSet = NodeSet
            sys.modules["cognee.modules.engine.models.node_set"] = node_set_module

            class FakeCognee:
                def __init__(self):
                    self.search_calls = []

                async def search(self, **kwargs):
                    self.search_calls.append(kwargs)
                    return [{"search_result": "stored memory", "dataset_name": "default"}]

            fake_cognee = FakeCognee()
            memory_module._get_cognee = lambda: (fake_cognee, None)

            memory = memory_module.Memory("default", "default")
            asyncio.run(
                memory.search_similarity_threshold(
                    query="test",
                    limit=5,
                    threshold=0.7,
                    filter="",
                )
            )

            self.assertEqual(
                fake_cognee.search_calls[0]["node_name"],
                ["main", "fragments", "solutions"],
            )

    def test_similarity_search_serializes_cognee_search_calls(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = FakeDirtyWorker()
            memory_module = _load_memory_module(tmp_dir, worker)

            class NodeSet:
                pass

            node_set_module = types.ModuleType("cognee.modules.engine.models.node_set")
            node_set_module.NodeSet = NodeSet
            sys.modules["cognee.modules.engine.models.node_set"] = node_set_module

            class FakeCognee:
                def __init__(self):
                    self.active = 0
                    self.max_active = 0

                async def search(self, **kwargs):
                    self.active += 1
                    self.max_active = max(self.max_active, self.active)
                    await asyncio.sleep(0.01)
                    self.active -= 1
                    return [{"search_result": "stored memory", "dataset_name": "default"}]

            fake_cognee = FakeCognee()
            memory_module._get_cognee = lambda: (fake_cognee, None)

            memory = memory_module.Memory("default", "default")

            async def run_searches():
                await asyncio.gather(
                    memory.search_similarity_threshold("first", 5, 0.7),
                    memory.search_similarity_threshold("second", 5, 0.7),
                )

            asyncio.run(run_searches())

            self.assertEqual(fake_cognee.max_active, 1)

    def test_insert_text_reports_underlying_cognee_add_error(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = FakeDirtyWorker()
            memory_module = _load_memory_module(tmp_dir, worker)

            class FakeCognee:
                async def add(self, *args, **kwargs):
                    raise ValueError("structured output probe returned testtest")

            memory_module._get_cognee = lambda: (FakeCognee(), None)

            memory = memory_module.Memory("default", "default")
            with self.assertRaisesRegex(
                RuntimeError,
                "Last insert error: ValueError: structured output probe returned testtest",
            ):
                asyncio.run(
                    memory.insert_text(
                        "new memory",
                        {"area": "fragments", "timestamp": "now"},
                    )
                )

    def test_insert_documents_persists_non_cognee_metadata_by_content_hash(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = FakeDirtyWorker()
            memory_module = _load_memory_module(tmp_dir, worker)

            class FakeCognee:
                async def add(self, *args, **kwargs):
                    return None

            memory_module._get_cognee = lambda: (FakeCognee(), None)

            memory = memory_module.Memory("default", "default")
            asyncio.run(
                memory.insert_documents(
                    [
                        memory_module.Document(
                            page_content="stored memory",
                            metadata={
                                "area": "main",
                                "tags": ["important"],
                                "source_file": "notes.md",
                                "id": "old-id",
                            },
                        )
                    ]
                )
            )

            metadata = memory_module.get_persisted_metadata(
                "default",
                "default",
                "stored memory",
            )

            self.assertEqual(metadata["tags"], ["important"])
            self.assertEqual(metadata["source_file"], "notes.md")
            self.assertNotIn("id", metadata)

    def test_update_documents_inserts_replacement_before_deleting_old_memory(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = FakeDirtyWorker()
            memory_module = _load_memory_module(tmp_dir, worker)
            calls = []

            memory = memory_module.Memory("default", "default")

            async def insert_documents(docs):
                calls.append(("insert", [doc.page_content for doc in docs]))
                return ["new-id"]

            async def delete_documents_by_ids(ids):
                calls.append(("delete", ids))
                return []

            memory.insert_documents = insert_documents
            memory.delete_documents_by_ids = delete_documents_by_ids

            doc = memory_module.Document(
                page_content="new content",
                metadata={"id": "old-id"},
            )

            result = asyncio.run(memory.update_documents([doc]))

            self.assertEqual(result, ["new-id"])
            self.assertEqual(calls, [("insert", ["new content"]), ("delete", ["old-id"])])

    def test_update_documents_does_not_delete_old_memory_when_insert_fails(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = FakeDirtyWorker()
            memory_module = _load_memory_module(tmp_dir, worker)
            calls = []

            memory = memory_module.Memory("default", "default")

            async def insert_documents(docs):
                calls.append(("insert", [doc.page_content for doc in docs]))
                return []

            async def delete_documents_by_ids(ids):
                calls.append(("delete", ids))
                return []

            memory.insert_documents = insert_documents
            memory.delete_documents_by_ids = delete_documents_by_ids

            doc = memory_module.Document(
                page_content="new content",
                metadata={"id": "old-id"},
            )

            result = asyncio.run(memory.update_documents([doc]))

            self.assertEqual(result, [])
            self.assertEqual(calls, [("insert", ["new content"])])

    def test_simple_dedup_inserts_before_deleting_similar_memory(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = FakeDirtyWorker()
            memory_module = _load_memory_module(tmp_dir, worker)
            calls = []

            class FakeDB:
                async def search_similarity_threshold(self, **kwargs):
                    calls.append(("search", kwargs["query"]))
                    return [
                        memory_module.Document(
                            page_content="old content",
                            metadata={"id": "old-id"},
                        )
                    ]

                async def insert_text(self, *, text, metadata):
                    calls.append(("insert", text))
                    return "new-id"

                async def delete_documents_by_ids(self, ids):
                    calls.append(("delete", ids))
                    return []

            result = asyncio.run(
                memory_module.insert_with_simple_dedup(FakeDB(), "new content", "main", 0.9)
            )

            self.assertEqual(result, "new-id")
            self.assertEqual(
                calls,
                [("search", "new content"), ("insert", "new content"), ("delete", ["old-id"])],
            )

    def test_simple_dedup_does_not_delete_similar_memory_when_insert_fails(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = FakeDirtyWorker()
            memory_module = _load_memory_module(tmp_dir, worker)
            calls = []

            class FakeDB:
                async def search_similarity_threshold(self, **kwargs):
                    calls.append(("search", kwargs["query"]))
                    return [
                        memory_module.Document(
                            page_content="old content",
                            metadata={"id": "old-id"},
                        )
                    ]

                async def insert_text(self, *, text, metadata):
                    calls.append(("insert", text))
                    raise RuntimeError("insert failed")

                async def delete_documents_by_ids(self, ids):
                    calls.append(("delete", ids))
                    return []

            with self.assertRaisesRegex(RuntimeError, "insert failed"):
                asyncio.run(
                    memory_module.insert_with_simple_dedup(
                        FakeDB(),
                        "new content",
                        "main",
                        0.9,
                    )
                )

            self.assertEqual(calls, [("search", "new content"), ("insert", "new content")])


if __name__ == "__main__":
    unittest.main()
