import asyncio
import importlib.util
import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_consolidation_module():
    package_names = [
        "usr",
        "usr.plugins",
        "usr.plugins.memory_cognee",
        "usr.plugins.memory_cognee.helpers",
        "usr.plugins.memory_cognee.tools",
    ]
    for name in package_names:
        package = types.ModuleType(name)
        package.__path__ = []
        sys.modules[name] = package

    memory = types.ModuleType("usr.plugins.memory_cognee.helpers.memory")

    class Memory:
        pass

    memory.Memory = Memory

    class SearchUnavailable(RuntimeError):
        pass

    memory.SearchUnavailable = SearchUnavailable

    llm_json = types.ModuleType("usr.plugins.memory_cognee.helpers.llm_json")
    llm_json.parse_llm_json_response = lambda *args, **kwargs: {}

    dirty_json = types.ModuleType("helpers.dirty_json")
    dirty_json.DirtyJson = object

    log = types.ModuleType("helpers.log")
    log.LogItem = object

    print_style = types.ModuleType("helpers.print_style")

    class PrintStyle:
        def error(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

    print_style.PrintStyle = PrintStyle

    memory_load = types.ModuleType("usr.plugins.memory_cognee.tools.memory_load")
    memory_load.DEFAULT_THRESHOLD = 0.7

    sys.modules.update(
        {
            "usr.plugins.memory_cognee.helpers.memory": memory,
            "usr.plugins.memory_cognee.helpers.llm_json": llm_json,
            "helpers.dirty_json": dirty_json,
            "helpers.log": log,
            "helpers.print_style": print_style,
            "agent": types.SimpleNamespace(Agent=object),
            "usr.plugins.memory_cognee.tools.memory_load": memory_load,
        }
    )

    module_path = REPO_ROOT / "helpers" / "memory_consolidation.py"
    spec = importlib.util.spec_from_file_location(
        "usr.plugins.memory_cognee.helpers.memory_consolidation",
        module_path,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeDB:
    def __init__(self, *, fail_insert: bool = False):
        self.fail_insert = fail_insert
        self.calls: list[tuple[str, object]] = []
        self.insert_metadata: list[dict] = []

    async def insert_text(self, text, metadata):
        self.calls.append(("insert", text))
        self.insert_metadata.append(dict(metadata))
        if self.fail_insert:
            raise RuntimeError("insert failed")
        return "new-id"

    async def delete_documents_by_ids(self, ids):
        self.calls.append(("delete", ids))
        return []


class MemoryConsolidationUpdateTest(unittest.TestCase):
    def tearDown(self):
        for name in list(sys.modules):
            if (
                name == "agent"
                or name.startswith("helpers.")
                or name.startswith("usr.plugins.memory_cognee")
            ):
                sys.modules.pop(name, None)

    def test_update_inserts_replacement_before_deleting_old_memory(self):
        module = _load_consolidation_module()
        consolidator = module.MemoryConsolidator(agent=object())
        db = FakeDB()
        result = module.ConsolidationResult(
            action=module.ConsolidationAction.UPDATE,
            memories_to_update=[{"id": "old-id", "new_content": "updated memory"}],
        )

        ids = asyncio.run(
            consolidator._handle_update(db, result, "main", {}, None)
        )

        self.assertEqual(ids, ["new-id"])
        self.assertEqual(
            db.calls,
            [("insert", "updated memory"), ("delete", ["old-id"])],
        )

    def test_update_does_not_delete_old_memory_when_insert_fails(self):
        module = _load_consolidation_module()
        consolidator = module.MemoryConsolidator(agent=object())
        db = FakeDB(fail_insert=True)
        result = module.ConsolidationResult(
            action=module.ConsolidationAction.UPDATE,
            memories_to_update=[{"id": "old-id", "new_content": "updated memory"}],
        )

        ids = asyncio.run(
            consolidator._handle_update(db, result, "main", {}, None)
        )

        self.assertEqual(ids, [])
        self.assertEqual(db.calls, [("insert", "updated memory")])

    def test_merge_inserts_consolidated_memory_before_deleting_old_memories(self):
        module = _load_consolidation_module()
        consolidator = module.MemoryConsolidator(agent=object())
        db = FakeDB()
        result = module.ConsolidationResult(
            action=module.ConsolidationAction.MERGE,
            memories_to_remove=["old-a", "old-b"],
            new_memory_content="merged memory",
        )

        ids = asyncio.run(consolidator._handle_merge(db, result, "main", {}, None))

        self.assertEqual(ids, ["new-id"])
        self.assertEqual(
            db.calls,
            [("insert", "merged memory"), ("delete", ["old-a", "old-b"])],
        )

    def test_merge_does_not_delete_old_memories_when_insert_fails(self):
        module = _load_consolidation_module()
        consolidator = module.MemoryConsolidator(agent=object())
        db = FakeDB(fail_insert=True)
        result = module.ConsolidationResult(
            action=module.ConsolidationAction.MERGE,
            memories_to_remove=["old-a", "old-b"],
            new_memory_content="merged memory",
        )

        ids = asyncio.run(consolidator._handle_merge(db, result, "main", {}, None))

        self.assertEqual(ids, [])
        self.assertEqual(db.calls, [("insert", "merged memory")])

    def test_replace_inserts_replacement_before_deleting_old_memories(self):
        module = _load_consolidation_module()
        consolidator = module.MemoryConsolidator(agent=object())
        db = FakeDB()
        result = module.ConsolidationResult(
            action=module.ConsolidationAction.REPLACE,
            memories_to_remove=["old-a"],
            new_memory_content="replacement memory",
        )

        ids = asyncio.run(consolidator._handle_replace(db, result, "main", {}, None))

        self.assertEqual(ids, ["new-id"])
        self.assertEqual(
            db.calls,
            [("insert", "replacement memory"), ("delete", ["old-a"])],
        )

    def test_replace_does_not_delete_old_memory_when_insert_fails(self):
        module = _load_consolidation_module()
        consolidator = module.MemoryConsolidator(agent=object())
        db = FakeDB(fail_insert=True)
        result = module.ConsolidationResult(
            action=module.ConsolidationAction.REPLACE,
            memories_to_remove=["old-a"],
            new_memory_content="replacement memory",
        )

        ids = asyncio.run(consolidator._handle_replace(db, result, "main", {}, None))

        self.assertEqual(ids, [])
        self.assertEqual(db.calls, [("insert", "replacement memory")])

    def test_result_metadata_cannot_override_reserved_memory_fields(self):
        module = _load_consolidation_module()
        consolidator = module.MemoryConsolidator(agent=object())
        db = FakeDB()
        result = module.ConsolidationResult(
            action=module.ConsolidationAction.KEEP_SEPARATE,
            new_memory_content="stored memory",
            metadata={
                "area": "solutions",
                "id": "attacker-id",
                "node_set": ["solutions"],
                "node_name": "solutions",
                "safe": "kept",
            },
        )

        ids = asyncio.run(
            consolidator._handle_keep_separate(
                db,
                result,
                "fragments",
                {"area": "main", "source": "original"},
                None,
            )
        )

        self.assertEqual(ids, ["new-id"])
        self.assertEqual(db.insert_metadata[0]["area"], "fragments")
        self.assertNotIn("id", db.insert_metadata[0])
        self.assertNotIn("node_set", db.insert_metadata[0])
        self.assertNotIn("node_name", db.insert_metadata[0])
        self.assertEqual(db.insert_metadata[0]["safe"], "kept")
        self.assertEqual(db.insert_metadata[0]["source"], "original")

    def test_find_similar_memories_requests_unavailable_search_errors(self):
        module = _load_consolidation_module()
        consolidator = module.MemoryConsolidator(agent=object())
        captured = []

        class FakeDB:
            async def search_similarity_threshold(self, **kwargs):
                captured.append(kwargs)
                return []

        async def get_memory(agent):
            return FakeDB()

        async def no_keywords(new_memory, log_item):
            return []

        module.Memory.get = get_memory
        consolidator._extract_search_keywords = no_keywords

        result = asyncio.run(
            consolidator._find_similar_memories("new memory", "main", None)
        )

        self.assertEqual(result, [])
        self.assertEqual(captured[0]["raise_unavailable"], True)


if __name__ == "__main__":
    unittest.main()
