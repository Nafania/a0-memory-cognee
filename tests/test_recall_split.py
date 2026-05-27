import importlib.util
import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_memory_module():
    helpers = types.ModuleType("helpers")
    files = types.ModuleType("helpers.files")
    print_style = types.ModuleType("helpers.print_style")
    log = types.ModuleType("helpers.log")

    files.get_abs_path = lambda *parts: "/tmp/" + "/".join(parts)

    class PrintStyle:
        @staticmethod
        def error(*args, **kwargs):
            pass

        @staticmethod
        def warning(*args, **kwargs):
            pass

    print_style.PrintStyle = PrintStyle
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
    knowledge_import.KnowledgeImport = dict

    cognee_init = types.ModuleType("usr.plugins.memory_cognee.helpers.cognee_init")
    cognee_init.get_cognee_setting = lambda key, default=None: default

    sys.modules.update(
        {
            "helpers": helpers,
            "helpers.files": files,
            "helpers.print_style": print_style,
            "helpers.log": log,
            "agent": types.SimpleNamespace(Agent=object, AgentContext=object),
            "models": types.ModuleType("models"),
            "usr.plugins.memory_cognee.helpers.knowledge_import": knowledge_import,
            "usr.plugins.memory_cognee.helpers.cognee_init": cognee_init,
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


class RecallSplitTest(unittest.TestCase):
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

    def test_split_uses_metadata_not_text_shape(self):
        memory = _load_memory_module()

        memories, solutions = memory.split_recall_answers_by_area(
            [{"text": "# Solution\nThis is only text, no area metadata"}],
            memory_limit=5,
            solution_limit=3,
        )

        self.assertEqual([doc.page_content for doc in memories], [
            "# Solution\nThis is only text, no area metadata"
        ])
        self.assertEqual(solutions, [])

    def test_split_uses_node_area_metadata(self):
        memory = _load_memory_module()
        node = types.SimpleNamespace(
            id="solution-node",
            attributes={"text": "stored solution", "area": "solutions"},
        )

        memories, solutions = memory.split_recall_answers_by_area(
            [{"dataset_name": "default", "objects_result": [node]}],
            memory_limit=5,
            solution_limit=3,
        )

        self.assertEqual(memories, [])
        self.assertEqual([doc.page_content for doc in solutions], ["stored solution"])

    def test_results_use_context_when_verbose_objects_have_no_text(self):
        memory = _load_memory_module()
        empty_verbose_node = types.SimpleNamespace(
            id="chunk-node",
            attributes={},
        )

        docs = memory._results_to_documents(
            [
                {
                    "dataset_name": "default",
                    "context_result": "stored chunk text",
                    "objects_result": [empty_verbose_node],
                }
            ],
            limit=5,
        )

        self.assertEqual([doc.page_content for doc in docs], ["stored chunk text"])

    def test_recall_feedback_accepts_split_document_results(self):
        memory = _load_memory_module()
        doc = memory.Document(
            page_content="stored split memory",
            metadata={"id": "memory-id", "dataset": "default"},
        )

        texts, feedback = memory.recall_text_and_feedback_items(
            [doc],
            limit=5,
            context_id="ctx",
            fallback_dataset="fallback",
            kind="memory",
        )

        self.assertEqual(texts, ["stored split memory"])
        self.assertEqual(
            feedback,
            [
                {
                    "text": "stored split memory",
                    "memory_id": "memory-id",
                    "dataset": "default",
                    "context_id": "ctx",
                    "kind": "memory",
                }
            ],
        )

    def test_results_expand_ranked_chunk_payloads_preserving_cognee_order(self):
        memory = _load_memory_module()

        docs = memory._results_to_documents(
            [
                {
                    "dataset_name": "default",
                    "search_result": [
                        {
                            "text": "first ranked memory chunk",
                            "metadata": {"area": "main"},
                            "id": "chunk-1",
                        },
                        {
                            "text": "second ranked solution chunk",
                            "metadata": {"area": "solutions"},
                            "id": "chunk-2",
                        },
                    ],
                }
            ],
            limit=5,
        )

        self.assertEqual(
            [doc.page_content for doc in docs],
            ["first ranked memory chunk", "second ranked solution chunk"],
        )
        self.assertEqual([doc.metadata["id"] for doc in docs], ["chunk-1", "chunk-2"])
        self.assertEqual([doc.metadata["dataset"] for doc in docs], ["default", "default"])

        memories, solutions = memory.split_recall_answers_by_area(
            [
                {
                    "dataset_name": "default",
                    "search_result": [
                        {
                            "text": "first ranked memory chunk",
                            "metadata": {"area": "main"},
                            "id": "chunk-1",
                        },
                        {
                            "text": "second ranked solution chunk",
                            "metadata": {"area": "solutions"},
                            "id": "chunk-2",
                        },
                    ],
                }
            ],
            memory_limit=5,
            solution_limit=3,
        )

        self.assertEqual([doc.page_content for doc in memories], ["first ranked memory chunk"])
        self.assertEqual([doc.page_content for doc in solutions], ["second ranked solution chunk"])

    def test_split_uses_cognee_chunk_belongs_to_set_metadata(self):
        memory = _load_memory_module()

        memories, solutions = memory.split_recall_answers_by_area(
            [
                {
                    "text": "chunk text stored in solution area",
                    "belongs_to_set": ["solutions"],
                    "id": "chunk-solution",
                }
            ],
            memory_limit=5,
            solution_limit=3,
        )

        self.assertEqual(memories, [])
        self.assertEqual([doc.page_content for doc in solutions], [
            "chunk text stored in solution area"
        ])

    def test_split_uses_cognee_camel_case_chunk_area_metadata(self):
        memory = _load_memory_module()

        memories, solutions = memory.split_recall_answers_by_area(
            [
                {
                    "text": "camel case chunk text stored in solution area",
                    "belongsToSet": ["solutions"],
                    "nodeName": "solutions",
                    "id": "chunk-solution",
                }
            ],
            memory_limit=5,
            solution_limit=3,
        )

        self.assertEqual(memories, [])
        self.assertEqual([doc.page_content for doc in solutions], [
            "camel case chunk text stored in solution area"
        ])

    def test_split_uses_normalized_chunk_raw_metadata(self):
        memory = _load_memory_module()
        result = types.SimpleNamespace(
            text="normalized chunk text stored in solution area",
            metadata={},
            raw={"belongs_to_set": ["solutions"]},
            dataset_name="default",
        )

        memories, solutions = memory.split_recall_answers_by_area(
            [result],
            memory_limit=5,
            solution_limit=3,
        )

        self.assertEqual(memories, [])
        self.assertEqual([doc.page_content for doc in solutions], [
            "normalized chunk text stored in solution area"
        ])

    def test_split_uses_normalized_chunk_payload_metadata(self):
        memory = _load_memory_module()
        result = types.SimpleNamespace(
            text="payload chunk text stored in solution area",
            metadata={},
            payload={"belongsToSet": ["solutions"]},
            dataset_name="default",
        )

        memories, solutions = memory.split_recall_answers_by_area(
            [result],
            memory_limit=5,
            solution_limit=3,
        )

        self.assertEqual(memories, [])
        self.assertEqual([doc.page_content for doc in solutions], [
            "payload chunk text stored in solution area"
        ])

    def test_split_uses_direct_object_area_metadata(self):
        memory = _load_memory_module()
        result = types.SimpleNamespace(
            text="direct chunk text stored in solution area",
            metadata={},
            nodeName="solutions",
            dataset_name="default",
        )

        memories, solutions = memory.split_recall_answers_by_area(
            [result],
            memory_limit=5,
            solution_limit=3,
        )

        self.assertEqual(memories, [])
        self.assertEqual([doc.page_content for doc in solutions], [
            "direct chunk text stored in solution area"
        ])


if __name__ == "__main__":
    unittest.main()
