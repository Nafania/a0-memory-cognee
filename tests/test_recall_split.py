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
        package.__path__ = []
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


if __name__ == "__main__":
    unittest.main()
