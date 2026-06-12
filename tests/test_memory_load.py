import asyncio
import importlib.util
import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _install_memory_tool_stubs(search_error_message: str | None = None):
    package_names = [
        "usr",
        "usr.plugins",
        "usr.plugins.memory_cognee",
        "usr.plugins.memory_cognee.tools",
        "usr.plugins.memory_cognee.helpers",
    ]
    for name in package_names:
        package = types.ModuleType(name)
        package.__path__ = [
            str(REPO_ROOT / "tools")
        ] if name.endswith(".tools") else []
        sys.modules[name] = package

    helpers_tool = types.ModuleType("helpers.tool")

    class Response:
        def __init__(self, message="", break_loop=False):
            self.message = message
            self.break_loop = break_loop

    class Tool:
        pass

    helpers_tool.Tool = Tool
    helpers_tool.Response = Response

    memory_module = types.ModuleType("usr.plugins.memory_cognee.helpers.memory")

    class SearchUnavailable(Exception):
        pass

    class Memory:
        get_calls = []

        @staticmethod
        async def get(agent, **kwargs):
            Memory.get_calls.append(kwargs)
            return Memory()

        async def search_similarity_threshold(self, **kwargs):
            if search_error_message:
                raise SearchUnavailable(search_error_message)
            return []

        async def insert_text(self, text, metadata):
            return "memory-id"

        async def delete_documents_by_ids(self, ids):
            return []

        async def delete_documents_by_query(self, **kwargs):
            if search_error_message:
                raise SearchUnavailable(search_error_message)
            return []

        @staticmethod
        def format_docs_plain(docs):
            return [doc.page_content for doc in docs]

    memory_module.Memory = Memory
    memory_module.SearchUnavailable = SearchUnavailable

    sys.modules.update(
        {
            "helpers.tool": helpers_tool,
            "usr.plugins.memory_cognee.helpers.memory": memory_module,
        }
    )

    return SearchUnavailable


def _load_tool_module(module_filename: str, module_name: str, search_error_message: str | None = None):
    SearchUnavailable = _install_memory_tool_stubs(search_error_message)
    module_path = REPO_ROOT / "tools" / module_filename
    spec = importlib.util.spec_from_file_location(
        f"usr.plugins.memory_cognee.tools.{module_name}",
        module_path,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, SearchUnavailable


def _load_memory_load_module(search_error_message: str | None = None):
    return _load_tool_module("memory_load.py", "memory_load", search_error_message)


class FakeAgent:
    context = types.SimpleNamespace(id="ctx-1")

    def read_prompt(self, name, **kwargs):
        return f"{name}:{kwargs}"


class MemoryLoadTest(unittest.TestCase):
    def tearDown(self):
        for name in list(sys.modules):
            if (
                name == "helpers.tool"
                or name.startswith("usr.plugins.memory_cognee")
                or name in ("usr", "usr.plugins")
            ):
                sys.modules.pop(name, None)

    def test_memory_load_returns_explicit_unavailable_error(self):
        module, _ = _load_memory_load_module(
            "Cognee memory graph rebuild failed"
        )
        tool = module.MemoryLoad()
        tool.agent = FakeAgent()

        response = asyncio.run(tool.execute(query="project memory"))

        self.assertIn("Memory search unavailable", response.message)
        self.assertIn("Cognee memory graph rebuild failed", response.message)
        self.assertEqual(module.Memory.get_calls[0], {"preload_knowledge": False})

    def test_memory_save_uses_read_only_get_before_write(self):
        module, _ = _load_tool_module("memory_save.py", "memory_save")
        tool = module.MemorySave()
        tool.agent = FakeAgent()

        response = asyncio.run(tool.execute(text="remember me", area="main"))

        self.assertIn("fw.memory_saved.md", response.message)
        self.assertEqual(module.Memory.get_calls[0], {"preload_knowledge": False})

    def test_memory_delete_uses_read_only_get_before_delete(self):
        module, _ = _load_tool_module("memory_delete.py", "memory_delete")
        tool = module.MemoryDelete()
        tool.agent = FakeAgent()

        response = asyncio.run(tool.execute(ids="id-1,id-2"))

        self.assertIn("fw.memories_deleted.md", response.message)
        self.assertEqual(module.Memory.get_calls[0], {"preload_knowledge": False})

    def test_memory_forget_returns_explicit_unavailable_error(self):
        module, _ = _load_tool_module(
            "memory_forget.py",
            "memory_forget",
            "Cognee memory graph rebuild pending",
        )
        tool = module.MemoryForget()
        tool.agent = FakeAgent()

        response = asyncio.run(tool.execute(query="old memory"))

        self.assertIn("Memory search unavailable", response.message)
        self.assertIn("Cognee memory graph rebuild pending", response.message)
        self.assertEqual(module.Memory.get_calls[0], {"preload_knowledge": False})


if __name__ == "__main__":
    unittest.main()
