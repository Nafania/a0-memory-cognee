import asyncio
import importlib.util
import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_memory_init_module(block_reason: str | None = None):
    package_names = [
        "usr",
        "usr.plugins",
        "usr.plugins.memory_cognee",
        "usr.plugins.memory_cognee.helpers",
        "usr.plugins.memory_cognee.extensions",
        "usr.plugins.memory_cognee.extensions.python",
        "usr.plugins.memory_cognee.extensions.python.monologue_start",
    ]
    for name in package_names:
        package = types.ModuleType(name)
        package.__path__ = [str(REPO_ROOT / "helpers")] if name.endswith(".helpers") else []
        sys.modules[name] = package

    helpers_extension = types.ModuleType("helpers.extension")

    class Extension:
        pass

    helpers_extension.Extension = Extension

    print_style = types.ModuleType("helpers.print_style")
    print_style.PrintStyle = types.SimpleNamespace(warning=lambda *args, **kwargs: None)

    agent_module = types.ModuleType("agent")
    agent_module.LoopData = object

    calls = []

    class FakeMemoryInstance:
        def get_search_datasets(self):
            return ["default"]

    class FakeMemory:
        @staticmethod
        async def get(agent, **kwargs):
            calls.append(kwargs)
            return FakeMemoryInstance()

    memory_module = types.ModuleType("usr.plugins.memory_cognee.helpers.memory")
    memory_module.Memory = FakeMemory

    background_module = types.ModuleType(
        "usr.plugins.memory_cognee.helpers.cognee_background"
    )

    class CogneeBackgroundWorker:
        @staticmethod
        def get_instance():
            return types.SimpleNamespace(
                get_search_block_reason=lambda datasets: block_reason
            )

    background_module.CogneeBackgroundWorker = CogneeBackgroundWorker

    sys.modules.update(
        {
            "helpers.extension": helpers_extension,
            "helpers.print_style": print_style,
            "agent": agent_module,
            "usr.plugins.memory_cognee.helpers.memory": memory_module,
            "usr.plugins.memory_cognee.helpers.cognee_background": background_module,
        }
    )

    module_path = (
        REPO_ROOT / "extensions" / "python" / "monologue_start" / "_10_memory_init.py"
    )
    spec = importlib.util.spec_from_file_location(
        "usr.plugins.memory_cognee.extensions.python.monologue_start._10_memory_init",
        module_path,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, calls


class MemoryInitExtensionTest(unittest.TestCase):
    def tearDown(self):
        for name in list(sys.modules):
            if (
                name == "helpers.extension"
                or name == "helpers.print_style"
                or name == "agent"
                or name.startswith("usr.plugins.memory_cognee")
                or name in ("usr", "usr.plugins")
            ):
                sys.modules.pop(name, None)

    def test_skips_preload_when_background_rebuild_blocks_search(self):
        module, calls = _load_memory_init_module(
            "Cognee memory graph rebuild running for dataset(s): ['default']"
        )
        extension = module.MemoryInit()
        extension.agent = object()

        asyncio.run(extension.execute())

        self.assertEqual(calls, [{"preload_knowledge": False}])

    def test_preloads_when_rebuild_not_blocking(self):
        module, calls = _load_memory_init_module()
        extension = module.MemoryInit()
        extension.agent = object()

        asyncio.run(extension.execute())

        self.assertEqual(calls, [{"preload_knowledge": False}, {}])


if __name__ == "__main__":
    unittest.main()
