import asyncio
import importlib.util
import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_cognee_ops_module():
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

    module_path = REPO_ROOT / "helpers" / "cognee_ops.py"
    spec = importlib.util.spec_from_file_location(
        "usr.plugins.memory_cognee.helpers.cognee_ops",
        module_path,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeChild:
    def __init__(self, pid: int):
        self.pid = pid
        self.name = f"child-{pid}"
        self.terminated = False
        self.killed = False

    def is_alive(self):
        return not self.terminated and not self.killed

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def join(self, timeout=None):
        return None


class CogneeOpsTest(unittest.TestCase):
    def setUp(self):
        self.module = _load_cognee_ops_module()
        self.original_active_children = self.module.multiprocessing.active_children
        self.original_child_holds = self.module._child_holds_cognee_file
        self.module._COGNEE_CHILD_PROCESS_PIDS.clear()

    def tearDown(self):
        self.module.multiprocessing.active_children = self.original_active_children
        self.module._child_holds_cognee_file = self.original_child_holds
        self.module._COGNEE_CHILD_PROCESS_PIDS.clear()
        for name in list(sys.modules):
            if name.startswith("usr.plugins.memory_cognee") or name == "usr":
                sys.modules.pop(name, None)

    def test_run_cognee_operation_cleans_new_child_after_success(self):
        children: list[FakeChild] = []
        spawned_child = FakeChild(202)
        self.module.multiprocessing.active_children = lambda: children
        self.module._child_holds_cognee_file = lambda pid: False

        async def operation():
            children.append(spawned_child)
            return "ok"

        result = asyncio.run(
            self.module.run_cognee_operation("test search", operation)
        )

        self.assertEqual(result, "ok")
        self.assertTrue(spawned_child.terminated)

    def test_run_cognee_operation_cleans_stale_cognee_fd_child_before_call(self):
        stale_child = FakeChild(303)
        call_observed_stale_terminated = []
        self.module.multiprocessing.active_children = lambda: [stale_child]
        self.module._child_holds_cognee_file = lambda pid: pid == stale_child.pid

        async def operation():
            call_observed_stale_terminated.append(stale_child.terminated)
            return "ok"

        result = asyncio.run(
            self.module.run_cognee_operation("test search", operation)
        )

        self.assertEqual(result, "ok")
        self.assertEqual(call_observed_stale_terminated, [True])

    def test_run_cognee_operation_preserves_unrelated_baseline_child(self):
        baseline_child = FakeChild(404)
        self.module.multiprocessing.active_children = lambda: [baseline_child]
        self.module._child_holds_cognee_file = lambda pid: False

        async def operation():
            return "ok"

        result = asyncio.run(
            self.module.run_cognee_operation("test search", operation)
        )

        self.assertEqual(result, "ok")
        self.assertFalse(baseline_child.terminated)
        self.assertFalse(baseline_child.killed)

    def test_run_cognee_operation_cleans_new_child_after_timeout(self):
        children: list[FakeChild] = []
        spawned_child = FakeChild(505)
        self.module.multiprocessing.active_children = lambda: children
        self.module._child_holds_cognee_file = lambda pid: False

        async def operation():
            children.append(spawned_child)
            await asyncio.sleep(1)

        with self.assertRaises(TimeoutError):
            asyncio.run(
                self.module.run_cognee_operation(
                    "test timeout",
                    operation,
                    operation_timeout=0.01,
                )
            )

        self.assertTrue(spawned_child.terminated)


if __name__ == "__main__":
    unittest.main()
