import asyncio
import importlib.util
import sys
import threading
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
    cognee_init = types.ModuleType("usr.plugins.memory_cognee.helpers.cognee_init")
    cognee_init.ensure_cognee_llm_config_current = lambda a0_agent=None: None
    sys.modules["usr.plugins.memory_cognee.helpers.cognee_init"] = cognee_init

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
        if hasattr(self.module, "_stop_cognee_operation_loop"):
            self.module._stop_cognee_operation_loop()
        with self.module._COGNEE_WAIT_STATE_LOCK:
            self.module._USER_WAITERS = 0
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

    def test_user_priority_skips_waiting_background_operation(self):
        self.module.multiprocessing.active_children = lambda: []
        self.module._child_holds_cognee_file = lambda pid: False

        async def run_priority_race():
            self.module._COGNEE_OP_LOCK.acquire()
            order = []

            async def background_operation():
                order.append("background")
                return "background-ok"

            async def user_operation():
                order.append("user")
                return "user-ok"

            background_task = asyncio.create_task(
                self.module.run_cognee_operation(
                    "background wait",
                    background_operation,
                    timeout=1,
                    priority="background",
                )
            )
            await asyncio.sleep(0.02)
            user_task = asyncio.create_task(
                self.module.run_cognee_operation(
                    "user search",
                    user_operation,
                    timeout=1,
                    priority="user",
                )
            )
            await asyncio.sleep(0.02)
            self.module._COGNEE_OP_LOCK.release()
            results = await asyncio.gather(user_task, background_task)
            return order, results

        order, results = asyncio.run(run_priority_race())

        self.assertEqual(order, ["user", "background"])
        self.assertEqual(results, ["user-ok", "background-ok"])

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

    def test_run_cognee_operation_refreshes_llm_config_before_call(self):
        agent = object()
        refresh_calls = []
        operation_calls = []
        self.module.multiprocessing.active_children = lambda: []
        self.module._child_holds_cognee_file = lambda pid: False

        cognee_init = types.ModuleType("usr.plugins.memory_cognee.helpers.cognee_init")
        cognee_init.ensure_cognee_llm_config_current = (
            lambda a0_agent=None: refresh_calls.append(a0_agent)
        )
        sys.modules["usr.plugins.memory_cognee.helpers.cognee_init"] = cognee_init

        def operation():
            operation_calls.append("called")
            return "ok"

        result = asyncio.run(
            self.module.run_cognee_operation("test search", operation, a0_agent=agent)
        )

        self.assertEqual(result, "ok")
        self.assertEqual(refresh_calls, [agent])
        self.assertEqual(operation_calls, ["called"])

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

    def test_run_cognee_operation_reuses_stable_worker_loop_across_callers(self):
        self.module.multiprocessing.active_children = lambda: []
        self.module._child_holds_cognee_file = lambda pid: False

        caller_loops = []
        operation_loops = []

        async def call_once():
            caller_loops.append(asyncio.get_running_loop())

            async def operation():
                operation_loops.append(asyncio.get_running_loop())
                return "ok"

            return await self.module.run_cognee_operation(
                "test stable operation loop",
                operation,
            )

        self.assertEqual(asyncio.run(call_once()), "ok")
        self.assertEqual(asyncio.run(call_once()), "ok")

        self.assertEqual(len(operation_loops), 2)
        self.assertIs(operation_loops[0], operation_loops[1])
        self.assertIsNot(operation_loops[0], caller_loops[0])
        self.assertIsNot(operation_loops[1], caller_loops[1])

    def test_run_cognee_operation_cancels_worker_loop_future_on_caller_cancel(self):
        self.module.multiprocessing.active_children = lambda: []
        self.module._child_holds_cognee_file = lambda pid: False
        started = threading.Event()

        async def run_and_cancel():
            async def operation():
                started.set()
                await asyncio.sleep(60)

            task = asyncio.create_task(
                self.module.run_cognee_operation(
                    "test cancelled stable loop operation",
                    operation,
                )
            )
            while not started.wait(timeout=0.01):
                await asyncio.sleep(0.01)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        asyncio.run(run_and_cancel())

    def test_run_cognee_operation_preserves_existing_litellm_worker(self):
        self.module.multiprocessing.active_children = lambda: []
        self.module._child_holds_cognee_file = lambda pid: False

        class FakeLiteLLMWorker:
            def __init__(self):
                self._worker_task = None
                self._running_tasks = set()
                self.flush_calls = 0
                self.stop_calls = 0

            async def flush(self):
                self.flush_calls += 1

            async def stop(self):
                self.stop_calls += 1

        worker = FakeLiteLLMWorker()
        litellm = types.ModuleType("litellm")
        litellm_core_utils = types.ModuleType("litellm.litellm_core_utils")
        logging_worker = types.ModuleType("litellm.litellm_core_utils.logging_worker")
        logging_worker.GLOBAL_LOGGING_WORKER = worker
        old_modules = {
            name: sys.modules.get(name)
            for name in (
                "litellm",
                "litellm.litellm_core_utils",
                "litellm.litellm_core_utils.logging_worker",
            )
        }
        sys.modules.update(
            {
                "litellm": litellm,
                "litellm.litellm_core_utils": litellm_core_utils,
                "litellm.litellm_core_utils.logging_worker": logging_worker,
            }
        )

        async def run_operation():
            worker._worker_task = asyncio.create_task(asyncio.sleep(60))

            async def operation():
                return "ok"

            try:
                return await self.module.run_cognee_operation(
                    "test existing litellm worker",
                    operation,
                )
            finally:
                worker._worker_task.cancel()
                await asyncio.gather(worker._worker_task, return_exceptions=True)

        try:
            result = asyncio.run(run_operation())
        finally:
            for name, module in old_modules.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

        self.assertEqual(result, "ok")
        self.assertEqual(worker.flush_calls, 0)
        self.assertEqual(worker.stop_calls, 0)

    def test_gate_timeout_reports_current_operation_holder(self):
        self.module.multiprocessing.active_children = lambda: []
        self.module._child_holds_cognee_file = lambda pid: False
        started = threading.Event()

        async def run_blocked_waiter():
            async def holder_operation():
                started.set()
                await asyncio.sleep(0.2)

            holder_task = asyncio.create_task(
                self.module.run_cognee_operation(
                    "background rebuild",
                    holder_operation,
                    operation_timeout=1,
                )
            )
            while not started.wait(timeout=0.01):
                await asyncio.sleep(0.01)

            async def waiter_operation():
                return "unexpected"

            try:
                with self.assertRaisesRegex(
                    TimeoutError,
                    r"Timed out waiting for Cognee operation gate: user search; "
                    r"current='background rebuild'; held_for=",
                ):
                    await self.module.run_cognee_operation(
                        "user search",
                        waiter_operation,
                        timeout=0.01,
                    )
            finally:
                await holder_task

        asyncio.run(run_blocked_waiter())


if __name__ == "__main__":
    unittest.main()
