import asyncio
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_feedback_module(tmp_dir: str):
    helpers = types.ModuleType("helpers")
    files = types.ModuleType("helpers.files")
    settings = types.ModuleType("helpers.settings")

    files.get_abs_path = lambda *parts: os.path.join(tmp_dir, *parts)
    settings.get_settings = lambda: {"cognee_feedback_enabled": True}

    sys.modules.update(
        {
            "helpers": helpers,
            "helpers.files": files,
            "helpers.settings": settings,
        }
    )

    module_path = REPO_ROOT / "helpers" / "cognee_feedback.py"
    spec = importlib.util.spec_from_file_location(
        "usr.plugins.memory_cognee.helpers.cognee_feedback",
        module_path,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class CogneeFeedbackTest(unittest.TestCase):
    def tearDown(self):
        for name in list(sys.modules):
            if name == "helpers" or name.startswith("helpers."):
                sys.modules.pop(name, None)

    def test_parallel_queue_drains_do_not_forward_same_file_twice(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            module = _load_feedback_module(tmp_dir)
            pending_dir = module._pending_dir()
            os.makedirs(pending_dir, exist_ok=True)
            path = os.path.join(pending_dir, "one.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "context_id": "ctx",
                        "dataset": "default",
                        "memory_id": "memory-id",
                        "feedback": "positive",
                        "attempts": 0,
                    },
                    f,
                )

            state = {"active": 0, "max_active": 0, "count": 0}

            async def try_forward(cognee_module, payload):
                state["active"] += 1
                state["max_active"] = max(state["max_active"], state["active"])
                state["count"] += 1
                await asyncio.sleep(0.01)
                state["active"] -= 1
                return True

            module._try_forward = try_forward

            async def run_drains():
                return await asyncio.gather(
                    module.drain_feedback_queue(cognee_module=object()),
                    module.drain_feedback_queue(cognee_module=object()),
                )

            forwarded = asyncio.run(run_drains())

            self.assertEqual(sum(forwarded), 1)
            self.assertEqual(state["count"], 1)
            self.assertEqual(state["max_active"], 1)


if __name__ == "__main__":
    unittest.main()
