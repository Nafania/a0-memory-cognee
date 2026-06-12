from helpers import plugins
from helpers.defer import DeferredTask, THREAD_BACKGROUND
from helpers.extension import Extension
from agent import LoopData
from usr.plugins.memory_cognee.helpers.deferred_tasks import track_deferred_task
from usr.plugins.memory_cognee.helpers.session_memory import safe_remember_session_turn


class RememberSessionTurn(Extension):
    async def execute(self, loop_data: LoopData = LoopData(), **kwargs):
        cfg = plugins.get_plugin_config("memory_cognee", self.agent)
        if not cfg or not cfg.get("memory_session_enabled", True):
            return

        task = DeferredTask(thread_name=THREAD_BACKGROUND)
        task.start_task(safe_remember_session_turn, self.agent)
        return track_deferred_task(task)
