from helpers.extension import Extension


class StartCogneeWorker(Extension):

    def execute(self, **kwargs):
        from usr.plugins.memory_cognee.helpers.cognee_init import (
            run_memory_cognee_start_worker_extension,
        )

        run_memory_cognee_start_worker_extension()
