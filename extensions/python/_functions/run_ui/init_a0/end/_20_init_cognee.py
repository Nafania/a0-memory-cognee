from helpers.extension import Extension


class InitCognee(Extension):

    def execute(self, **kwargs):
        from usr.plugins.memory_cognee.helpers.cognee_init import (
            run_memory_cognee_init_a0_extension,
        )

        run_memory_cognee_init_a0_extension()
