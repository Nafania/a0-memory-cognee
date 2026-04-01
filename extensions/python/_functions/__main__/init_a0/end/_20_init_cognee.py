from helpers.extension import Extension


class InitCognee(Extension):

    def execute(self, **kwargs):
        import asyncio

        try:
            from usr.plugins._memory_cognee.helpers.cognee_init import init_cognee

            asyncio.get_event_loop().run_until_complete(init_cognee())

            from usr.plugins._memory_cognee.helpers.cognee_background import CogneeBackgroundWorker

            CogneeBackgroundWorker.get_instance().start()
        except Exception as e:
            from helpers.print_style import PrintStyle

            PrintStyle.error(f"Cognee initialization failed: {e}")
