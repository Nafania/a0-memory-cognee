from helpers.extension import Extension


class InitCognee(Extension):

    def execute(self, **kwargs):
        try:
            from usr.plugins.memory_cognee.helpers.cognee_init import configure_cognee
            configure_cognee()

            from usr.plugins.memory_cognee.helpers.cognee_background import CogneeBackgroundWorker
            CogneeBackgroundWorker.get_instance().start()
        except Exception as e:
            from helpers.print_style import PrintStyle
            PrintStyle.error(f"Cognee eager init failed (will retry lazily): {e}")
