import subprocess
import sys

def install():
    # 1. Install Cognee
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "cognee[fastembed]"
    ])

    # 2. Disable builtin _memory
    try:
        from helpers.plugins import toggle_plugin, after_plugin_change
        toggle_plugin("_memory", False)
        after_plugin_change(["_memory"])
        from helpers.print_style import PrintStyle
        PrintStyle.standard("Builtin _memory plugin has been disabled.")
    except Exception as e:
        from helpers.print_style import PrintStyle
        PrintStyle.warning(f"Could not auto-disable _memory: {e}. Please disable manually in Settings > Plugins.")

    # 3. Migrate FAISS data if present
    try:
        from .helpers.faiss_migration import migrate
        migrate()
    except Exception as e:
        from helpers.print_style import PrintStyle
        PrintStyle.warning(f"FAISS migration skipped: {e}")
