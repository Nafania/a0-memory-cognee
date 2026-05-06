import os
import subprocess
import sys

_PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
_REQUIREMENTS_TXT = os.path.join(_PLUGIN_DIR, "requirements.txt")


def install():
    # 1. Keep OpenAI bounded to the compatible major while allowing Cognee's
    # LiteLLM dependency to select the required minor/major version.
    result = subprocess.run(
        [sys.executable, "-m", "pip", "show", "openai"],
        capture_output=True, text=True
    )
    pinned_openai = ""
    for line in result.stdout.splitlines():
        if line.startswith("Version:"):
            ver = line.split(":", 1)[1].strip()
            major = int(ver.split(".")[0])
            pinned_openai = "openai<3" if major < 3 else f"openai<{major + 1}"
            break

    # Install from requirements.txt, keeping OpenAI below the next breaking major.
    cmd = [sys.executable, "-m", "pip", "install", "-r", _REQUIREMENTS_TXT]
    if pinned_openai:
        cmd.append(pinned_openai)
    subprocess.check_call(cmd)

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

    # 3. Initialize Cognee DB (run migrations / create tables)
    try:
        from usr.plugins.memory_cognee.helpers.cognee_init import ensure_tables_sync
        ensure_tables_sync()
    except Exception as e:
        from helpers.print_style import PrintStyle
        PrintStyle.warning(f"Cognee DB init during install: {e}")

    # 4. Migrate FAISS data if present
    try:
        from usr.plugins.memory_cognee.helpers.faiss_migration import migrate
        migrate()
    except Exception as e:
        from helpers.print_style import PrintStyle
        PrintStyle.warning(f"FAISS migration skipped: {e}")
