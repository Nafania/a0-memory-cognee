import subprocess
import sys

def install():
    # 1. Snapshot openai version before cognee install to prevent breakage
    result = subprocess.run(
        [sys.executable, "-m", "pip", "show", "openai"],
        capture_output=True, text=True
    )
    pinned_openai = ""
    for line in result.stdout.splitlines():
        if line.startswith("Version:"):
            ver = line.split(":", 1)[1].strip()
            major = int(ver.split(".")[0])
            pinned_openai = f"openai=={ver}" if major < 2 else f"openai<{major + 1}"
            break

    # Install cognee while keeping openai compatible with litellm
    cmd = [sys.executable, "-m", "pip", "install", "cognee[fastembed]"]
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

    # 3. Migrate FAISS data if present
    try:
        import os, importlib.util
        plugin_dir = os.path.dirname(os.path.abspath(__file__))
        spec = importlib.util.spec_from_file_location(
            "faiss_migration",
            os.path.join(plugin_dir, "helpers", "faiss_migration.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.migrate()
    except Exception as e:
        from helpers.print_style import PrintStyle
        PrintStyle.warning(f"FAISS migration skipped: {e}")
