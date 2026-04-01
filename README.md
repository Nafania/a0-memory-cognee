# _memory_cognee — Cognee-backed Memory Plugin for Agent Zero

Replaces the builtin FAISS-based `_memory` plugin with a [Cognee](https://github.com/topoteretes/cognee) backend, providing knowledge-graph-powered memory with background processing, consolidation, and feedback.

## Features

- **Cognee knowledge graph** — memories are indexed via Cognee's graph-completion search instead of raw FAISS vector similarity.
- **Background cognify** — a background worker periodically runs `cognee.cognify()` and `cognee.memify()` on dirty datasets.
- **Memory feedback** — durable disk queue forwards user feedback (positive/negative) to Cognee's `add_feedback` API.
- **FAISS migration** — on install, existing FAISS data under `usr/memory/` is automatically migrated to Cognee.
- **Full dashboard** — WebUI memory dashboard with search, edit, bulk delete, and export.

## Installation

### Via UI (recommended)

1. Open **Settings → Plugins → Install**.
2. Switch to the **Git URL** tab.
3. Enter the repository URL for this plugin.
4. Click **Install**. The plugin installer will clone the repo into `usr/plugins/`, run `hooks.py`, and enable the plugin automatically.

### Via API

```bash
curl -X POST "$BASE/api/plugins/_plugin_installer/plugin_install" \
  -H "Content-Type: application/json" \
  -H "X-CSRF-Token: $TOKEN" \
  -H "Origin: $ORIGIN" \
  -d '{"action": "install_git", "git_url": "https://github.com/<user>/a0-memory-cognee"}'
```

### Manual (last resort)

```bash
git clone https://github.com/<user>/a0-memory-cognee /a0/usr/plugins/_memory_cognee
```

Then restart Agent Zero or toggle the plugin off/on in **Settings → Plugins**.

### What happens on install

The `hooks.py` install hook will:
- `pip install cognee[fastembed]`
- Disable the builtin `_memory` plugin
- Migrate any existing FAISS data

## Configuration

All settings are in `default_config.yaml` and can be overridden per-project or per-agent in the Agent Zero settings UI.

## License

Same license as the parent Agent Zero project.
