# memory_cognee — Cognee-backed Memory Plugin for Agent Zero

Replaces the builtin FAISS-based `_memory` plugin with an embedded [Cognee](https://github.com/topoteretes/cognee) backend (runs in-process, no external server needed), providing knowledge-graph-powered memory with background processing, consolidation, and feedback.

## Features

- **Embedded Cognee engine** — Cognee runs in-process alongside Agent Zero, no separate service or Docker container required.
- **Cognee knowledge graph** — memories are indexed via Cognee's graph-completion search instead of raw FAISS vector similarity.
- **Interactive graph visualization** — explore the knowledge graph visually with Cytoscape.js, click nodes, navigate relationships.
- **Background processing** — a background worker periodically runs `cognee.cognify()` and `cognee.improve()` on dirty datasets.
- **Memory feedback** — durable disk queue forwards user feedback (positive/negative) to Cognee's feedback API.
- **FAISS migration** — on install, existing FAISS data under `usr/memory/` is automatically migrated to Cognee.
- **Full dashboard** — WebUI memory dashboard with search, dynamic area filtering, bulk delete, export, and knowledge graph view.
- **Cognee 1.0.7** — uses Cognee V2 APIs where beneficial (`improve()`, `forget()`, `run_startup_migrations()`) and is exact-pinned because Cognee storage and dependency migrations can affect production data.

## Installation

### Via UI (recommended)

1. Open **Settings → Plugins → Install**.
2. Switch to the **Git URL** tab.
3. Enter: `https://github.com/Nafania/a0-memory-cognee`
4. Click **Install**. The plugin installer will clone the repo into `usr/plugins/`, run `hooks.py`, and enable the plugin automatically.

### Via API

```bash
curl -X POST "$BASE/api/plugins/_plugin_installer/plugin_install" \
  -H "Content-Type: application/json" \
  -H "X-CSRF-Token: $TOKEN" \
  -H "Origin: $ORIGIN" \
  -d '{"action": "install_git", "git_url": "https://github.com/Nafania/a0-memory-cognee"}'
```

### Manual

```bash
git clone https://github.com/Nafania/a0-memory-cognee /a0/usr/plugins/memory_cognee
```

Then restart Agent Zero or toggle the plugin off/on in **Settings → Plugins**.

### What happens on install

The `hooks.py` install hook will:
- `pip install cognee[fastembed]==1.0.7` (exact-pinned in `requirements.txt`)
- Disable the builtin `_memory` plugin
- Migrate any existing FAISS data

## Configuration

All settings are in `default_config.yaml` and can be overridden per-project or per-agent in the Agent Zero settings UI.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
