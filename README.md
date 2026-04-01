# _memory_cognee — Cognee-backed Memory Plugin for Agent Zero

Replaces the builtin FAISS-based `_memory` plugin with a [Cognee](https://github.com/topoteretes/cognee) backend, providing knowledge-graph-powered memory with background processing, consolidation, and feedback.

## Features

- **Cognee knowledge graph** — memories are indexed via Cognee's graph-completion search instead of raw FAISS vector similarity.
- **Background cognify** — a background worker periodically runs `cognee.cognify()` and `cognee.memify()` on dirty datasets.
- **Memory feedback** — durable disk queue forwards user feedback (positive/negative) to Cognee's `add_feedback` API.
- **FAISS migration** — on install, existing FAISS data under `usr/memory/` is automatically migrated to Cognee.
- **Full dashboard** — WebUI memory dashboard with search, edit, bulk delete, and export.

## Installation

1. Copy (or symlink) this directory into your Agent Zero `plugins/` folder as `_memory_cognee`.
2. Enable the plugin in **Settings → Plugins**.
3. The `hooks.py` install hook will:
   - `pip install cognee[fastembed]`
   - Disable the builtin `_memory` plugin
   - Migrate any existing FAISS data

## Configuration

All settings are in `default_config.yaml` and can be overridden per-project or per-agent in the Agent Zero settings UI.

## License

Same license as the parent Agent Zero project.
