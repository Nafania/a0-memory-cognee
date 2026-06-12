# Cognee Agent Zero Memory Flow

This document defines the runtime contract for `memory_cognee` inside Agent Zero.
It is based on the current plugin components and the required hot-path behavior:
normal chat must stay responsive while Cognee rebuild, consolidation, migration,
and graph repair work happens outside the user path.

## Goals and invariants

- `memory_cognee` replaces Agent Zero's builtin FAISS `_memory` plugin. Startup
  must try to disable `_memory` so both backends do not write or recall in
  parallel.
- Cognee is embedded in the Agent Zero process. All Cognee backend calls must go
  through the serialized `run_cognee_operation()` gate so in-process graph,
  vector, SQLite, and LanceDB state is not mutated concurrently.
- The user path must only do bounded, explainable work. It may do short query
  preparation, short search, explicit tool writes, and status checks. It must
  never do `cognee.cognify()`, `cognee.improve()`, FAISS bulk migration cleanup,
  full graph repair, or embedding/vector rebuild inline with a normal chat turn.
- Recall must prefer a known readable snapshot over freshness. A dataset can be
  `dirty` and still readable: in that case recall may continue from the last
  readable graph/vector snapshot while the background worker catches up.
- If a graph is unreadable, empty after data exists, failed without a readable
  snapshot, or actively unsafe to search, search must fail fast or skip with an
  explicit reason. It must not be reported as "no memories found".
- Writes must preserve source data first, then mark the dataset dirty for graph
  rebuild. New data is not guaranteed to be graph-searchable until background
  rebuild succeeds.
- Errors must be surfaced at the boundary where they matter: tool calls should
  return explicit unavailable/failure messages, auto-recall should log and
  continue the chat without memories, and operators should see readiness state in
  startup/background logs.
- Temporal Cognee processing is opt-in. `cognee_temporal_enabled` defaults to
  `false`; background rebuild and FAISS migration must use that setting.

## User-path synchronous operations

Allowed synchronous user-path work:

- `Memory.get(..., preload_knowledge=False)` to resolve the current dataset
  without importing knowledge.
- Auto-recall query extraction/preparation. Defaults:
  `memory_recall_query_prep_timeout_seconds: 10`,
  `memory_recall_history_len: 10000`,
  `memory_recall_interval: 3`.
- Cognee search through `Memory.SEARCH_TIMEOUT = 15` seconds. This 15 second
  budget applies both to waiting for the Cognee operation gate and to the search
  operation itself.
- Explicit `memory_load` tool search. It must call search with
  `raise_unavailable=True` and return `Memory search unavailable: ...` when the
  graph is blocked.
- Explicit `memory_save`, delete, and update tool writes. Direct Cognee add/delete
  operations use `Memory.WRITE_TIMEOUT = 30` seconds, then mark the dataset dirty.
- Session-memory write dispatch may be started from the turn end hook, but it
  must run as a background task, not inline with response generation.

Configured but bounded wrapper behavior:

- Auto-recall has an outer `memory_recall_timeout_seconds` default of `180`
  seconds. This is only an outer safety guard around the recall task. It is not a
  license for search to wait through rebuild or migration. Cognee search itself
  remains capped by `Memory.SEARCH_TIMEOUT = 15`.
- If `memory_recall_delayed` is `true`, the current turn may continue while recall
  finishes and a delay notice is added. If it is `false`, the wait hook may await
  the recall task, but rebuild blockers and Cognee gate contention must still
  resolve quickly through the 15 second search budget or explicit block reason.
- Lazy Cognee initialization on user path is a degraded fallback. Normal startup
  should initialize Cognee before chat traffic. If lazy init is reached, it must
  finish within its startup guard or raise/log; it must not hide an init failure
  as an empty memory result.

## Background/async operations

Background-only work:

- Cognee graph rebuild: `cognee.cognify()` and optional `cognee.improve()`.
- Embedding/vector rebuild, LanceDB vector purge, vector table optimization, and
  graph readiness verification after rebuild.
- Memory extraction from monologue end hooks, then queued writes through
  `MemoryWriteWorker`.
- LLM-based memory consolidation and keyword extraction.
- Session memory `cognee.remember()` writes.
- Legacy FAISS bulk migration and cleanup.

Default background budgets and pacing:

- `cognee_operation_timeout_seconds: 1800` for background Cognify/improve,
  preflight, readiness verification, and long session-memory Cognee calls.
- `cognee_rebuild_stale_after_seconds: 3600` before a non-running stale rebuild
  state is expired to failed and queued for retry.
- `cognee_cognify_interval: 5` minutes. Readable dirty datasets wait until this
  interval has elapsed after the last memory activity before rebuild.
- `cognee_cognify_debounce_seconds: 2` default for near-immediate scheduled
  rebuild checks.
- `cognee_rebuild_retry_min_seconds: 30` and
  `cognee_rebuild_retry_max_seconds: 300` defaults for exponential retry delays.
- `memory_consolidation_idle_seconds: 60`. Write/consolidation jobs and Cognee
  graph rebuild checks wait for memory activity to be idle before starting
  heavy work. For a new unreadable dirty dataset, this idle window still applies:
  recall may report the dataset pending, but foreground write/search must not be
  forced to wait behind an immediate background Cognify.
- `memory_consolidation_timeout_seconds: 30` for each consolidation job.
- `memory_consolidation_utility_timeout_seconds: 10` for utility-model calls used
  by consolidation.

Background work must be cancellable or time-bounded at its own layer, must update
dataset readiness, and must never make normal recall wait for the full background
budget.

## Startup flow

Startup is split into eager initialization and worker start:

1. `_20_init_cognee.py` calls `run_memory_cognee_init_a0_extension()`.
2. The plugin disables builtin `_memory` if it is enabled.
3. `configure_cognee()` sets Cognee storage env vars, logging, LLM config,
   embedding config, chunk sizing, and directory config before importing Cognee.
4. `init_cognee()` runs Cognee relational migrations with fallback table creation,
   startup vector payload migrations when safe, schema sync, legacy data-path
   rewrites, missing-data quarantine, LanceDB optimization, and graph readiness
   detection.
5. Startup graph readiness detection may reset Cognee pipeline status and mark
   affected datasets dirty with `preserve_readable=False` when data exists but
   graph is empty or unreadable.
6. Legacy FAISS migration runs after Cognee init and before the background worker.
   It is idempotent and may be incomplete; incomplete migration is degraded state,
   not a reason to run rebuild synchronously in startup.
7. Embedding config state is checked. If embedding config changed or unknown, all
   affected datasets are marked dirty/unreadable for background rebuild.
8. `_log_startup_readiness()` emits one operator-readable summary:
   `READY`, `DEGRADED`, `BLOCKED`, or `UNKNOWN`.
9. `_90_start_cognee_worker.py` starts `CogneeBackgroundWorker` only if Cognee
   initialization completed.

Startup may block Agent Zero process readiness while migrations/init run, but it
must not perform full rebuild/cognify as a substitute for the background worker.
If startup init fails, it should log `Cognee eager init failed ...`; the worker
must not start, and later user-path calls must surface init failure explicitly.

Startup readiness semantics:

- `READY`: readable graphs were verified and migration completed.
- `DEGRADED`: readable graphs exist, but FAISS migration is incomplete or some
  non-blocking startup maintenance failed.
- `BLOCKED`: required graph data is empty/unreadable, embedding rebuild is
  pending, or dirty/failed readiness has no readable snapshot.
- `UNKNOWN`: graph status could not be checked. This must be visible in logs and
  must not be treated as a proven-ready graph.

When embedding rebuild is pending, startup must not open graph/vector stores just
to report status; the background rebuild owns that state.

## Recall/search flow

Auto-recall flow:

1. The message-loop recall hook checks `memory_recall_enabled` and
   `memory_recall_interval`.
2. It records memory activity so readable dirty rebuilds can wait for idle time.
3. It prepares a query from the current user message and sanitized history.
   Tool payloads, stale `[EXTRAS]`, attachments, and system metadata must not
   pollute the recall query.
4. It resolves `Memory.get(..., preload_knowledge=False)`.
5. It searches `default` plus the current project/agent dataset.
6. Before loading/running Cognee search, it asks
   `CogneeBackgroundWorker.get_search_block_reason(datasets)`.
7. If blocked, auto-recall logs a specific heading such as
   `Memory rebuild pending; skipping recall` or
   `Memory rebuild failed; skipping recall`, then continues without injecting
   memories.
8. If searchable, it performs one Cognee search for memory and solution areas,
   capped by `Memory.SEARCH_TIMEOUT = 15`.
9. Results are split by area (`main`, `fragments`, `solutions`), limited by
   `memory_recall_memories_max_result: 5` and
   `memory_recall_solutions_max_result: 3`, then written to
   `loop_data.extras_persistent`.

Tool search flow:

- `memory_load` follows the same block check through
  `Memory.search_similarity_threshold(..., raise_unavailable=True)`.
- Blocked search returns an explicit tool response. It must not return the
  "not found" prompt unless Cognee search was actually allowed and returned zero
  documents.

Readable snapshot policy:

- `dirty` plus `readable=true` means recall can use the last readable snapshot.
  Freshly written memories may be absent until rebuild succeeds.
- `dirty` plus no readable snapshot blocks search with a pending reason.
- `failed` plus readable snapshot may continue serving the last readable snapshot
  while retry is pending.
- `failed` without readable snapshot blocks search with a failed reason including
  the stored rebuild error when available.
- An empty search result alone must not schedule a rebuild. Startup graph checks,
  explicit writes/deletes, embedding changes, and explicit resets own rebuild
  scheduling.

Active rebuild policy:

- Because Cognee is embedded and guarded by a global operation lock, an actively
  running rebuild may make graph/vector search unsafe or unavailable. In that
  case search must skip or time out quickly with `Cognee memory graph rebuild
  running ...`; it must not wait for the rebuild's 1800 second background budget.

## Write/memorize/consolidation flow

Explicit writes:

- `memory_save` calls `Memory.insert_text()` directly because the user explicitly
  requested a write. It uses the 30 second write budget, stores metadata sidecar
  data, returns a deterministic content hash id, and marks the dataset dirty.
- Deletes and updates use the 30 second write budget and mark the dataset dirty
  only after data was removed or updated.

Automatic memorization:

- Fragment and solution memorization run from monologue end hooks in
  `DeferredTask(THREAD_BACKGROUND)`.
- The utility model extracts candidate memories/solutions in the background.
- Each candidate is enqueued into `MemoryWriteWorker`. Enqueue must not run
  Cognee add/search inline.
- The queue is in-process and non-durable in current code; failed jobs are logged
  through `last_error`. If durable delivery is required, this needs a separate
  queue contract and implementation.

Write worker:

- The worker waits until `CogneeBackgroundWorker.is_memory_idle()` reports idle
  for `memory_consolidation_idle_seconds` before processing a job.
- If `memory_memorize_consolidation` is enabled (default `true`), the worker runs
  consolidation with the configured 30 second processing timeout and 10 second
  utility timeout.
- Consolidation searches similar memories with `raise_unavailable=True`. If the
  graph is blocked, consolidation must not trigger rebuild or wait for rebuild.
  It must either skip visibly, retry later, or fall back to direct insert by an
  explicit policy. It must not silently lose extracted memories.
- If consolidation is disabled, the worker performs simple dedup/write through
  `insert_with_simple_dedup()`.
- Any successful add marks the dataset dirty; the background worker later turns
  stored source data into graph-searchable memory after the memory idle window.

Session memory:

- Session turn memory runs in a background task via `safe_remember_session_turn()`.
- It stores the latest question/answer pair through `cognee.remember()` with
  `self_improvement=False`.
- It uses `cognee_operation_timeout_seconds` as its Cognee operation budget, but
  it must not block response generation. If it holds the Cognee operation gate,
  foreground search must still be bounded by the 15 second search gate timeout.

## Rebuild/cognify readiness state machine

Dataset readiness state is tracked per dataset with:

- `state`: `ready`, `dirty`, `rebuilding`, or `failed`.
- `readable`: whether a prior graph/vector snapshot may serve recall.
- `reason`: operator/user-facing reason for the state.
- `updated_at`, `last_ready_at`, `last_ready_reason`, and optional `progress`.

State transitions:

- Unknown/no state: search is not blocked by worker state alone. Startup graph
  readiness should populate real state when data exists.
- Startup verified non-empty graph: `ready`, `readable=true`.
- Insert/delete/preload on a readable dataset:
  `dirty`, `readable=true`, reason `Cognee memory graph rebuild pending`.
- Insert/delete/preload on a dataset without readable snapshot:
  `dirty`, `readable=false`, search blocked until rebuild.
- Full reset, unready startup graph, incompatible embedding config, or explicit
  pipeline reset: `dirty`, `readable=false`.
- Worker starts a dataset rebuild: `rebuilding`. Existing `readable=true` may be
  preserved as a last snapshot, but active rebuild can still make search skip
  quickly because of the global Cognee operation gate.
- Cognify/vector rebuild succeeds and readiness verification finds a non-empty
  graph: `ready`, `readable=true`, dirty flag cleared if the dataset version did
  not change during rebuild.
- Dataset changes during rebuild: return to `dirty`; preserve readability if a
  readable snapshot exists.
- Cognify/improve/vector rebuild/import/readiness verification fails:
  `failed`, retry scheduled, `_needs_pipeline_reset` set. If a readable snapshot
  existed, recall may continue from it; otherwise search blocks with failed
  reason.
- Rebuild is cancelled or exits before readiness update: mark affected datasets
  `failed`, keep them dirty, schedule retry.
- Rebuilding state older than `cognee_rebuild_stale_after_seconds` while worker
  is not running: mark `failed`, keep dirty, schedule retry. If worker is still
  running, do not clear active state.

Rebuild execution:

- Rebuilds process dirty datasets one at a time.
- Readable dirty datasets wait for idle time before rebuild.
- Unreadable dirty datasets run as soon as the worker can run; they do not wait
  for the idle interval because recall is blocked until they are rebuilt.
- `cognee.cognify()` runs with `temporal_cognify=false` by default.
- Optional `cognee.improve()` runs only after successful cognify and is skipped
  only for known empty-graph improve errors.
- Readiness verification after cognify/improve must confirm the dataset graph is
  non-empty when dataset data exists. A completed Cognee pipeline is not enough
  by itself.

## Legacy FAISS migration policy

- FAISS migration is startup maintenance and explicit operator maintenance, not
  normal chat work.
- Migration state lives in `usr/cognee_migration_state.json`. It tracks per-index
  status, migrated document ids, total count, errors, completion, and Cognee data
  directory.
- Migration is idempotent and resumable. Interrupted or partial migration must
  retry later without duplicating already migrated documents.
- Migration discovers global memory, project memory, and backup FAISS indices.
  Backup directories are not treated as primary sources when primary indices
  exist.
- Source FAISS data must be preserved. Completed primary FAISS directories are
  copied to `_faiss_backup`; startup must not delete legacy data.
- Missing FAISS dependencies or load errors mark the index error/partial and
  leave migration incomplete for retry. They must not block normal chat once
  Cognee has any readable graph.
- Startup `run_migration()` must import source documents only. Full cleanup of
  wrongly named old datasets is explicit maintenance (`cleanup=True`) and must
  not delete data during normal startup.
- If migration imports documents into Cognee and no readable graph exists for
  those datasets, the datasets should be considered dirty/unreadable and rebuilt
  by the background worker, not by startup.
- FAISS migration `run_cognify()` uses `cognee_temporal_enabled` and therefore
  defaults to non-temporal cognify.

## Failure/timeout/logging policy

- User-path search failure modes:
  - Blocked graph: return/log explicit pending/running/failed reason.
  - Cognee search exception: log `cognee.search failed: ...`; explicit tools
    return unavailable when requested through `raise_unavailable`.
  - OS/file-descriptor failure: log the OSError specifically; do not mask it as
    no results.
  - Timeout: log timeout and continue without memories for auto-recall; explicit
    tools return an error/unavailable response.
- Background rebuild failure modes:
  - Store `last_error`.
  - Mark dataset `failed`.
  - Keep dirty and schedule retry.
  - Reset pipeline status before retry when required.
  - Log a readiness summary that says `BLOCKED` while recall may be unavailable.
- Startup failure modes:
  - Cognee import/init/migration/readiness failures are logged with type and
    message.
  - Init failure prevents worker start.
  - Degraded migration or maintenance failures must be visible but must not force
    full rebuild on user path.
- Logging must redact API keys and secret-looking values. Cognee/LiteLLM logs
  default to WARNING; raw request loggers stay WARNING even in plugin debug mode.
- Error handling must distinguish:
  - "search allowed and found nothing";
  - "search skipped because graph is rebuilding";
  - "search blocked because graph rebuild failed";
  - "Cognee operation failed";
  - "startup migration incomplete".

## What must never happen

- Normal chat path waits for `cognee.cognify()`, `cognee.improve()`, FAISS bulk
  migration, graph repair, or embedding rebuild.
- Auto-recall waits minutes for a rebuild to finish.
- A failed/unreadable graph is reported as "no memories found".
- Dirty unreadable state is treated as safe to search.
- A readable dirty state blocks recall just because freshness is pending, except
  when an active in-process rebuild makes search unsafe and skip-fast behavior is
  required.
- `cognee_temporal_enabled` defaults to `true` or temporal Cognify is hardcoded in
  rebuild/migration.
- Startup deletes legacy FAISS data or wrongly named Cognee datasets without an
  explicit cleanup operation.
- Empty search results alone schedule a rebuild.
- The builtin `_memory` plugin writes/recalls alongside `memory_cognee`.
- Background write/consolidation failures disappear without logs/status.
- Logs expose API keys, OAuth tokens, or raw secret-bearing LLM kwargs.
- Cognee operation gate waits indefinitely.

## Verification checklist

Run baseline checks:

- `python3 -m unittest discover -s tests`
- `git diff --check`

Targeted unit coverage should include:

- `tests/test_memory_dirty_marking.py`
  - blocked search skips Cognee before import/search;
  - `SearchUnavailable` is raised for explicit tool search;
  - search uses the 15 second user-path timeout;
  - writes use the 30 second write timeout and mark dirty.
- `tests/test_recall_single_search.py`
  - auto-recall uses one bounded search;
  - query prep timeout falls back to sanitized message/history;
  - configured outer recall timeout wraps the recall task;
  - tool payloads and stale extras do not pollute queries.
- `tests/test_memory_write_worker.py`
  - enqueue starts background worker without inline Cognee processing;
  - worker waits for idle before consolidation/write.
- `tests/test_cognee_background.py`
  - dirty readable datasets remain searchable before rebuild;
  - dirty unreadable datasets block search and rebuild immediately;
  - readable datasets wait for idle interval;
  - background rebuild uses operation timeout and retry state;
  - temporal Cognify defaults false and only enables by config;
  - failed, cancelled, hung, and stale rebuild states are marked failed and
    retried;
  - readiness verification rejects empty graphs with data.
- `tests/test_startup_faiss_migration.py`
  - startup order is disable `_memory`, configure, init, migrate, then worker
    start;
  - startup does not synchronously rebuild dirty datasets;
  - readable dirty startup state does not warn that rebuild is required before
    recall.
- `tests/test_cognee_init_graph_purge.py`
  - startup readiness logs READY/DEGRADED/BLOCKED/UNKNOWN correctly;
  - embedding rebuild pending avoids opening graph/vector stores;
  - unready graphs reset pipeline status and mark datasets dirty.
- `tests/test_faiss_migration.py`
  - migration is idempotent/resumable;
  - migration Cognify uses `cognee_temporal_enabled=false` by default.

Manual/runtime checks:

- Start with a verified readable graph and one new memory write. Confirm logs say
  rebuild is pending for a readable dataset and recall still returns old
  snapshot results.
- Start with data but an empty/unreadable graph. Confirm startup logs BLOCKED and
  `memory_load` returns explicit unavailable text.
- Force `cognee.cognify()` to hang or time out. Confirm user recall does not wait
  for the background timeout, rebuild state becomes failed, and retry is
  scheduled.
- Run with `cognee_temporal_enabled` unset. Confirm rebuild logs
  `temporal_cognify=False`.
- Run with `cognee_temporal_enabled=true`. Confirm only then rebuild logs
  `temporal_cognify=True`.
- Simulate incomplete FAISS migration. Confirm startup logs DEGRADED/failed
  migration details but does not delete legacy data and does not run full Cognify
  inline.
- Inspect logs under debug mode and confirm API keys/secrets are redacted.
