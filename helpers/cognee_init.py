import os
from typing import Any, TypeVar

from helpers import dotenv, files
from helpers.settings import get_settings
from helpers.print_style import PrintStyle

T = TypeVar("T")

_COGNEE_DEFAULTS: dict[str, Any] = {
    "cognee_search_type": "GRAPH_COMPLETION",
    "cognee_search_types": "GRAPH_COMPLETION",
    "cognee_multi_search_enabled": True,
    "cognee_cognify_interval": 5,
    "cognee_cognify_after_n_inserts": 10,
    "cognee_temporal_enabled": True,
    "cognee_memify_enabled": True,
    "cognee_feedback_enabled": True,
    "cognee_session_cache": "filesystem",
    "cognee_data_dir": "usr/cognee",
    "cognee_chunk_size": 512,
    "cognee_chunk_overlap": 50,
    "cognee_search_system_prompt": "",
}

_PROVIDER_MAP: dict[str, str] = {
    "openrouter": "openrouter",
    "huggingface": "huggingface",
    "openai": "openai",
    "anthropic": "anthropic",
    "gemini": "gemini",
    "ollama": "ollama",
    "lmstudio": "custom",
    # Agent Zero's OAuth wrapper for OpenAI/Codex (see plugins/_oauth/conf/model_providers.yaml).
    # Agent Zero proxies requests via 127.0.0.1/oauth/codex/v1 and swaps the dummy
    # api_key="oauth" for the real OAuth token at the proxy layer, so cognee just
    # needs to treat it as a plain OpenAI provider.
    "codex_oauth": "openai",
}

_EMBED_DIMENSIONS: dict[str, int] = {
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "nomic-embed-text:latest": 768,
}

_configured = False
_init_done = False
_cognee_module = None
_search_type_class = None


def get_cognee_setting(name: str, default: T) -> T:
    env_key = f"A0_SET_{name}"
    env_value = dotenv.get_dotenv_value(env_key, dotenv.get_dotenv_value(env_key.upper(), None))
    if env_value is None:
        return _COGNEE_DEFAULTS.get(name, default)  # type: ignore
    try:
        if isinstance(default, bool):
            return env_value.strip().lower() in ("true", "1", "yes", "on")  # type: ignore
        elif isinstance(default, int):
            return type(default)(env_value.strip())  # type: ignore
        elif isinstance(default, str):
            return str(env_value).strip()  # type: ignore
        return default
    except (ValueError, TypeError):
        return _COGNEE_DEFAULTS.get(name, default)  # type: ignore


def _map_provider(a0_provider: str) -> str:
    return _PROVIDER_MAP.get(a0_provider.lower(), a0_provider)


def _resolve_provider_with_defaults(
    a0_provider: str, model_type: str = "chat"
) -> tuple[str, dict[str, str]]:
    """Resolve Agent Zero provider id -> (cognee/litellm provider, default kwargs).

    Agent Zero stores only ``provider`` and ``name`` in _model_config; per-provider
    defaults like ``api_base`` and the OAuth dummy ``api_key`` live in
    ``conf/model_providers.yaml`` and are merged only at runtime inside
    ``_merge_provider_defaults`` (see Agent Zero ``models.py``).

    This mirrors that merge for cognee so providers such as ``codex_oauth``
    (which proxies through ``http://127.0.0.1/oauth/codex/v1`` and uses a dummy
    key of ``"oauth"``) work out of the box.

    Returns:
        (final_provider_name, extra_kwargs) where extra_kwargs may contain
        ``api_base`` and ``api_key`` defaults from the provider registry.
    """
    if not a0_provider:
        return "", {}

    provider_key = a0_provider.lower()
    # Our static fallback map handles providers not in Agent Zero's registry.
    mapped = _PROVIDER_MAP.get(provider_key, provider_key)
    extra: dict[str, str] = {}

    try:
        from helpers.providers import get_provider_config  # type: ignore

        cfg = get_provider_config(model_type, provider_key)
        if isinstance(cfg, dict):
            litellm_provider = str(cfg.get("litellm_provider") or "").strip().lower()
            if litellm_provider:
                mapped = litellm_provider
            provider_kwargs = cfg.get("kwargs") if isinstance(cfg, dict) else None
            if isinstance(provider_kwargs, dict):
                for k in ("api_base", "api_key"):
                    v = provider_kwargs.get(k)
                    if isinstance(v, str) and v:
                        extra[k] = v
    except Exception as e:
        PrintStyle.warning(
            f"Could not load Agent Zero provider registry for '{a0_provider}' "
            f"(model_type={model_type}): {e}. Falling back to static map."
        )

    return mapped, extra


def _get_api_key(provider: str, api_keys: dict[str, str] | None = None) -> str:
    dotenv.load_dotenv()
    key = dotenv.get_dotenv_value(f"API_KEY_{provider.upper()}")
    if key:
        return key
    if api_keys is not None:
        return api_keys.get(provider, "") or ""
    return get_settings().get("api_keys", {}).get(provider, "") or ""


def configure_cognee() -> None:
    global _configured, _cognee_module, _search_type_class
    if _configured:
        return

    dotenv.load_dotenv()
    settings = get_settings()

    # --- Storage directories (MUST be set BEFORE import cognee) ---
    data_dir = files.get_abs_path(get_cognee_setting("cognee_data_dir", "usr/cognee"))
    os.makedirs(data_dir, exist_ok=True)

    data_storage = os.path.join(data_dir, "data_storage")
    system_storage = os.path.join(data_dir, "cognee_system")
    cache_storage = os.path.join(data_dir, "cognee_cache")

    os.makedirs(data_storage, exist_ok=True)
    os.makedirs(system_storage, exist_ok=True)
    os.makedirs(cache_storage, exist_ok=True)
    # Cognee expects a `databases/` subdir inside system_storage for SQLite files
    os.makedirs(os.path.join(system_storage, "databases"), exist_ok=True)

    os.environ["DATA_ROOT_DIRECTORY"] = data_storage
    os.environ["SYSTEM_ROOT_DIRECTORY"] = system_storage
    os.environ["CACHE_ROOT_DIRECTORY"] = cache_storage
    os.environ["DB_PROVIDER"] = "sqlite"
    os.environ["DB_NAME"] = "cognee_db"
    os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] = "true"
    os.environ["CACHING"] = "true"
    os.environ["CACHE_ADAPTER"] = get_cognee_setting("cognee_session_cache", "filesystem")

    PrintStyle.standard(f"Cognee env configured: system={system_storage}, data={data_storage}")

    # --- Now safe to import cognee (env vars are set) ---
    try:
        import cognee
        from cognee import SearchType
    except Exception as e:
        import traceback
        PrintStyle.error(f"Cognee import failed — memory features will not work: {e}")
        PrintStyle.error(traceback.format_exc())
        return

    _cognee_module = cognee
    _search_type_class = SearchType

    # --- Read model config from _model_config plugin ---
    from helpers import plugins as _plugins
    import models as _models

    model_cfg = _plugins.get_plugin_config("_model_config") or {}
    util_cfg = model_cfg.get("utility_model", {})
    embed_cfg = model_cfg.get("embedding_model", {})

    util_provider = util_cfg.get("provider", "")
    util_model = util_cfg.get("name", "")
    if not util_provider or not util_model:
        PrintStyle.warning("Cognee: utility_model not configured yet, skipping LLM/embedding setup")
        _configured = True
        return

    # --- LLM ---
    llm_provider, llm_extra = _resolve_provider_with_defaults(util_provider, "chat")
    # User-set values in _model_config win over registry defaults (same as Agent Zero's
    # _merge_provider_defaults which uses setdefault).
    llm_api_key = (
        util_cfg.get("api_key", "")
        or _models.get_api_key(util_provider)
        or llm_extra.get("api_key", "")
    )
    util_api_base = util_cfg.get("api_base", "") or llm_extra.get("api_base", "")

    try:
        cognee.config.set_llm_config({
            "llm_provider": llm_provider,
            "llm_model": util_model,
            "llm_api_key": llm_api_key,
        })
        if util_api_base:
            cognee.config.set_llm_endpoint(util_api_base)
    except Exception as e:
        PrintStyle.error(f"cognee.config LLM setup failed, falling back to env vars: {e}")
        os.environ["LLM_PROVIDER"] = llm_provider
        os.environ["LLM_MODEL"] = util_model
        os.environ["LLM_API_KEY"] = llm_api_key
        if util_api_base:
            os.environ["LLM_API_BASE"] = util_api_base

    # --- Embedding ---
    raw_embed_provider = embed_cfg.get("provider", "")
    embed_provider, embed_extra = _resolve_provider_with_defaults(raw_embed_provider, "embedding")
    embed_model = embed_cfg.get("name", "")
    embed_api_key = (
        embed_cfg.get("api_key", "")
        or _models.get_api_key(raw_embed_provider)
        or embed_extra.get("api_key", "")
    )
    embed_api_base = embed_cfg.get("api_base", "") or embed_extra.get("api_base", "")

    if embed_provider in ("huggingface", "fastembed"):
        os.environ["EMBEDDING_PROVIDER"] = "fastembed"
        os.environ["EMBEDDING_MODEL"] = embed_model
        os.environ["EMBEDDING_DIMENSIONS"] = str(_EMBED_DIMENSIONS.get(embed_model, 384))
    else:
        os.environ["EMBEDDING_PROVIDER"] = embed_provider
        if "/" not in embed_model or not embed_model.startswith(embed_provider):
            embed_model = f"{embed_provider}/{embed_model}"
        os.environ["EMBEDDING_MODEL"] = embed_model
    os.environ["EMBEDDING_API_KEY"] = embed_api_key
    if embed_api_base:
        os.environ["EMBEDDING_API_BASE"] = embed_api_base

    # --- Chunking ---
    try:
        cognee.config.set_chunk_size(get_cognee_setting("cognee_chunk_size", 512))
        cognee.config.set_chunk_overlap(get_cognee_setting("cognee_chunk_overlap", 50))
    except Exception as e:
        PrintStyle.error(f"cognee.config chunk setup failed: {e}")

    # --- Apply directory config via cognee API (0.5.x dropped the set_ prefix) ---
    try:
        cognee.config.data_root_directory(data_storage)
        cognee.config.system_root_directory(system_storage)
    except Exception as e:
        PrintStyle.error(f"cognee.config directory setup failed: {e}")

    _configured = True


async def _create_db_tables():
    try:
        from cognee.run_migrations import run_startup_migrations

        await run_startup_migrations()
    except BaseException as mig_err:
        # Must catch BaseException: Cognee's run_migrations() calls sys.exit(1)
        # on alembic failure, raising SystemExit which is BaseException, not Exception.
        PrintStyle.error(f"Cognee run_startup_migrations failed ({type(mig_err).__name__}), trying create_db_and_tables: {mig_err}")
        try:
            from cognee.infrastructure.databases.relational import create_db_and_tables

            await create_db_and_tables()
        except Exception as e:
            PrintStyle.error(f"Cognee DB table creation failed: {e}")
            return

    _sync_missing_columns()
    affected_datasets = _purge_stale_graph_dbs()
    if affected_datasets:
        await _reset_cognify_status_for_datasets(affected_datasets)
    PrintStyle.standard("Cognee DB tables initialized")


def _purge_stale_graph_dbs() -> set[str]:
    """Detect and wipe stale Kuzu/Ladybug graph DBs that can't be auto-migrated.

    After upgrading cognee 0.5.x -> 1.0.1, the bundled graph backend switched from
    old Kuzu to Ladybug (renamed Kuzu with a new storage format). Cognee's built-in
    auto-migration only covers Kuzu 0.7.0-0.11.0 transitions; the Kuzu->Ladybug
    jump is not handled and surfaces as a flood of errors like:

        Failed to initialize Ladybug database: Could not map version_code to proper Ladybug version.
        Error: cognee.search failed: ...
        Error: Cognee insert failed: ...
        Error: Direct memory insertion failed: list index out of range

    The underlying data in SQLite + LanceDB is intact -- only the graph layer is
    unreadable. The safest recovery is to delete the unreadable graph files and
    reset the cognify pipeline status so the graph rebuilds on next cognify().

    Returns:
        Non-empty set of affected graph markers if any graph DB was purged.
        Caller should reset pipeline run status so cognify actually re-runs
        (cognify_pipeline skips datasets marked DATASET_PROCESSING_COMPLETED).
        The markers are intentionally best-effort because Cognee graph paths are
        not documented as stable across versions.
    """
    affected_dataset_ids: set[str] = set()
    try:
        system_storage = os.environ.get("SYSTEM_ROOT_DIRECTORY", "")
        if not system_storage:
            return affected_dataset_ids
        databases_dir = os.path.join(system_storage, "databases")
        if not os.path.isdir(databases_dir):
            return affected_dataset_ids

        purged_count = 0

        for graph_path in _iter_ladybug_graph_candidates(databases_dir):
            purged, marker = _purge_graph_db_if_unreadable(graph_path, databases_dir)
            if purged:
                purged_count += 1
                affected_dataset_ids.add(marker)

        if purged_count:
            PrintStyle.warning(
                f"Purged {purged_count} stale graph DB(s). Will reset cognify_pipeline "
                f"status for affected datasets so the graph rebuilds."
            )
    except Exception as e:
        PrintStyle.error(f"Stale graph DB detection failed (non-fatal): {e}")

    return affected_dataset_ids


def _iter_ladybug_graph_candidates(databases_dir: str) -> list[str]:
    """Return likely local Ladybug/Kuzu graph DB roots under Cognee databases dir."""
    candidates: set[str] = set()
    for root, dirs, files_in_dir in os.walk(databases_dir, topdown=True):
        is_graph_dir = (
            "catalog.kz" in files_in_dir
            and os.path.abspath(root) != os.path.abspath(databases_dir)
        )
        if is_graph_dir:
            candidates.add(root)
            # Directory graph DB: the catalog belongs to this root, not a
            # separate file-based DB candidate.
            dirs[:] = []
            continue

        for file_name in files_in_dir:
            # File-based Kuzu/Ladybug DBs are supported by Cognee's own migration
            # helper. Avoid lock/WAL files and obvious relational/vector files.
            if file_name.endswith((".lock", ".wal", ".sqlite", ".db", ".json", ".log")):
                continue
            path = os.path.join(root, file_name)
            if _read_ladybug_version_code(path) is not None:
                candidates.add(path)

        # If a directory is a graph DB, do not report its internal files as
        # separate candidates; the root deletion handles the whole database.
        dirs[:] = [
            d
            for d in dirs
            if os.path.join(root, d) not in candidates
        ]

    return sorted(candidates, key=lambda p: (p.count(os.sep), p))


def _read_ladybug_version_code(graph_path: str) -> int | None:
    """Read Kuzu/Ladybug storage code from a graph dir or file, if it looks like one."""
    try:
        import struct

        version_file_path = (
            os.path.join(graph_path, "catalog.kz") if os.path.isdir(graph_path) else graph_path
        )
        if not os.path.isfile(version_file_path):
            return None

        with open(version_file_path, "rb") as f:
            magic = f.read(4)
            if not (magic.startswith(b"KUZ") or magic == b"LBUG"):
                return None
            data = f.read(8)
        if len(data) < 8:
            return None
        return struct.unpack("<Q", data)[0]
    except Exception:
        return None


def _is_graph_readable_by_current_ladybug(graph_path: str) -> bool:
    """Return True if the installed Ladybug can open this graph DB as-is."""
    try:
        import ladybug

        db = ladybug.Database(graph_path)
        try:
            # init_database creates schema when needed and is what Cognee calls
            # before issuing queries. If this succeeds, the graph is not stale.
            db.init_database()
        finally:
            close = getattr(db, "close", None)
            if callable(close):
                close()
        return True
    except Exception:
        return False


def _is_known_legacy_ladybug_code(version_code: int | None) -> bool:
    # Known storage-version codes supported by cognee's Kuzu migration table.
    return version_code in {34, 35, 36, 37, 38, 39}


def _purge_graph_db_if_unreadable(graph_path: str, databases_dir: str) -> tuple[bool, str]:
    version_code = _read_ladybug_version_code(graph_path)
    if version_code is None:
        return False, ""

    if _is_graph_readable_by_current_ladybug(graph_path):
        return False, ""

    # If Cognee's migrator knows the legacy code, let Cognee attempt its normal
    # export/import path instead of deleting potentially recoverable graph data.
    if _is_known_legacy_ladybug_code(version_code):
        PrintStyle.warning(
            f"Graph DB is not directly readable by current Ladybug but has known "
            f"legacy version_code={version_code}; leaving it for Cognee migration: {graph_path}"
        )
        return False, ""

    try:
        import shutil

        if os.path.isdir(graph_path):
            shutil.rmtree(graph_path, ignore_errors=True)
        elif os.path.exists(graph_path):
            os.remove(graph_path)

        for sibling in (graph_path + ".lock", graph_path + ".wal"):
            if os.path.isdir(sibling):
                shutil.rmtree(sibling, ignore_errors=True)
            elif os.path.exists(sibling):
                os.remove(sibling)

        relative_path = os.path.relpath(graph_path, databases_dir)
        marker = (
            relative_path.split(os.sep, 1)[0]
            if relative_path and relative_path != "."
            else "__global_graph__"
        )
        PrintStyle.warning(
            f"Purged unreadable stale graph DB (unsupported version_code={version_code}, "
            f"marker={marker}): {graph_path}"
        )
        return True, marker
    except Exception as e:
        PrintStyle.error(f"Failed to purge stale graph DB {graph_path}: {e}")
        return False, ""


async def _reset_cognify_status_for_datasets(dataset_ids: set[str]) -> None:
    """Reset cognify_pipeline status for ALL datasets (robust rebuild after purge).

    We reset status for every dataset (not just those whose graph DB folder we
    could identify), because:
      1. Cognee's on-disk layout isn't documented as stable -- a purged folder
         name may or may not map 1:1 to a dataset UUID.
      2. After a major backend upgrade, it's safer to re-cognify everything
         than to silently leave some datasets with a stale graph.
      3. Data in SQLite + LanceDB is unchanged, so re-cognify is idempotent
         and non-destructive.

    Also marks every dataset as dirty so the background worker picks them up
    without waiting for a new insert.
    """
    if not dataset_ids:
        return
    try:
        import cognee
        from cognee.modules.pipelines.layers.reset_dataset_pipeline_run_status import (
            reset_dataset_pipeline_run_status,
        )
        from cognee.modules.users.methods import get_default_user
    except Exception as e:
        PrintStyle.error(
            f"Cannot import cognee reset helpers: {e}. "
            f"Graph will NOT auto-rebuild -- re-add data or call cognify manually."
        )
        return

    try:
        user = await get_default_user()
    except Exception as e:
        PrintStyle.error(
            f"Could not resolve default cognee user: {e}. Graph will NOT auto-rebuild."
        )
        return

    try:
        all_datasets = await cognee.datasets.list_datasets()
    except Exception as e:
        PrintStyle.error(
            f"Could not list datasets to reset cognify status: {e}. "
            f"Graph will NOT auto-rebuild -- re-add data or call cognify manually."
        )
        return

    reset_count = 0
    dataset_names: list[str] = []
    for ds in all_datasets:
        try:
            await reset_dataset_pipeline_run_status(
                ds.id, user, pipeline_names=["cognify_pipeline"]
            )
            reset_count += 1
            if getattr(ds, "name", None):
                dataset_names.append(ds.name)
        except Exception as e:
            PrintStyle.warning(
                f"Failed to reset cognify_pipeline status for dataset "
                f"{getattr(ds, 'name', ds.id)}: {e}"
            )

    if reset_count:
        PrintStyle.standard(
            f"Reset cognify_pipeline status for {reset_count} dataset(s). "
            f"Graph will rebuild on next cognify()."
        )

    try:
        from .cognee_background import CogneeBackgroundWorker

        worker = CogneeBackgroundWorker.get_instance()
        for name in dataset_names:
            worker.mark_dirty(name)
        if dataset_names:
            PrintStyle.standard(
                f"Marked {len(dataset_names)} dataset(s) dirty for background rebuild: {dataset_names}"
            )
    except Exception as e:
        PrintStyle.warning(
            f"Could not mark datasets dirty (graph will rebuild on next insert instead): {e}"
        )


def _sync_missing_columns():
    """Compare Cognee ORM models against actual DB schema and add missing columns.

    TEMPORARY WORKAROUND for https://github.com/topoteretes/cognee/issues/TBD
    Cognee 0.5.7 added importance_weight to the Data ORM model (PR #2447) but
    shipped no alembic migration, so run_startup_migrations() never adds the column to
    existing databases.  This function generically detects columns present in
    the ORM but absent from the DB and adds them via DDL.

    TODO: remove once Cognee ships a proper alembic migration for importance_weight
    (track upstream fix, then drop this function and its call site).
    """
    try:
        from cognee.infrastructure.databases.relational.ModelBase import Base
        from sqlalchemy import create_engine, inspect as sa_inspect, text

        db_path = os.path.join(
            os.environ.get("SYSTEM_ROOT_DIRECTORY", ""),
            "databases",
            os.environ.get("DB_NAME", "cognee_db"),
        )
        if not os.path.exists(db_path):
            return
        engine = create_engine(f"sqlite:///{db_path}")
        inspector = sa_inspect(engine)

        for table_name, table_obj in Base.metadata.tables.items():
            if not inspector.has_table(table_name):
                continue
            existing = {c["name"] for c in inspector.get_columns(table_name)}
            for col in table_obj.columns:
                if col.name in existing:
                    continue
                col_type = col.type.compile(dialect=engine.dialect)
                parts = [f"ALTER TABLE [{table_name}] ADD COLUMN [{col.name}] {col_type}"]
                if not col.nullable:
                    parts.append("NOT NULL")
                if col.server_default is not None:
                    parts.append(f"DEFAULT {col.server_default.arg}")
                elif col.nullable:
                    parts.append("DEFAULT NULL")
                ddl = " ".join(parts)
                with engine.begin() as conn:
                    conn.execute(text(ddl))
                PrintStyle.standard(f"Schema sync: added {table_name}.{col.name} ({col_type})")
    except Exception as e:
        PrintStyle.error(f"Schema column sync failed: {e}")


async def init_cognee() -> None:
    """One-time startup initialization. Idempotent — safe to call multiple times."""
    global _init_done
    if _init_done:
        return
    configure_cognee()
    await _create_db_tables()
    _init_done = True
    PrintStyle.standard("Cognee fully initialized")


def ensure_tables_sync() -> None:
    """Run init_cognee() from a sync context (e.g. hooks.install).

    Works whether or not an event loop is already running:
    - No loop → asyncio.run()
    - Loop running (web server) → new thread with its own loop
    """
    if _init_done:
        return
    import asyncio
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop — safe to use asyncio.run()
        try:
            asyncio.run(init_cognee())
        except BaseException as e:
            PrintStyle.error(f"ensure_tables_sync (asyncio.run): {type(e).__name__}: {e}")
        return

    import threading

    def _run():
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(init_cognee())
        except BaseException as e:
            PrintStyle.error(f"ensure_tables_sync (thread): {type(e).__name__}: {e}")
        finally:
            loop.close()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=60)


def run_memory_cognee_init_a0_extension() -> None:
    """Entry for Agent Zero `init_a0` / `end` extensions.

    Upstream `run_ui.run()` calls `init_a0()` before starting the server; extension
    folders are `_functions/<run_ui.init_a0.__module__>/init_a0/end/` per
    `helpers.extension.extensible` (see agent0ai/agent-zero). Official Docker starts
    `python run_ui.py`, so `__module__` is usually `__main__`; if `run_ui` is imported
    as a module, use the duplicate extension under `_functions/run_ui/...`.
    """
    import asyncio

    try:
        configure_cognee()
        asyncio.run(init_cognee())
        from .cognee_background import CogneeBackgroundWorker

        CogneeBackgroundWorker.get_instance().start()
    except BaseException as e:
        # BaseException: asyncio.run() re-raises SystemExit from run_migrations
        PrintStyle.error(f"Cognee eager init failed ({type(e).__name__}): {e}")


def get_cognee():
    """Get initialized cognee module. Lazy-initializes on first call if needed."""
    if _cognee_module is None:
        configure_cognee()
    if not _init_done:
        ensure_tables_sync()
    if _cognee_module is None:
        raise RuntimeError("Cognee could not be initialized — check logs for details")
    return _cognee_module, _search_type_class
