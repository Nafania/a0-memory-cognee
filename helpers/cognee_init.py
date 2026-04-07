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
    llm_provider = _map_provider(util_provider)
    llm_api_key = util_cfg.get("api_key", "") or _models.get_api_key(util_provider)

    try:
        cognee.config.set_llm_config({
            "llm_provider": llm_provider,
            "llm_model": util_model,
            "llm_api_key": llm_api_key,
        })
        util_api_base = util_cfg.get("api_base", "")
        if util_api_base:
            cognee.config.set_llm_endpoint(util_api_base)
    except Exception as e:
        PrintStyle.error(f"cognee.config LLM setup failed, falling back to env vars: {e}")
        os.environ["LLM_PROVIDER"] = llm_provider
        os.environ["LLM_MODEL"] = util_model
        os.environ["LLM_API_KEY"] = llm_api_key
        util_api_base = util_cfg.get("api_base", "")
        if util_api_base:
            os.environ["LLM_API_BASE"] = util_api_base

    # --- Embedding ---
    embed_provider = _map_provider(embed_cfg.get("provider", ""))
    embed_model = embed_cfg.get("name", "")
    embed_api_key = embed_cfg.get("api_key", "") or _models.get_api_key(embed_cfg.get("provider", ""))

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
    embed_api_base = embed_cfg.get("api_base", "")
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
        from cognee.run_migrations import run_migrations

        await run_migrations()
    except BaseException as mig_err:
        # Must catch BaseException: Cognee's run_migrations() calls sys.exit(1)
        # on alembic failure, raising SystemExit which is BaseException, not Exception.
        PrintStyle.error(f"Cognee run_migrations failed ({type(mig_err).__name__}), trying create_db_and_tables: {mig_err}")
        try:
            from cognee.infrastructure.databases.relational import create_db_and_tables

            await create_db_and_tables()
        except Exception as e:
            PrintStyle.error(f"Cognee DB table creation failed: {e}")
            return

    _sync_missing_columns()
    PrintStyle.standard("Cognee DB tables initialized")


def _sync_missing_columns():
    """Compare Cognee ORM models against actual DB schema and add missing columns.

    TEMPORARY WORKAROUND for https://github.com/topoteretes/cognee/issues/TBD
    Cognee 0.5.7 added importance_weight to the Data ORM model (PR #2447) but
    shipped no alembic migration, so run_migrations() never adds the column to
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
