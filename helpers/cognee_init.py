import os
import logging
import importlib
import threading
import asyncio
import json
import shutil
import uuid
from datetime import timedelta
from pathlib import Path
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
    "cognee_debug_enabled": False,
    "cognee_operation_timeout_seconds": 1800,
    "cognee_rebuild_stale_after_seconds": 3600,
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
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 384,
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": 768,
    "intfloat/multilingual-e5-large": 1024,
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "nomic-embed-text:latest": 768,
}

_configured = False
_init_done = False
_init_error: BaseException | None = None
_init_running = False
_init_condition = threading.Condition(threading.RLock())
_cognee_module = None
_search_type_class = None

_LANCEDB_OPTIMIZE_FILE_THRESHOLD = 512
_EMBEDDING_CONFIG_STATE_FILE = "embedding_config_state.json"
_EMBEDDING_CONFIG_PENDING_FILE = "embedding_config_pending.json"
_LEGACY_DEFAULT_EMBEDDING_CONFIG: dict[str, str] = {
    "provider": "fastembed",
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "dimensions": "384",
    "api_base": "",
}
_embedding_rebuild_scheduled = False
_WATCHDOG_EXCLUDE_DIRS: set[str] = set()
_WATCHDOG_EXCLUDE_DIR_NAMES: set[str] = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".time_travel",
    "__pycache__",
    "node_modules",
}


def get_cognee_setting(name: str, default: T) -> T:
    target_default = _COGNEE_DEFAULTS.get(name, default)
    env_key = f"A0_SET_{name}"
    env_value = dotenv.get_dotenv_value(env_key, dotenv.get_dotenv_value(env_key.upper(), None))
    if env_value is not None:
        return _coerce_cognee_setting(env_value, target_default)

    plugin_value = _get_plugin_config_setting(name)
    if plugin_value is not None:
        return _coerce_cognee_setting(plugin_value, target_default)

    return target_default  # type: ignore


def _get_plugin_config_setting(name: str) -> Any:
    try:
        plugins = importlib.import_module("helpers.plugins")
        config = plugins.get_plugin_config("memory_cognee") or {}
        if isinstance(config, dict) and name in config:
            return config[name]
    except Exception:
        return None
    return None


def _coerce_cognee_setting(value: Any, default: T) -> T:
    try:
        if isinstance(default, bool):
            if isinstance(value, bool):
                return value  # type: ignore
            return str(value).strip().lower() in ("true", "1", "yes", "on")  # type: ignore
        elif isinstance(default, int):
            return type(default)(str(value).strip())  # type: ignore
        elif isinstance(default, str):
            return str(value).strip()  # type: ignore
        return default
    except (ValueError, TypeError):
        return default


def _normalize_watchdog_exclude_path(path: os.PathLike[str] | str | bytes) -> str:
    return os.path.abspath(os.path.normpath(os.fsdecode(path)))


def _is_watchdog_excluded_path(path: os.PathLike[str] | str | bytes) -> bool:
    normalized = _normalize_watchdog_exclude_path(path)
    for excluded in _WATCHDOG_EXCLUDE_DIRS:
        if normalized == excluded or normalized.startswith(excluded + os.sep):
            return True
    parts = normalized.split(os.sep)
    if any(part in _WATCHDOG_EXCLUDE_DIR_NAMES for part in parts):
        return True
    for index, part in enumerate(parts[:-1]):
        if part == ".a0proj" and parts[index + 1] == "memory":
            return True
    return False


def _patch_watchdog_inotify_excludes(paths: list[str]) -> None:
    """Keep Cognee database files out of Agent Zero's recursive file watcher."""
    normalized_paths = {
        _normalize_watchdog_exclude_path(path)
        for path in paths
        if path
    }
    if not normalized_paths:
        return
    _WATCHDOG_EXCLUDE_DIRS.update(normalized_paths)

    try:
        inotify_c = importlib.import_module("watchdog.observers.inotify_c")
        inotify_cls = getattr(inotify_c, "Inotify", None)
    except Exception:
        return
    if inotify_cls is None:
        return
    if getattr(inotify_cls, "_a0_memory_cognee_excludes_patch", False):
        return

    def _add_dir_watch(self, path, mask, *, recursive: bool) -> None:
        if _is_watchdog_excluded_path(path):
            return
        if not os.path.isdir(path):
            import errno

            raise OSError(errno.ENOTDIR, os.strerror(errno.ENOTDIR), path)
        self._add_watch(path, mask)
        if recursive:
            for root, dirnames, _ in os.walk(path):
                dirnames[:] = [
                    dirname
                    for dirname in dirnames
                    if not _is_watchdog_excluded_path(os.path.join(root, dirname))
                ]
                for dirname in dirnames:
                    full_path = os.path.join(root, dirname)
                    if os.path.islink(full_path):
                        continue
                    self._add_watch(full_path, mask)

    inotify_cls._add_dir_watch = _add_dir_watch
    inotify_cls._a0_memory_cognee_excludes_patch = True


def is_cognee_debug_enabled() -> bool:
    return get_cognee_setting("cognee_debug_enabled", False)


def _configure_cognee_logging() -> None:
    """Keep Cognee quiet by default; allow full logs through plugin config."""
    level_name = "DEBUG" if is_cognee_debug_enabled() else "WARNING"
    os.environ["LOG_LEVEL"] = level_name
    if level_name != "DEBUG":
        os.environ.setdefault("LITELLM_LOG", "ERROR")
        os.environ.setdefault("LITELLM_SET_VERBOSE", "False")

    level = getattr(logging, level_name, logging.WARNING)
    for logger_name in (
        "cognee",
        "GraphCompletionRetriever",
        "litellm",
        "LiteLLM",
        "httpx",
        "httpcore",
        "openai",
        "urllib3",
    ):
        logging.getLogger(logger_name).setLevel(level)


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


def _plugin_root_dir() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _configure_temporal_graph_prompt() -> None:
    """Point Cognee temporal extraction at an EventList-compatible prompt."""
    env_key = "TEMPORAL_GRAPH_PROMPT_PATH"
    if os.environ.get(env_key):
        return

    prompt_path = os.path.join(
        _plugin_root_dir(),
        "prompts",
        "cognee.generate_event_graph_prompt.txt",
    )
    if os.path.exists(prompt_path):
        os.environ[env_key] = prompt_path
    else:
        PrintStyle.warning(f"Cognee temporal prompt not found: {prompt_path}")


def _embedding_config_state_path(filename: str = _EMBEDDING_CONFIG_STATE_FILE) -> str:
    data_dir = files.get_abs_path(get_cognee_setting("cognee_data_dir", "usr/cognee"))
    return os.path.join(data_dir, "a0_state", filename)


def _current_embedding_config_state() -> dict[str, str]:
    return {
        "provider": os.environ.get("EMBEDDING_PROVIDER", ""),
        "model": os.environ.get("EMBEDDING_MODEL", ""),
        "dimensions": os.environ.get("EMBEDDING_DIMENSIONS", ""),
        "api_base": os.environ.get("EMBEDDING_API_BASE", ""),
    }


def _load_embedding_config_state() -> dict[str, str] | None:
    path = _embedding_config_state_path()
    return _load_embedding_config_state_file(path)


def _load_pending_embedding_config_state() -> dict[str, str] | None:
    path = _embedding_config_state_path(_EMBEDDING_CONFIG_PENDING_FILE)
    return _load_embedding_config_state_file(path)


def _load_embedding_config_state_file(path: str) -> dict[str, str] | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            value = json.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:
        PrintStyle.warning(f"Could not read Cognee embedding config state: {e}")
        return None

    if not isinstance(value, dict):
        PrintStyle.warning(
            f"Ignoring invalid Cognee embedding config state at {path}: not an object"
        )
        return None

    return {
        "provider": str(value.get("provider") or ""),
        "model": str(value.get("model") or ""),
        "dimensions": str(value.get("dimensions") or ""),
        "api_base": str(value.get("api_base") or ""),
    }


def _save_embedding_config_state(state: dict[str, str]) -> None:
    _save_embedding_config_state_file(
        _embedding_config_state_path(),
        state,
    )


def _save_pending_embedding_config_state(state: dict[str, str]) -> None:
    _save_embedding_config_state_file(
        _embedding_config_state_path(_EMBEDDING_CONFIG_PENDING_FILE),
        state,
    )


def _save_embedding_config_state_file(path: str, state: dict[str, str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, sort_keys=True)
        f.write("\n")
    os.replace(tmp_path, path)


def _clear_pending_embedding_config_state() -> None:
    path = _embedding_config_state_path(_EMBEDDING_CONFIG_PENDING_FILE)
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def _format_embedding_config_state(state: dict[str, str] | None) -> str:
    if not state:
        return "unknown"
    return (
        f"provider={state.get('provider') or ''}, "
        f"model={state.get('model') or ''}, "
        f"dimensions={state.get('dimensions') or ''}, "
        f"api_base={state.get('api_base') or ''}"
    )


def _embedding_config_rebuild_needed(
    current: dict[str, str] | None = None,
) -> bool:
    current = current or _current_embedding_config_state()
    if not current.get("provider") or not current.get("model"):
        return False

    pending = _load_pending_embedding_config_state()
    if pending == current:
        return True

    previous = _load_embedding_config_state()
    unknown_nonlegacy = previous is None and current != _LEGACY_DEFAULT_EMBEDDING_CONFIG
    changed = previous is not None and previous != current
    return unknown_nonlegacy or changed


async def _ensure_embedding_config_state(
    current: dict[str, str] | None = None,
) -> None:
    global _embedding_rebuild_scheduled
    current = current or _current_embedding_config_state()
    if not current.get("provider") or not current.get("model"):
        PrintStyle.warning(
            "Cognee embedding config state not updated: embedding provider/model is empty"
        )
        return

    previous = _load_embedding_config_state()
    pending = _load_pending_embedding_config_state()
    if pending == current:
        if not _embedding_rebuild_scheduled:
            PrintStyle.warning(
                "Cognee embedding config rebuild is already pending; resuming "
                "background rebuild without resetting pipeline status again. "
                f"current=({_format_embedding_config_state(current)})"
            )
            await mark_all_datasets_dirty_for_rebuild(
                "pending embedding config rebuild"
            )
            _embedding_rebuild_scheduled = True
        return

    if previous == current:
        _clear_pending_embedding_config_state()
        return

    unknown_nonlegacy = previous is None and current != _LEGACY_DEFAULT_EMBEDDING_CONFIG
    changed = previous is not None and previous != current
    if unknown_nonlegacy or changed:
        if _embedding_rebuild_scheduled and pending == current:
            return
        if previous is None:
            PrintStyle.warning(
                "Cognee embedding config provenance is unknown and current config is "
                "not the legacy default; scheduling full memory graph rebuild. "
                f"current=({_format_embedding_config_state(current)})"
            )
        else:
            PrintStyle.warning(
                "Cognee embedding config changed; scheduling full memory graph "
                "rebuild before recall can use the new vectors. "
                f"previous=({_format_embedding_config_state(previous)}); "
                f"current=({_format_embedding_config_state(current)})"
            )
        await reset_cognify_status_for_all_datasets()
        _save_pending_embedding_config_state(current)
        _embedding_rebuild_scheduled = True
        return

    _save_embedding_config_state(current)
    _clear_pending_embedding_config_state()


def _mark_embedding_config_rebuild_completed() -> None:
    pending = _load_pending_embedding_config_state()
    if not pending:
        return
    current = _current_embedding_config_state()
    if current.get("provider") and current.get("model") and current != pending:
        PrintStyle.warning(
            "Cognee embedding rebuild completed, but pending embedding config no "
            "longer matches current config; leaving pending state for next startup. "
            f"pending=({_format_embedding_config_state(pending)}); "
            f"current=({_format_embedding_config_state(current)})"
        )
        return

    _save_embedding_config_state(pending)
    _clear_pending_embedding_config_state()
    PrintStyle.standard(
        "Cognee embedding config state applied after successful rebuild: "
        f"{_format_embedding_config_state(pending)}"
    )


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
    _configure_cognee_logging()
    settings = get_settings()

    # --- Storage directories (MUST be set BEFORE import cognee) ---
    data_dir = files.get_abs_path(get_cognee_setting("cognee_data_dir", "usr/cognee"))
    os.makedirs(data_dir, exist_ok=True)
    _patch_watchdog_inotify_excludes(
        [
            data_dir,
            files.get_abs_path("usr/cognee_state"),
            files.get_abs_path("usr/memory"),
            files.get_abs_path("usr/mcp"),
            files.get_abs_path("usr/lib"),
            files.get_abs_path("usr/npm-global"),
            files.get_abs_path("usr/.npm"),
        ]
    )

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
    _configure_temporal_graph_prompt()

    PrintStyle.standard(f"Cognee env configured: system={system_storage}, data={data_storage}")

    # --- Now safe to import cognee (env vars are set) ---
    try:
        import cognee
        from cognee import SearchType
    except Exception as e:
        import traceback
        PrintStyle.error(f"Cognee import failed — memory features will not work: {e}")
        PrintStyle.error(traceback.format_exc())
        raise RuntimeError(f"Cognee import failed: {e}") from e

    _cognee_module = cognee
    _search_type_class = SearchType
    _patch_lancedb_remote_table_replay_refs()

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
    _patch_lancedb_migration_defaults()
    embedding_rebuild_needed = _embedding_config_rebuild_needed()
    try:
        if embedding_rebuild_needed:
            from cognee.run_migrations import run_migrations

            await run_migrations()
            PrintStyle.warning(
                "Skipping Cognee startup vector migrations because embedding "
                "rebuild is pending; the background rebuild will recreate vectors."
            )
        else:
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
            raise RuntimeError(f"Cognee DB table creation failed: {e}") from e

    _sync_missing_columns()
    _rewrite_legacy_data_storage_locations()
    _quarantine_missing_data_files()
    if embedding_rebuild_needed:
        PrintStyle.warning(
            "Skipping startup LanceDB optimization because Cognee embedding "
            "rebuild is pending; the rebuild will refresh vector tables."
        )
    else:
        try:
            await _optimize_fragmented_lancedb_tables()
        except Exception as e:
            PrintStyle.error(
                "LanceDB optimization failed during startup; continuing Cognee init "
                f"with degraded search risk: {e}"
            )
    if embedding_rebuild_needed:
        PrintStyle.warning(
            "Skipping startup Cognee graph readiness scan because embedding "
            "rebuild is pending; all affected datasets will be reset by the "
            "embedding rebuild scheduler."
        )
    else:
        affected_datasets = await _detect_datasets_with_unready_graphs()
        if affected_datasets:
            await _reset_cognify_status_for_datasets(affected_datasets)
    PrintStyle.standard("Cognee DB tables initialized")


def _patch_lancedb_migration_defaults() -> None:
    """Patch Cognee LanceDB migration defaults for old vector rows.

    Cognee's generated LanceDB payload schema makes source_* fields required
    even though the original Pydantic model defaults them to None. Existing
    1.0.1-era rows do not contain those fields, so startup migration aborts
    before rebuilding the table. Adding explicit None defaults lets Cognee's own
    migration preserve rows instead of skipping them.
    """
    try:
        from cognee.infrastructure.databases.vector.lancedb.LanceDBAdapter import (
            LanceDBAdapter,
        )

        if getattr(LanceDBAdapter, "_a0_memory_cognee_defaults_patch", False):
            return

        original = LanceDBAdapter._get_payload_defaults

        def _patched_get_payload_defaults(self, payload_schema):
            defaults = dict(original(self, payload_schema) or {})
            try:
                schema_model = self.get_data_point_schema(payload_schema)
                fields = getattr(schema_model, "model_fields", {})
            except Exception:
                fields = {}

            for key in (
                "source_pipeline",
                "source_task",
                "source_node_set",
                "source_user",
                "source_content_hash",
            ):
                if key in fields and key not in defaults:
                    defaults[key] = None

            if "metadata" in fields and "metadata" not in defaults:
                defaults["metadata"] = {}

            return defaults

        LanceDBAdapter._get_payload_defaults = _patched_get_payload_defaults
        LanceDBAdapter._a0_memory_cognee_defaults_patch = True
    except Exception as e:
        PrintStyle.warning(f"Could not patch LanceDB migration defaults: {e}")


def _patch_lancedb_remote_table_replay_refs() -> None:
    """Avoid retaining every subprocess LanceDB table through replay callbacks.

    Cognee's subprocess proxy registers a replay step for each opened LanceDB
    table so worker respawns can reopen it. In 1.1.0 that replay step closes
    over the RemoteLanceDBTable instance directly, so the session keeps the
    table alive and its __del__ never releases the worker-side handle. Graph
    search opens several tables per query; under normal chat recall this can
    grow until LanceDB fails with "Too many open files".
    """
    try:
        import weakref
        from cognee.infrastructure.databases.vector.lancedb.subprocess.proxy import (
            OP_OPEN_TABLE,
            RemoteLanceDBTable,
            ReplayStep,
            Request,
        )

        if getattr(RemoteLanceDBTable, "_a0_memory_cognee_weak_replay_patch", False):
            return

        def _patched_init(self, session, handle_id: int, name: str):
            self._session = session
            self._handle_id = handle_id
            self.name = name

            table_ref = weakref.ref(self)
            table_name = name

            def _make_request():
                return Request(op=OP_OPEN_TABLE, args=(table_name,))

            def _apply_new_handle(new_handle_id: int):
                table = table_ref()
                if table is None:
                    return None
                return table._apply_new_handle(new_handle_id)

            self._replay_step = ReplayStep(
                make_request=_make_request,
                apply_new_handle=_apply_new_handle,
            )
            self._session.add_replay_step(self._replay_step)

        RemoteLanceDBTable.__init__ = _patched_init
        RemoteLanceDBTable._a0_memory_cognee_weak_replay_patch = True
    except Exception as e:
        PrintStyle.warning(f"Could not patch LanceDB subprocess table replay refs: {e}")


def _count_files_under(path: Path) -> int:
    return sum(1 for child in path.rglob("*") if child.is_file())


def _find_fragmented_lancedb_tables(
    system_root: str,
    *,
    min_files: int = _LANCEDB_OPTIMIZE_FILE_THRESHOLD,
) -> list[tuple[int, Path, str, Path]]:
    databases_root = Path(system_root) / "databases"
    if not databases_root.exists():
        return []

    targets: list[tuple[int, Path, str, Path]] = []
    for table_dir in databases_root.rglob("*.lance"):
        if not table_dir.is_dir():
            continue
        file_count = _count_files_under(table_dir)
        if file_count >= min_files:
            table_name = table_dir.name[: -len(".lance")]
            targets.append((file_count, table_dir.parent, table_name, table_dir))

    targets.sort(reverse=True, key=lambda item: item[0])
    return targets


async def _optimize_fragmented_lancedb_tables(
    *,
    min_files: int = _LANCEDB_OPTIMIZE_FILE_THRESHOLD,
) -> int:
    """Compact fragmented LanceDB tables before graph search can open them.

    Cognee GRAPH_COMPLETION searches with a node_name filter can scan whole
    vector collections. On heavily updated LanceDB tables, each collection may
    contain thousands of segment/version files; a single search can then exceed
    production's nofile limit. LanceDB optimize compacts live fragments and
    prunes old versions without changing the stored data or recall flow.
    """
    system_root = os.environ.get("SYSTEM_ROOT_DIRECTORY", "")
    if not system_root:
        return 0

    targets = _find_fragmented_lancedb_tables(system_root, min_files=min_files)
    if not targets:
        return 0

    try:
        import lancedb
    except Exception as e:
        raise RuntimeError(
            f"Cannot import lancedb to optimize {len(targets)} fragmented table(s): {e}"
        ) from e

    total_before = sum(file_count for file_count, _, _, _ in targets)
    PrintStyle.warning(
        "Detected "
        f"{len(targets)} fragmented LanceDB table(s) with {total_before} files; "
        "optimizing before Cognee search is enabled."
    )

    optimized = 0
    failures: list[str] = []
    for before_count, db_dir, table_name, table_dir in targets:
        try:
            db = await lancedb.connect_async(str(db_dir))
            table = await db.open_table(table_name)
            stats = await table.optimize(cleanup_older_than=timedelta(seconds=0))
            after_count = _count_files_under(table_dir)
            optimized += 1
            PrintStyle.standard(
                "Optimized LanceDB table "
                f"{db_dir.name}/{table_name}: files {before_count}->{after_count}; "
                f"stats={stats}"
            )
        except Exception as e:
            detail = f"{db_dir.name}/{table_name}: {type(e).__name__}: {e}"
            failures.append(detail)
            PrintStyle.error(f"LanceDB optimization failed for {detail}")

    if failures:
        raise RuntimeError(
            "LanceDB optimization failed for "
            f"{len(failures)} table(s); Cognee search may still fail with open FD "
            f"exhaustion: {failures[:5]}"
        )

    return optimized


def _lancedb_dataset_dir_name(dataset_id: Any) -> str | None:
    value = str(dataset_id or "").strip()
    if not value:
        return None

    try:
        normalized = str(uuid.UUID(value))
    except ValueError:
        try:
            normalized = str(uuid.UUID(hex=value))
        except ValueError:
            normalized = value
    return f"{normalized}.lance.db"


async def purge_lancedb_vector_tables_for_dataset_names(
    dataset_names: list[str],
) -> list[str]:
    """Delete LanceDB vector stores for named datasets before embedding rebuild.

    This only removes per-dataset ``*.lance.db`` vector directories. Relational
    data, graph files, and Cognee source files remain intact; cognify recreates
    vectors from the existing data.
    """
    clean_names = {name for name in dataset_names if name}
    if not clean_names:
        return []

    system_root = os.environ.get("SYSTEM_ROOT_DIRECTORY", "")
    if not system_root:
        return []

    databases_root = Path(system_root) / "databases"
    if not databases_root.exists():
        return []

    try:
        import cognee
    except Exception as e:
        PrintStyle.warning(f"Cannot import cognee to purge LanceDB vectors: {e}")
        return []

    try:
        all_datasets = await cognee.datasets.list_datasets()
    except Exception as e:
        PrintStyle.warning(f"Could not list Cognee datasets to purge LanceDB vectors: {e}")
        return []

    target_dirs: dict[str, str] = {}
    for ds in all_datasets:
        name = getattr(ds, "name", None)
        if name not in clean_names:
            continue
        dir_name = _lancedb_dataset_dir_name(getattr(ds, "id", None))
        if dir_name:
            target_dirs[str(name)] = dir_name

    purged: list[str] = []
    for dataset_name, dir_name in target_dirs.items():
        removed_paths: list[str] = []
        for vector_dir in databases_root.rglob(dir_name):
            if not vector_dir.is_dir() or not vector_dir.name.endswith(".lance.db"):
                continue
            shutil.rmtree(vector_dir)
            removed_paths.append(str(vector_dir))

        if removed_paths:
            purged.append(dataset_name)
            PrintStyle.warning(
                "Purged stale LanceDB vector store before embedding rebuild "
                f"for dataset {dataset_name}: {removed_paths}"
            )

    return purged


def _rewrite_legacy_data_storage_locations() -> int:
    """Rewrite old absolute Cognee file:// data paths to the current data root.

    Cognee stores raw document locations in SQLite. When an Agent Zero `usr`
    directory is moved between host paths or into Docker, rows can still point at
    the old absolute `.../usr/cognee/data_storage/<data-id>` path even though the
    actual files exist under the current DATA_ROOT_DIRECTORY. Rewriting the URI is
    non-destructive and lets Cognee rebuild graphs from the preserved source text.
    """
    try:
        import sqlite3

        data_root = os.environ.get("DATA_ROOT_DIRECTORY", "")
        system_root = os.environ.get("SYSTEM_ROOT_DIRECTORY", "")
        if not data_root or not system_root:
            return 0

        db_path = os.path.join(
            system_root,
            "databases",
            os.environ.get("DB_NAME", "cognee_db"),
        )
        if not os.path.exists(db_path):
            return 0

        updated = 0
        conn = sqlite3.connect(db_path)
        try:
            cols = {
                row[1]
                for row in conn.execute("PRAGMA table_info(data)").fetchall()
            }
            location_cols = [
                col
                for col in ("raw_data_location", "original_data_location")
                if col in cols
            ]
            if not location_cols:
                return 0

            select_cols = ", ".join(["id", *location_cols])
            rows = conn.execute(f"SELECT {select_cols} FROM data").fetchall()
            for row in rows:
                data_id = row[0]
                changes: dict[str, str] = {}
                for col, value in zip(location_cols, row[1:]):
                    rewritten = _rewrite_data_storage_uri(value, data_root)
                    if rewritten and rewritten != value:
                        changes[col] = rewritten

                if not changes:
                    continue

                assignments = ", ".join(f"{col} = ?" for col in changes)
                conn.execute(
                    f"UPDATE data SET {assignments} WHERE id = ?",
                    [*changes.values(), data_id],
                )
                updated += len(changes)

            conn.commit()
        finally:
            conn.close()

        if updated:
            PrintStyle.standard(
                f"Rewrote {updated} legacy Cognee data storage path(s) "
                "to current DATA_ROOT_DIRECTORY."
            )
        return updated
    except Exception as e:
        PrintStyle.warning(f"Legacy Cognee data path rewrite failed (non-fatal): {e}")
        return 0


def _rewrite_data_storage_uri(value: Any, data_root: str) -> str | None:
    if not isinstance(value, str) or "/data_storage/" not in value:
        return None

    prefix = "file://"
    path = value[len(prefix):] if value.startswith(prefix) else value
    marker = "/data_storage/"
    marker_index = path.find(marker)
    if marker_index == -1:
        return None

    suffix = path[marker_index + len(marker):].lstrip("/")
    if not suffix:
        return None

    candidate = os.path.abspath(os.path.join(data_root, suffix))
    data_root_abs = os.path.abspath(data_root)
    if not candidate.startswith(data_root_abs + os.sep) and candidate != data_root_abs:
        return None

    current_path = os.path.abspath(path)
    if current_path == candidate:
        return None

    if not os.path.exists(candidate):
        return None

    return f"{prefix}{candidate}"


def _quarantine_missing_data_files() -> int:
    """Detach data rows whose preserved source file is no longer present.

    Cognee's cognify pipeline aborts the whole dataset when any data item points
    at a missing source file. The SQLite row may still be useful for forensic
    inspection, so we keep it and only remove dataset associations, recording the
    quarantine reason in a small plugin-owned table.
    """
    try:
        import sqlite3

        system_root = os.environ.get("SYSTEM_ROOT_DIRECTORY", "")
        if not system_root:
            return 0

        db_path = os.path.join(
            system_root,
            "databases",
            os.environ.get("DB_NAME", "cognee_db"),
        )
        if not os.path.exists(db_path):
            return 0

        conn = sqlite3.connect(db_path)
        try:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
            if "data" not in tables or "dataset_data" not in tables:
                return 0

            conn.execute(
                "CREATE TABLE IF NOT EXISTS a0_cognee_quarantined_data ("
                "data_id TEXT PRIMARY KEY, "
                "raw_data_location TEXT, "
                "reason TEXT, "
                "quarantined_at TEXT DEFAULT CURRENT_TIMESTAMP"
                ")"
            )

            rows = conn.execute(
                "SELECT id, name, extension, raw_data_location FROM data"
            ).fetchall()
            quarantined = 0
            for data_id, name, extension, raw_location in rows:
                missing_path = _missing_data_source_path(raw_location, name, extension)
                if not missing_path:
                    continue

                assoc_count = conn.execute(
                    "SELECT COUNT(*) FROM dataset_data WHERE data_id = ?",
                    (data_id,),
                ).fetchone()[0]
                if not assoc_count:
                    continue

                conn.execute(
                    "INSERT OR REPLACE INTO a0_cognee_quarantined_data "
                    "(data_id, raw_data_location, reason) VALUES (?, ?, ?)",
                    (
                        data_id,
                        raw_location,
                        f"Missing source file: {missing_path}",
                    ),
                )
                conn.execute(
                    "DELETE FROM dataset_data WHERE data_id = ?",
                    (data_id,),
                )
                quarantined += 1

            conn.commit()
        finally:
            conn.close()

        if quarantined:
            PrintStyle.warning(
                f"Quarantined {quarantined} Cognee data item(s) with missing "
                "source files so dataset rebuild can continue. Original rows "
                "were preserved in SQLite."
            )
        return quarantined
    except Exception as e:
        PrintStyle.warning(f"Missing Cognee source quarantine failed (non-fatal): {e}")
        return 0


def _missing_data_source_path(
    raw_location: Any,
    name: Any = None,
    extension: Any = None,
) -> str | None:
    if not isinstance(raw_location, str) or not raw_location.startswith("file://"):
        return None

    path = raw_location[len("file://"):]
    if os.path.isfile(path):
        return None

    if os.path.isdir(path):
        if isinstance(name, str) and name:
            suffix = f".{extension}" if isinstance(extension, str) and extension else ""
            expected_file = os.path.join(path, f"{name}{suffix}")
            if os.path.isfile(expected_file):
                return None
            return expected_file
        return None

    return path


async def _detect_datasets_with_unready_graphs() -> set[str]:
    """Find datasets that have data but an empty or unreadable dataset graph."""
    unready_dataset_ids: set[str] = set()
    try:
        import cognee

        try:
            from usr.plugins.memory_cognee.helpers.cognee_graph import read_dataset_graphs
        except Exception:
            from .cognee_graph import read_dataset_graphs

        dataset_graphs = await read_dataset_graphs(
            cognee,
            skip_empty_data=True,
            repair_unreadable=True,
            include_graph_data=False,
        )
        for graph in dataset_graphs:
            dataset_id = str(getattr(graph, "dataset_id", "") or "")
            dataset_name = str(getattr(graph, "dataset_name", "") or dataset_id)
            data_count = getattr(graph, "data_count", None)
            if data_count is None:
                PrintStyle.warning(
                    f"Could not confirm Cognee dataset '{dataset_name}' data count; "
                    "skipping startup graph reset"
                )
                continue
            if data_count <= 0:
                continue

            graph_error = getattr(graph, "error", None)
            graph_empty = getattr(graph, "graph_empty", None)
            if graph_error:
                PrintStyle.warning(
                    f"Could not verify Cognee dataset '{dataset_name}' graph during "
                    f"startup; leaving pipeline status unchanged: {graph_error}"
                )
                continue
            if graph_empty is False and not graph_error:
                continue

            is_unready = graph_empty is True
            if is_unready and dataset_id:
                unready_dataset_ids.add(dataset_id)
                detail = f": {graph_error}" if graph_error else ""
                PrintStyle.warning(
                    f"Detected dataset '{dataset_name}' with data but an unready "
                    f"knowledge graph. Will reset cognify_pipeline{detail}"
                )
    except Exception as e:
        PrintStyle.warning(f"Unready graph detection failed (non-fatal): {e}")

    return unready_dataset_ids


async def _log_startup_readiness(migration_completed: bool, worker_status: dict) -> None:
    """Log a single operator-readable Cognee readiness summary."""
    dirty = sorted(str(name) for name in (worker_status.get("dirty_datasets") or []))
    readiness = worker_status.get("dataset_readiness") or {}
    blocked_states = []
    readable_rebuild_states = []
    if isinstance(readiness, dict):
        for dataset_name, state in readiness.items():
            if not isinstance(state, dict):
                continue
            state_name = str(state.get("state") or "")
            if state_name and state_name != "ready" and not state.get("readable"):
                blocked_states.append(f"{dataset_name}:{state_name}")
            elif state_name and state_name != "ready":
                readable_rebuild_states.append(f"{dataset_name}:{state_name}")

    running = bool(worker_status.get("running"))
    migration_status = "complete" if migration_completed else "incomplete"

    if _embedding_config_rebuild_needed():
        PrintStyle.warning(
            "Cognee startup readiness: BLOCKED; embedding config rebuild is pending. "
            "Dataset graph status was not read during startup to avoid opening Cognee "
            "graph/vector subprocesses before the background rebuild. "
            f"migration={migration_status}; running={running}; "
            f"dirty={dirty}; blocked_states={blocked_states}"
        )
        return

    try:
        import cognee

        try:
            from usr.plugins.memory_cognee.helpers.cognee_graph import read_dataset_graphs
        except Exception:
            from .cognee_graph import read_dataset_graphs

        dataset_graphs = await read_dataset_graphs(
            cognee,
            skip_empty_data=True,
            repair_unreadable=False,
            include_graph_data=False,
        )
    except Exception as e:
        PrintStyle.warning(
            "Cognee startup readiness: UNKNOWN; could not read dataset graph "
            f"status: {e}"
        )
        return

    ready: list[str] = []
    empty: list[str] = []
    errors: list[str] = []
    unknown: list[str] = []

    for graph in dataset_graphs:
        dataset_name = str(getattr(graph, "dataset_name", "") or "unknown")
        graph_error = getattr(graph, "error", None)
        graph_empty = getattr(graph, "graph_empty", None)

        if graph_error:
            errors.append(f"{dataset_name}: {graph_error}")
        elif graph_empty is False:
            ready.append(dataset_name)
        elif graph_empty is True:
            empty.append(dataset_name)
        else:
            unknown.append(dataset_name)

    if ready:
        try:
            from .cognee_background import CogneeBackgroundWorker

            CogneeBackgroundWorker.get_instance().mark_datasets_readable(
                ready,
                "Cognee startup graph readiness verified",
            )
            if isinstance(readiness, dict):
                for dataset_name in ready:
                    state = readiness.setdefault(dataset_name, {})
                    if isinstance(state, dict):
                        state["readable"] = True
                        state.setdefault("state", "ready")
        except Exception as e:
            PrintStyle.warning(f"Could not record Cognee readable datasets: {e}")

    dirty_blocking = []
    if isinstance(readiness, dict):
        for dataset_name in dirty:
            state = readiness.get(dataset_name, {})
            if not isinstance(state, dict) or not state.get("readable"):
                dirty_blocking.append(dataset_name)
    else:
        dirty_blocking = dirty

    graph_blocked = bool(empty or errors or unknown)
    worker_blocked = bool(dirty_blocking or blocked_states)
    graph_total = len(ready) + len(empty) + len(errors) + len(unknown)

    if graph_blocked or worker_blocked:
        PrintStyle.warning(
            "Cognee startup readiness: BLOCKED; recall may be unavailable. "
            f"graphs_ready={len(ready)}/{graph_total}; "
            f"migration={migration_status}; running={running}; "
            f"dirty={dirty}; dirty_blocking={dirty_blocking}; "
            f"blocked_states={blocked_states}; "
            f"readable_rebuild_states={readable_rebuild_states}; "
            f"empty_graphs={empty}; graph_errors={_short_list(errors)}; "
            f"unknown_graphs={unknown}"
        )
    elif not migration_completed:
        PrintStyle.warning(
            "Cognee startup readiness: DEGRADED; Cognee recall is enabled for "
            "existing graph datasets, but legacy FAISS migration is incomplete. "
            f"graphs_ready={len(ready)}/{graph_total}; migration=incomplete"
        )
    else:
        PrintStyle.standard(
            "Cognee startup readiness: READY; recall enabled. "
            f"graphs_ready={len(ready)}/{graph_total}; migration=complete"
        )

    PrintStyle.standard(
        "Cognee dataset graph status: "
        f"ready={_short_list(ready)}; "
        f"empty={_short_list(empty)}; "
        f"errors={_short_list(errors)}; "
        f"unknown={_short_list(unknown)}"
    )


def _short_list(items: list[str], limit: int = 12) -> list[str]:
    if len(items) <= limit:
        return items
    return [*items[:limit], f"... +{len(items) - limit} more"]


async def _reset_cognify_status_for_datasets(
    dataset_ids: set[str],
    *,
    reset_all: bool = False,
) -> None:
    """Reset cognify_pipeline status and mark only affected datasets dirty."""
    if not dataset_ids:
        return
    try:
        import cognee
    except Exception as e:
        PrintStyle.error(
            f"Cannot import cognee dataset helpers: {e}. "
            f"Graph will NOT auto-rebuild -- re-add data or call cognify manually."
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

    requested_ids = {str(dataset_id) for dataset_id in dataset_ids}
    dataset_names: list[str] = []
    dataset_id_values = []
    for ds in all_datasets:
        dataset_id = str(getattr(ds, "id", "") or "")
        if not reset_all and dataset_id not in requested_ids:
            continue
        dataset_id_values.append(ds.id)
        if getattr(ds, "name", None):
            dataset_names.append(ds.name)

    if not dataset_id_values:
        PrintStyle.warning(
            f"No Cognee datasets matched reset request: {sorted(requested_ids)}"
        )
        return

    reset_count = await _delete_pipeline_runs_for_dataset_ids(dataset_id_values)

    if reset_count:
        PrintStyle.standard(
            f"Reset pipeline status for {reset_count} dataset(s). "
            f"Graph will rebuild on next cognify()."
        )

    if reset_count or reset_all:
        try:
            from .cognee_background import CogneeBackgroundWorker

            worker = CogneeBackgroundWorker.get_instance()
            for name in dataset_names:
                worker.mark_dirty(name, preserve_readable=False)
            if dataset_names:
                PrintStyle.standard(
                    f"Marked {len(dataset_names)} dataset(s) dirty for background rebuild: {dataset_names}"
                )
        except Exception as e:
            PrintStyle.warning(
                f"Could not mark datasets dirty (graph will rebuild on next insert instead): {e}"
            )


async def reset_cognify_status_for_dataset_names(dataset_names: list[str]) -> list[str]:
    """Reset cognify_pipeline status for explicit dataset names and return those reset."""
    clean_names = {name for name in dataset_names if name}
    if not clean_names:
        return []

    try:
        import cognee
    except Exception as e:
        PrintStyle.error(f"Cannot import cognee dataset helpers: {e}")
        return []

    try:
        all_datasets = await cognee.datasets.list_datasets()
    except Exception as e:
        PrintStyle.error(f"Could not prepare cognify status reset: {e}")
        return []

    reset_names: list[str] = []
    dataset_id_values = []
    for ds in all_datasets:
        name = getattr(ds, "name", None)
        if name not in clean_names:
            continue
        dataset_id_values.append(ds.id)
        reset_names.append(name)

    reset_count = await _delete_pipeline_runs_for_dataset_ids(dataset_id_values)

    if reset_count:
        PrintStyle.standard(f"Reset pipeline status for dataset(s): {reset_names}")
    return reset_names


async def reset_cognify_status_for_all_datasets() -> None:
    """Reset cognify_pipeline status for every dataset and mark them dirty."""
    await _reset_cognify_status_for_datasets({"manual_reindex"}, reset_all=True)


async def mark_all_datasets_dirty_for_rebuild(reason: str) -> None:
    """Resume a pending rebuild without deleting Cognee pipeline progress again."""
    try:
        import cognee
    except Exception as e:
        PrintStyle.error(f"Cannot import cognee dataset helpers: {e}")
        return

    try:
        all_datasets = await cognee.datasets.list_datasets()
    except Exception as e:
        PrintStyle.error(f"Could not list datasets to resume Cognee rebuild: {e}")
        return

    dataset_names = [
        ds.name
        for ds in all_datasets
        if getattr(ds, "name", None)
    ]
    if not dataset_names:
        PrintStyle.warning("No Cognee datasets found for pending rebuild resume")
        return

    try:
        from .cognee_background import CogneeBackgroundWorker

        worker = CogneeBackgroundWorker.get_instance()
        for name in dataset_names:
            worker.mark_dirty(name, reset_retry=False, preserve_readable=False)
        PrintStyle.standard(
            f"Marked {len(dataset_names)} dataset(s) dirty for background rebuild resume "
            f"({reason}): {dataset_names}"
        )
    except Exception as e:
        PrintStyle.warning(
            f"Could not mark datasets dirty for pending rebuild resume: {e}"
        )


async def _delete_pipeline_runs_for_dataset_ids(dataset_ids: list) -> int:
    """Clear Cognee pipeline metadata so the next cognify is not skipped."""
    if not dataset_ids:
        return 0

    try:
        from sqlalchemy import delete
        from sqlalchemy import select
        from cognee.infrastructure.databases.relational import get_relational_engine
        from cognee.modules.data.models import Data
        from cognee.modules.pipelines.models import PipelineRun
        from sqlalchemy.orm.attributes import flag_modified
    except Exception as e:
        PrintStyle.error(f"Cannot import Cognee pipeline models for reset: {e}")
        return 0

    dataset_id_strings = {str(dataset_id) for dataset_id in dataset_ids}

    try:
        db_engine = get_relational_engine()
        async with db_engine.get_async_session() as session:
            result = await session.execute(
                delete(PipelineRun).where(PipelineRun.dataset_id.in_(dataset_ids))
            )
            rows = (await session.execute(select(Data))).scalars().all()
            data_reset_count = 0
            for row in rows:
                status = getattr(row, "pipeline_status", None)
                if not isinstance(status, dict):
                    continue

                changed = False
                new_status = {}
                for pipeline_name, dataset_statuses in status.items():
                    if not isinstance(dataset_statuses, dict):
                        new_status[pipeline_name] = dataset_statuses
                        continue

                    remaining = {
                        dataset_id: item_status
                        for dataset_id, item_status in dataset_statuses.items()
                        if str(dataset_id) not in dataset_id_strings
                    }
                    if len(remaining) != len(dataset_statuses):
                        changed = True
                    if remaining:
                        new_status[pipeline_name] = remaining

                if changed:
                    row.pipeline_status = new_status
                    flag_modified(row, "pipeline_status")
                    data_reset_count += 1

            await session.commit()
            deleted = int(getattr(result, "rowcount", 0) or 0)
    except Exception as e:
        PrintStyle.error(f"Failed to delete Cognee pipeline status rows: {e}")
        return 0

    if data_reset_count:
        PrintStyle.standard(
            f"Reset pipeline status metadata for {data_reset_count} data item(s)."
        )
    return len(dataset_ids) if deleted or data_reset_count else 0


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
        engine = None
        try:
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
        finally:
            if engine is not None:
                dispose = getattr(engine, "dispose", None)
                if callable(dispose):
                    dispose()
    except Exception as e:
        PrintStyle.error(f"Schema column sync failed: {e}")


async def init_cognee() -> None:
    """One-time startup initialization. Idempotent — safe to call multiple times."""
    global _init_done, _init_error, _init_running
    while True:
        with _init_condition:
            if _init_done:
                return
            if _init_error is not None and not _init_running:
                raise RuntimeError(
                    f"Cognee initialization failed: {_init_error}"
                ) from _init_error
            if not _init_running:
                _init_running = True
                _init_error = None
                break

        await asyncio.to_thread(_wait_for_init_result)

    try:
        configure_cognee()
        if _cognee_module is None:
            raise RuntimeError("Cognee configure failed: module was not loaded")
        await _create_db_tables()
        PrintStyle.standard("Cognee core initialized")
    except BaseException as e:
        with _init_condition:
            _init_done = False
            _init_error = e
            _init_running = False
            _init_condition.notify_all()
        raise
    else:
        with _init_condition:
            _init_done = True
            _init_error = None
            _init_running = False
            _init_condition.notify_all()


def _wait_for_init_result() -> None:
    with _init_condition:
        while _init_running:
            _init_condition.wait()


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
            raise
        return

    import threading
    errors: list[BaseException] = []

    def _run():
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(init_cognee())
        except BaseException as e:
            PrintStyle.error(f"ensure_tables_sync (thread): {type(e).__name__}: {e}")
            errors.append(e)
        finally:
            loop.close()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=60)
    if t.is_alive():
        raise TimeoutError("Cognee initialization did not finish within 60 seconds")
    if errors:
        raise RuntimeError(f"Cognee initialization failed: {errors[0]}") from errors[0]


def run_memory_cognee_init_a0_extension() -> None:
    """Entry for Agent Zero `init_a0` / `end` extensions.

    Upstream `run_ui.run()` calls `init_a0()` before starting the server; extension
    folders are `_functions/<run_ui.init_a0.__module__>/init_a0/start/` per
    `helpers.extension.extensible` (see agent0ai/agent-zero). Official Docker starts
    `python run_ui.py`, so `__module__` is usually `__main__`; if `run_ui` is imported
    as a module, use the duplicate extension under `_functions/run_ui/...`.
    """
    import asyncio

    try:
        _disable_builtin_memory_plugin()
        configure_cognee()
        asyncio.run(init_cognee())
        from . import faiss_migration

        migrated = asyncio.run(faiss_migration.run_migration())
        if not migrated:
            PrintStyle.warning("FAISS -> Cognee migration did not complete; will retry on next startup")
        asyncio.run(_ensure_embedding_config_state())
        from .cognee_background import CogneeBackgroundWorker

        worker = CogneeBackgroundWorker.get_instance()
        status = worker.get_status()
        asyncio.run(_log_startup_readiness(migrated, status))
        dirty_datasets = list(status.get("dirty_datasets") or [])
        if dirty_datasets:
            PrintStyle.warning(
                "Cognee memory graph rebuild required before recall; "
                f"background rebuild pending for dataset(s): {dirty_datasets}"
            )

    except BaseException as e:
        # BaseException: asyncio.run() re-raises SystemExit from run_migrations
        PrintStyle.error(f"Cognee eager init failed ({type(e).__name__}): {e}")


def run_memory_cognee_start_worker_extension() -> None:
    """Start the background rebuild worker after Agent Zero startup hooks finish."""
    if _cognee_module is None or not _init_done:
        PrintStyle.warning(
            "Cognee background worker not started because Cognee initialization "
            "did not complete."
        )
        return
    try:
        from .cognee_background import CogneeBackgroundWorker

        CogneeBackgroundWorker.get_instance().start()
    except BaseException as e:
        PrintStyle.error(f"Cognee background worker start failed ({type(e).__name__}): {e}")


def _disable_builtin_memory_plugin() -> None:
    """Ensure the FAISS-backed builtin memory plugin cannot run beside Cognee."""
    try:
        from helpers import plugins

        enabled = plugins.get_enabled_plugins(None)
        if "_memory" not in enabled:
            return

        plugins.toggle_plugin("_memory", False)
        PrintStyle.warning(
            "Disabled builtin _memory plugin because memory_cognee replaces it."
        )
    except Exception as e:
        PrintStyle.warning(f"Could not disable builtin _memory plugin: {e}")


def get_cognee():
    """Get initialized cognee module. Lazy-initializes on first call if needed."""
    if _cognee_module is None:
        configure_cognee()
    if not _init_done:
        ensure_tables_sync()
    if _init_error is not None:
        raise RuntimeError(f"Cognee initialization failed: {_init_error}") from _init_error
    if not _init_done:
        raise RuntimeError("Cognee initialization did not complete")
    if _cognee_module is None:
        raise RuntimeError("Cognee could not be initialized — check logs for details")
    return _cognee_module, _search_type_class
