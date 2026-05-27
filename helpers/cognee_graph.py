"""Helpers for reading Cognee's dataset-scoped graph databases."""

from __future__ import annotations

from dataclasses import dataclass
import os
import time
from typing import Any

from helpers.print_style import PrintStyle


@dataclass
class DatasetGraph:
    dataset_id: str
    dataset_name: str
    data_count: int | None
    nodes: list
    edges: list
    graph_empty: bool | None = None
    error: str | None = None


async def read_dataset_graphs(
    cognee: Any,
    dataset_names: list[str] | None = None,
    *,
    skip_empty_data: bool = True,
    repair_unreadable: bool = False,
    include_graph_data: bool = True,
) -> list[DatasetGraph]:
    """Read graph data from Cognee's per-dataset graph stores.

    Cognee 1.x stores graph data in dataset-specific graph databases when backend
    access control is enabled. Plain ``get_graph_engine()`` reads the global graph
    context and can report an empty graph even after successful cognify.
    """
    datasets_api = getattr(cognee, "datasets", None)
    if datasets_api is None:
        return [
            DatasetGraph(
                dataset_id="",
                dataset_name="unknown",
                data_count=None,
                nodes=[],
                edges=[],
                error="Cognee datasets API is unavailable",
            )
        ]

    requested_names = {name for name in (dataset_names or []) if name}
    try:
        all_datasets = await datasets_api.list_datasets()
    except Exception as e:
        PrintStyle.warning(f"Could not list Cognee datasets for graph read: {e}")
        return [
            DatasetGraph(
                dataset_id="",
                dataset_name="unknown",
                data_count=None,
                nodes=[],
                edges=[],
                error=f"dataset list failed: {e}",
            )
        ]

    selected = []
    for dataset in all_datasets:
        name = str(getattr(dataset, "name", "") or getattr(dataset, "id", ""))
        if requested_names and name not in requested_names:
            continue
        selected.append(dataset)

    results: list[DatasetGraph] = []
    for dataset in selected:
        dataset_id = str(getattr(dataset, "id", "") or "")
        dataset_name = str(getattr(dataset, "name", "") or dataset_id)

        data_count: int | None
        try:
            data_items = await datasets_api.list_data(dataset.id)
            data_count = len(data_items or [])
        except Exception as e:
            PrintStyle.warning(
                f"Could not list Cognee data for graph read ({dataset_name}): {e}"
            )
            data_count = None

        if skip_empty_data and data_count == 0:
            continue

        try:
            nodes, edges, graph_empty = await _read_single_dataset_graph(
                dataset,
                repair_unreadable=repair_unreadable,
                include_graph_data=include_graph_data,
            )
            results.append(
                DatasetGraph(
                    dataset_id=dataset_id,
                    dataset_name=dataset_name,
                    data_count=data_count,
                    nodes=nodes,
                    edges=edges,
                    graph_empty=graph_empty,
                )
            )
        except Exception as e:
            results.append(
                DatasetGraph(
                    dataset_id=dataset_id,
                    dataset_name=dataset_name,
                    data_count=data_count,
                    nodes=[],
                    edges=[],
                    error=str(e),
                )
            )

    missing_names = sorted(requested_names - {result.dataset_name for result in results})
    for missing_name in missing_names:
        results.append(
            DatasetGraph(
                dataset_id="",
                dataset_name=missing_name,
                data_count=0,
                nodes=[],
                edges=[],
                error=None,
            )
        )

    return results


async def _read_single_dataset_graph(
    dataset: Any,
    *,
    repair_unreadable: bool = False,
    include_graph_data: bool = True,
) -> tuple[list, list, bool]:
    from cognee.context_global_variables import set_database_global_context_variables
    from cognee.infrastructure.databases.graph import get_graph_engine

    owner_id = await _resolve_dataset_owner_id(dataset)
    async with set_database_global_context_variables(dataset.id, owner_id):
        try:
            return await _read_current_graph_engine(
                get_graph_engine,
                include_graph_data=include_graph_data,
            )
        except Exception as error:
            if repair_unreadable and _repair_corrupt_kuzu_wal(error):
                return await _read_current_graph_engine(
                    get_graph_engine,
                    include_graph_data=include_graph_data,
                )
            raise


async def _read_current_graph_engine(
    get_graph_engine: Any,
    *,
    include_graph_data: bool,
) -> tuple[list, list, bool]:
    graph_engine = await get_graph_engine()
    if await graph_engine.is_empty():
        return [], [], True
    if not include_graph_data:
        return [], [], False
    nodes, edges = await graph_engine.get_graph_data()
    return list(nodes or []), list(edges or []), False


def _repair_corrupt_kuzu_wal(error: Exception) -> bool:
    message = str(error).lower()
    if "corrupted wal file" not in message and "invalid wal record" not in message:
        return False

    try:
        from cognee.infrastructure.databases.graph.config import get_graph_context_config
        from cognee.infrastructure.databases.graph.get_graph_engine import (
            evict_graph_engine,
        )

        config = get_graph_context_config()
        config_kwargs = _graph_config_to_kwargs(config)
        graph_file_path = str(config_kwargs.get("graph_file_path") or "").strip()
        if not graph_file_path:
            PrintStyle.warning(
                "Cognee graph WAL repair skipped: graph_file_path is unavailable"
            )
            return False

        wal_paths = _candidate_kuzu_wal_paths(graph_file_path)
        wal_path = next((path for path in wal_paths if os.path.exists(path)), "")
        if not wal_path:
            graph_path = _existing_rebuildable_kuzu_graph_path(graph_file_path)
            if not graph_path:
                PrintStyle.warning(
                    "Cognee graph WAL repair skipped: WAL file not found at "
                    f"{', '.join(wal_paths)}"
                )
                return False

            eviction_kwargs = _graph_repair_eviction_kwargs(
                config_kwargs,
                graph_file_path,
                graph_path,
            )
            for kwargs in eviction_kwargs:
                try:
                    evict_graph_engine(**kwargs)
                except Exception as evict_error:
                    PrintStyle.warning(
                        "Cognee graph engine eviction failed before graph "
                        f"quarantine: {evict_error}"
                    )

            moved_paths = _quarantine_graph_store_files(
                graph_path,
                include_graph=True,
            )
            for kwargs in eviction_kwargs:
                try:
                    evict_graph_engine(**kwargs)
                except Exception as evict_error:
                    PrintStyle.warning(
                        "Cognee graph engine eviction failed after graph "
                        f"quarantine: {evict_error}"
                    )
            PrintStyle.warning(
                "Cognee graph store was unreadable after WAL recovery; moved "
                "derived graph file(s) aside: "
                f"{', '.join(moved_paths)}"
            )
            return True

        repair_graph_path = wal_path[:-4] if wal_path.endswith(".wal") else graph_file_path
        eviction_kwargs = _graph_repair_eviction_kwargs(
            config_kwargs,
            graph_file_path,
            repair_graph_path,
        )

        for kwargs in eviction_kwargs:
            try:
                evict_graph_engine(**kwargs)
            except Exception as evict_error:
                PrintStyle.warning(
                    f"Cognee graph engine eviction failed before WAL repair: {evict_error}"
                )

        moved_paths = _quarantine_graph_store_files(
            repair_graph_path,
            include_graph=False,
        )
        for kwargs in eviction_kwargs:
            try:
                evict_graph_engine(**kwargs)
            except Exception as evict_error:
                PrintStyle.warning(
                    f"Cognee graph engine eviction failed after WAL repair: {evict_error}"
                )
        PrintStyle.warning(
            "Cognee graph WAL was corrupt; moved unreadable WAL aside "
            f"for {repair_graph_path}: {', '.join(moved_paths)}"
        )
        return True
    except Exception as repair_error:
        PrintStyle.warning(f"Cognee graph WAL repair failed: {repair_error}")
        return False


def _candidate_kuzu_wal_paths(graph_file_path: str) -> list[str]:
    paths = []
    if graph_file_path:
        paths.append(f"{graph_file_path}.wal")

    system_root = os.environ.get("SYSTEM_ROOT_DIRECTORY", "").strip()
    if system_root:
        global_graph_path = os.path.join(
            system_root,
            "databases",
            "cognee_graph_kuzu",
        )
        paths.append(f"{global_graph_path}.wal")

    unique_paths: list[str] = []
    seen = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        unique_paths.append(path)
    return unique_paths


def _existing_rebuildable_kuzu_graph_path(graph_file_path: str) -> str:
    global_graph_path = _global_kuzu_graph_path()
    candidates = []
    if global_graph_path:
        candidates.append(global_graph_path)
    if graph_file_path:
        candidates.append(graph_file_path)
    for path in candidates:
        if path and any(os.path.exists(candidate) for candidate in _graph_store_paths(path, True)):
            return path
    return ""


def _global_kuzu_graph_path() -> str:
    system_root = os.environ.get("SYSTEM_ROOT_DIRECTORY", "").strip()
    if not system_root:
        return ""
    return os.path.join(system_root, "databases", "cognee_graph_kuzu")


def _graph_repair_eviction_kwargs(
    config_kwargs: dict[str, Any],
    context_graph_path: str,
    repair_graph_path: str,
) -> list[dict[str, Any]]:
    eviction_kwargs = [dict(config_kwargs)]
    if repair_graph_path != context_graph_path:
        fallback_kwargs = dict(config_kwargs)
        fallback_kwargs["graph_file_path"] = repair_graph_path
        eviction_kwargs.append(fallback_kwargs)
    return eviction_kwargs


def _graph_store_paths(graph_path: str, include_graph: bool) -> list[str]:
    paths = []
    if include_graph:
        paths.append(graph_path)
    paths.extend(
        [
            f"{graph_path}.wal",
            f"{graph_path}.wal.checkpoint",
            f"{graph_path}.shadow",
        ]
    )
    return paths


def _quarantine_graph_store_files(graph_path: str, *, include_graph: bool) -> list[str]:
    suffix = f".corrupt.{int(time.time())}.{os.getpid()}"
    moved = []
    for path in _graph_store_paths(graph_path, include_graph):
        if not os.path.exists(path):
            continue
        repaired_path = f"{path}{suffix}"
        os.replace(path, repaired_path)
        moved.append(f"{path} -> {repaired_path}")
    return moved


def _graph_config_to_kwargs(config: Any) -> dict[str, Any]:
    if isinstance(config, dict):
        return dict(config)
    model_dump = getattr(config, "model_dump", None)
    if callable(model_dump):
        return dict(model_dump())
    dict_method = getattr(config, "dict", None)
    if callable(dict_method):
        return dict(dict_method())
    return {
        key: value
        for key, value in vars(config).items()
        if not key.startswith("_")
    }


async def _resolve_dataset_owner_id(dataset: Any) -> Any:
    owner_id = getattr(dataset, "owner_id", None)
    if owner_id:
        return owner_id

    from cognee.modules.users.methods import get_default_user

    default_user = await get_default_user()
    return default_user.id
