"""Helpers for reading Cognee's dataset-scoped graph databases."""

from __future__ import annotations

from dataclasses import dataclass
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


def clear_cognee_graph_engine_cache() -> None:
    """Force graph reads to instantiate an engine for the current dataset context."""
    try:
        import importlib

        graph_module = importlib.import_module(
            "cognee.infrastructure.databases.graph.get_graph_engine"
        )
        cached_factory = getattr(graph_module, "_create_graph_engine", None)
        cache_clear = getattr(cached_factory, "cache_clear", None)
        if callable(cache_clear):
            cache_clear()
    except Exception as e:
        PrintStyle.warning(f"Could not clear Cognee graph engine cache: {e}")


async def read_dataset_graphs(
    cognee: Any,
    dataset_names: list[str] | None = None,
    *,
    skip_empty_data: bool = True,
    repair_unreadable: bool = False,
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
            error = str(e)
            if repair_unreadable and _is_unreadable_ladybug_error(e):
                await _repair_unreadable_dataset_graph(dataset, dataset_name)
                error = f"{error}; graph DB was purged and scheduled for rebuild"
            results.append(
                DatasetGraph(
                    dataset_id=dataset_id,
                    dataset_name=dataset_name,
                    data_count=data_count,
                    nodes=[],
                    edges=[],
                    error=error,
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
) -> tuple[list, list, bool]:
    from cognee.context_global_variables import set_database_global_context_variables
    from cognee.infrastructure.databases.graph import get_graph_engine

    owner_id = await _resolve_dataset_owner_id(dataset)
    clear_cognee_graph_engine_cache()
    async with set_database_global_context_variables(dataset.id, owner_id):
        clear_cognee_graph_engine_cache()
        graph_engine = await get_graph_engine()
        if await graph_engine.is_empty():
            return [], [], True
        nodes, edges = await graph_engine.get_graph_data()
        return list(nodes or []), list(edges or []), False


async def _resolve_dataset_owner_id(dataset: Any) -> Any:
    owner_id = getattr(dataset, "owner_id", None)
    if owner_id:
        return owner_id

    from cognee.modules.users.methods import get_default_user

    default_user = await get_default_user()
    return default_user.id


def _is_unreadable_ladybug_error(error: Exception) -> bool:
    message = str(error).lower()
    return (
        "could not map version_code" in message
        or "failed to initialize ladybug database" in message
    )


async def _repair_unreadable_dataset_graph(dataset: Any, dataset_name: str) -> None:
    try:
        from cognee.context_global_variables import graph_db_config

        graph_config = graph_db_config.get() or {}
        graph_path = graph_config.get("graph_file_path")
        if not graph_path:
            return

        await _purge_and_schedule_dataset_graph(graph_path, dataset_name)
    except Exception as e:
        PrintStyle.warning(
            f"Could not repair unreadable dataset graph ({dataset_name}): {e}"
        )


async def _purge_and_schedule_dataset_graph(graph_path: str, dataset_name: str) -> bool:
    from . import cognee_init

    purged = cognee_init.purge_unreadable_graph_db(graph_path)
    if not purged:
        return False

    reset = await cognee_init.reset_cognify_status_for_dataset_names([dataset_name])
    try:
        from .cognee_background import CogneeBackgroundWorker

        for name in reset or [dataset_name]:
            CogneeBackgroundWorker.get_instance().mark_dirty(name)
    except Exception as e:
        PrintStyle.warning(
            f"Could not mark repaired dataset graph dirty ({dataset_name}): {e}"
        )
    return True

