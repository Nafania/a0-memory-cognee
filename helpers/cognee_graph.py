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
) -> tuple[list, list, bool]:
    from cognee.context_global_variables import set_database_global_context_variables
    from cognee.infrastructure.databases.graph import get_graph_engine

    owner_id = await _resolve_dataset_owner_id(dataset)
    async with set_database_global_context_variables(dataset.id, owner_id):
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
