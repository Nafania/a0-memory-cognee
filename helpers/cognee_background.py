"""
Background worker that periodically runs Cognee's knowledge graph building pipeline.
Integrates with Agent Zero's DeferredTask system.
"""

import asyncio
import gc
import inspect
import json
import multiprocessing
import os
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from types import SimpleNamespace
from typing import Any, Callable, Set

from helpers.defer import DeferredTask, THREAD_BACKGROUND
from helpers.print_style import PrintStyle
from .cognee_graph import (
    _is_repairable_graph_store_error,
    _repair_corrupt_kuzu_wal,
    read_dataset_graphs,
)
from .cognee_init import get_cognee_setting
from .cognee_ops import run_cognee_operation


_COGNEE_CHILD_PROCESS_PIDS: set[int] = set()


def _clean_vector_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(value).strip()


def _normalize_belongs_to_set(raw: Any) -> list[str]:
    if isinstance(raw, list):
        values = []
        for item in raw:
            if isinstance(item, str):
                values.append(item.strip())
            elif isinstance(item, dict):
                name = item.get("name") or item.get("id")
                if name:
                    values.append(str(name).strip())
            else:
                name = getattr(item, "name", None) or getattr(item, "id", None)
                values.append(str(name if name is not None else item).strip())
        return [value for value in values if value]
    if isinstance(raw, str):
        return [part.strip() for part in raw.split(",") if part.strip()]
    return []


def _merge_belongs_to_sets(*sets: Any) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for raw in sets:
        for value in _normalize_belongs_to_set(raw):
            if value in seen:
                continue
            seen.add(value)
            merged.append(value)
    return merged


def _lancedb_id_where_clause(ids: list[str]) -> str:
    escaped_ids = [id_.replace("'", "''") for id_ in ids]
    if len(escaped_ids) == 1:
        return f"id = '{escaped_ids[0]}'"
    id_list = ", ".join(f"'{id_}'" for id_ in escaped_ids)
    return f"id IN ({id_list})"


def _open_seen_manifest(seen_path: str | None) -> sqlite3.Connection | None:
    if not seen_path:
        return None
    conn = sqlite3.connect(seen_path)
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS seen_vectors ("
        "collection TEXT NOT NULL, "
        "id TEXT NOT NULL, "
        "text TEXT NOT NULL DEFAULT '', "
        "belongs_to_set TEXT NOT NULL, "
        "written INTEGER NOT NULL DEFAULT 0, "
        "PRIMARY KEY(collection, id)"
        ")"
    )
    columns = {
        row[1]
        for row in conn.execute("PRAGMA table_info(seen_vectors)").fetchall()
    }
    if "text" not in columns:
        conn.execute("ALTER TABLE seen_vectors ADD COLUMN text TEXT NOT NULL DEFAULT ''")
    if "written" not in columns:
        conn.execute("ALTER TABLE seen_vectors ADD COLUMN written INTEGER NOT NULL DEFAULT 0")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS seen_manifest_meta ("
        "key TEXT PRIMARY KEY, "
        "value TEXT NOT NULL"
        ")"
    )
    return conn


def _set_seen_manifest_prepared(conn: sqlite3.Connection | None) -> None:
    if conn is None:
        return
    conn.execute(
        "INSERT INTO seen_manifest_meta(key, value) VALUES ('prepared', '1') "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value"
    )
    conn.commit()


def _is_seen_manifest_prepared(conn: sqlite3.Connection | None) -> bool:
    if conn is None:
        return False
    row = conn.execute(
        "SELECT value FROM seen_manifest_meta WHERE key = 'prepared'"
    ).fetchone()
    return bool(row and str(row[0]) == "1")


def _decode_belongs_json(value: Any) -> list[str]:
    try:
        belongs_to_set = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        belongs_to_set = []
    return _normalize_belongs_to_set(belongs_to_set)


def _read_seen_manifest(
    conn: sqlite3.Connection | None,
    collection_name: str,
    ids: list[str],
) -> dict[str, list[str]]:
    if conn is None or not ids:
        return {}
    placeholders = ", ".join("?" for _ in ids)
    rows = conn.execute(
        "SELECT id, belongs_to_set FROM seen_vectors "
        f"WHERE collection = ? AND id IN ({placeholders})",
        [collection_name, *ids],
    ).fetchall()
    seen: dict[str, list[str]] = {}
    for row_id, belongs_json in rows:
        seen[str(row_id)] = _decode_belongs_json(belongs_json)
    return seen


def _read_manifest_entries(
    conn: sqlite3.Connection | None,
    collection_name: str,
    ids: list[str],
) -> dict[str, dict[str, Any]]:
    if conn is None or not ids:
        return {}
    placeholders = ", ".join("?" for _ in ids)
    rows = conn.execute(
        "SELECT id, text, belongs_to_set, written FROM seen_vectors "
        f"WHERE collection = ? AND id IN ({placeholders})",
        [collection_name, *ids],
    ).fetchall()
    return {
        str(row_id): {
            "text": str(text or ""),
            "belongs_to_set": _decode_belongs_json(belongs_json),
            "written": bool(written),
        }
        for row_id, text, belongs_json, written in rows
    }


def _claim_manifest_entries(
    conn: sqlite3.Connection | None,
    collection_name: str,
    ids: list[str],
) -> dict[str, dict[str, Any]]:
    entries = _read_manifest_entries(conn, collection_name, ids)
    if conn is None or not entries:
        return entries

    claim_ids = [row_id for row_id, entry in entries.items() if not entry["written"]]
    if claim_ids:
        placeholders = ", ".join("?" for _ in claim_ids)
        conn.execute(
            "UPDATE seen_vectors SET written = 1 "
            f"WHERE collection = ? AND id IN ({placeholders})",
            [collection_name, *claim_ids],
        )
        conn.commit()

    claimed = set(claim_ids)
    return {
        row_id: entry
        for row_id, entry in entries.items()
        if row_id in claimed
    }


def _upsert_manifest_entries(
    conn: sqlite3.Connection | None,
    collection_name: str,
    rows: list[tuple[str, str, list[str]]],
) -> None:
    if conn is None or not rows:
        return

    existing = _read_manifest_entries(
        conn,
        collection_name,
        [str(row_id) for row_id, _text, _belongs_to_set in rows],
    )
    upserts = []
    for row_id, text, belongs_to_set in rows:
        row_id = str(row_id)
        text = str(text or "")
        prior = existing.get(row_id)
        if prior:
            belongs_to_set = _merge_belongs_to_sets(
                prior.get("belongs_to_set"),
                belongs_to_set,
            )
        upserts.append(
            (
                collection_name,
                row_id,
                text,
                json.dumps(_normalize_belongs_to_set(belongs_to_set), ensure_ascii=False),
            )
        )

    conn.executemany(
        "INSERT INTO seen_vectors(collection, id, text, belongs_to_set) "
        "VALUES (?, ?, ?, ?) "
        "ON CONFLICT(collection, id) DO UPDATE SET "
        "text=excluded.text, "
        "belongs_to_set=excluded.belongs_to_set",
        upserts,
    )
    conn.commit()


def _write_seen_manifest(
    conn: sqlite3.Connection | None,
    collection_name: str,
    rows: list[tuple[str, list[str]]],
) -> None:
    if conn is None or not rows:
        return
    conn.executemany(
        "INSERT INTO seen_vectors(collection, id, text, belongs_to_set) VALUES (?, ?, '', ?) "
        "ON CONFLICT(collection, id) DO UPDATE SET belongs_to_set=excluded.belongs_to_set",
        [
            (collection_name, row_id, json.dumps(belongs_to_set, ensure_ascii=False))
            for row_id, belongs_to_set in rows
        ],
    )
    conn.commit()


def _supports_lancedb_delete_add_upsert(vector_engine: Any) -> bool:
    if str(getattr(vector_engine, "name", "")) != "LanceDB":
        return False
    required = [
        "create_vector_index",
        "get_collection",
        "embed_data",
        "get_data_point_schema",
        "_make_lance_datapoint_cls",
        "_records_for_write",
    ]
    return all(callable(getattr(vector_engine, name, None)) for name in required)


def _coerce_indexed_fields(raw: Any) -> list[str]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            raw = [raw]
    if not isinstance(raw, list):
        return []
    return [str(field or "").strip() for field in raw if str(field or "").strip()]


def _node_vector_entries(row: Any) -> list[tuple[str, str, str, list[str]]]:
    attributes = dict(getattr(row, "attributes", None) or {})
    type_name = str(getattr(row, "type", None) or attributes.get("type") or "").strip()
    if not type_name:
        return []

    indexed_fields = _coerce_indexed_fields(
        getattr(row, "indexed_fields", None)
        or attributes.get("metadata", {}).get("index_fields")
    )
    if not indexed_fields:
        return []

    node_id = (
        attributes.get("id")
        or getattr(row, "slug", None)
        or getattr(row, "id", None)
    )
    if node_id is None:
        return []

    belongs_to_set = _normalize_belongs_to_set(
        attributes.get("belongs_to_set") or attributes.get("source_node_set")
    )
    entries: list[tuple[str, str, str, list[str]]] = []
    for field in indexed_fields:
        text = _clean_vector_text(attributes.get(field))
        if not text:
            continue
        entries.append(
            (
                f"{type_name}_{field}",
                str(node_id),
                text,
                belongs_to_set,
            )
        )
    return entries


def _decode_json_column(value: Any, fallback: Any) -> Any:
    if value is None:
        return fallback
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return fallback
    return fallback


def _graph_node_from_raw(row: Any) -> SimpleNamespace:
    mapping = dict(row)
    return SimpleNamespace(
        id=str(mapping.get("id") or ""),
        slug=str(mapping.get("slug") or ""),
        type=str(mapping.get("type") or ""),
        indexed_fields=_decode_json_column(mapping.get("indexed_fields"), []),
        attributes=_decode_json_column(mapping.get("attributes"), {}),
    )


async def _fetch_graph_node_rows(
    session: Any,
    dataset_id: str,
    *,
    offset: int,
    limit: int,
) -> list[SimpleNamespace]:
    """Fetch graph node rows without SQLAlchemy UUID result processors.

    Some production backups contain malformed UUID values in Cognee graph rows.
    ORM-loading ``Node`` applies SQLAlchemy's UUID processor before our code can
    inspect the row, which can abort the entire rebuild. Raw SQL keeps rebuild
    tolerant of one malformed database value while preserving normal vector
    payload extraction from JSON attributes.
    """
    from sqlalchemy import text

    result = await session.execute(
        text(
            "SELECT CAST(id AS TEXT) AS id, CAST(slug AS TEXT) AS slug, "
            "type, indexed_fields, attributes "
            "FROM nodes "
            "WHERE CAST(dataset_id AS TEXT) = :dataset_id "
            "ORDER BY CAST(id AS TEXT) "
            "LIMIT :limit OFFSET :offset"
        ),
        {"dataset_id": str(dataset_id), "offset": int(offset), "limit": int(limit)},
    )
    return [_graph_node_from_raw(row) for row in result.mappings().all()]


async def _fetch_edge_relationship_counts(
    session: Any,
    dataset_id: str,
) -> list[tuple[str, int]]:
    from sqlalchemy import text

    result = await session.execute(
        text(
            "SELECT relationship_name, COUNT(*) AS count "
            "FROM edges "
            "WHERE CAST(dataset_id AS TEXT) = :dataset_id "
            "GROUP BY relationship_name"
        ),
        {"dataset_id": str(dataset_id)},
    )
    return [
        (str(row["relationship_name"]), int(row["count"] or 0))
        for row in result.mappings().all()
        if row.get("relationship_name")
    ]


async def _get_dataset_id_text(session: Any, dataset: str) -> str | None:
    from sqlalchemy import text

    result = await session.execute(
        text("SELECT CAST(id AS TEXT) AS id FROM datasets WHERE name = :name LIMIT 1"),
        {"name": str(dataset)},
    )
    row = result.mappings().first()
    if not row:
        return None
    value = row.get("id")
    return str(value) if value is not None else None


async def _index_data_points_for_vector_rebuild(
    data_points: list[Any],
    *,
    vector_engine: Any,
    max_batch_size: int,
    seen_path: str | None = None,
) -> None:
    if not data_points:
        return
    if _supports_lancedb_delete_add_upsert(vector_engine):
        await _lancedb_delete_add_upsert_index_data_points(
            data_points,
            vector_engine=vector_engine,
            max_batch_size=max_batch_size,
            seen_path=seen_path,
        )
        return

    from cognee.tasks.storage.index_data_points import index_data_points

    await index_data_points(data_points, vector_engine=vector_engine)


async def _lancedb_delete_add_upsert_index_data_points(
    data_points: list[Any],
    *,
    vector_engine: Any,
    max_batch_size: int,
    seen_path: str | None = None,
) -> None:
    """Upsert vector rebuild rows without LanceDB merge_insert.

    Cognee's normal LanceDB path uses merge_insert to preserve unique ids and
    merge belongs_to_set tags. On large existing tables that path can be killed
    by the OOM killer during embedding-model rebuilds. This rebuild path starts
    from a purged vector store, so delete+typed-add is enough to preserve the
    same id/upsert semantics while keeping memory bounded per batch.
    """
    from cognee.infrastructure.databases.vector.lancedb.LanceDBAdapter import IndexSchema

    grouped: dict[tuple[str, str], list[Any]] = {}
    for data_point in data_points:
        metadata = getattr(data_point, "metadata", None) or {}
        index_fields = metadata.get("index_fields") or []
        type_name = type(data_point).__name__
        for field_name in index_fields:
            field_name = str(field_name or "").strip()
            if not field_name:
                continue
            text = _clean_vector_text(getattr(data_point, field_name, None))
            if not text:
                continue
            grouped.setdefault((type_name, field_name), []).append(
                IndexSchema(
                    id=str(getattr(data_point, "id")),
                    text=text,
                    belongs_to_set=_normalize_belongs_to_set(
                        getattr(data_point, "belongs_to_set", None)
                    ),
                )
            )

    vector_batch_size = int(getattr(vector_engine.embedding_engine, "get_batch_size")())
    write_batch_size = max(1, min(max_batch_size, vector_batch_size, 256))
    vector_size = vector_engine.embedding_engine.get_vector_size()
    payload_schema = vector_engine.get_data_point_schema(IndexSchema)
    lance_data_point_cls = vector_engine._make_lance_datapoint_cls(
        payload_schema,
        vector_size,
    )

    seen_conn = _open_seen_manifest(seen_path)
    try:
        manifest_prepared = _is_seen_manifest_prepared(seen_conn)
        for (type_name, field_name), points in grouped.items():
            collection_name = f"{type_name}_{field_name}"
            await vector_engine.create_vector_index(type_name, field_name)
            collection = await vector_engine.get_collection(collection_name)

            for start in range(0, len(points), write_batch_size):
                batch = points[start : start + write_batch_size]
                deduped: dict[str, Any] = {}
                for point in batch:
                    existing = deduped.get(point.id)
                    if existing is not None:
                        point.belongs_to_set = _merge_belongs_to_sets(
                            getattr(existing, "belongs_to_set", None),
                            getattr(point, "belongs_to_set", None),
                        )
                    deduped[point.id] = point
                batch = list(deduped.values())

                ids = [str(point.id) for point in batch]
                existing_belongs_to_set: dict[str, list[str]] = {}
                if manifest_prepared:
                    claimed = _claim_manifest_entries(seen_conn, collection_name, ids)
                    claimed_batch = []
                    for point in batch:
                        entry = claimed.get(str(point.id))
                        if not entry:
                            continue
                        point.text = str(entry.get("text") or point.text)
                        point.belongs_to_set = _normalize_belongs_to_set(
                            entry.get("belongs_to_set")
                        )
                        claimed_batch.append(point)
                    batch = claimed_batch
                    if not batch:
                        continue
                else:
                    existing_belongs_to_set = _read_seen_manifest(
                        seen_conn,
                        collection_name,
                        ids,
                    )
                if seen_conn is None:
                    where_clause = _lancedb_id_where_clause(ids)
                    existing_rows = await collection.query().where(where_clause).to_list()
                    for row in existing_rows:
                        payload = row.get("payload") or {}
                        existing_belongs_to_set[str(row["id"])] = _normalize_belongs_to_set(
                            payload.get("belongs_to_set")
                        )
                if existing_belongs_to_set:
                    await collection.delete(
                        _lancedb_id_where_clause(list(existing_belongs_to_set))
                    )

                vectors = await vector_engine.embed_data([point.text for point in batch])
                lance_rows = []
                seen_updates: list[tuple[str, list[str]]] = []
                for index, point in enumerate(batch):
                    properties = payload_schema.model_validate(point.model_dump()).model_dump()
                    prior = existing_belongs_to_set.get(str(point.id))
                    if prior:
                        properties["belongs_to_set"] = _merge_belongs_to_sets(
                            prior,
                            properties.get("belongs_to_set"),
                        )
                    belongs_to_set = _normalize_belongs_to_set(
                        properties.get("belongs_to_set")
                    )
                    lance_rows.append(
                        lance_data_point_cls(
                            id=str(point.id),
                            vector=vectors[index],
                            payload=properties,
                        )
                    )
                    seen_updates.append((str(point.id), belongs_to_set))
                await collection.add(vector_engine._records_for_write(lance_rows))
                if not manifest_prepared:
                    _write_seen_manifest(seen_conn, collection_name, seen_updates)
    finally:
        if seen_conn is not None:
            seen_conn.close()


def _child_pid(child: Any) -> int | None:
    pid = getattr(child, "pid", None)
    if pid is None:
        return None
    try:
        return int(pid)
    except (TypeError, ValueError):
        return None


def _active_child_pids() -> set[int]:
    return {
        pid
        for child in multiprocessing.active_children()
        if (pid := _child_pid(child)) is not None
    }


def _cleanup_cognee_child_processes(
    label: str,
    *,
    baseline_pids: set[int] | None = None,
) -> None:
    """Release Cognee/FastEmbed multiprocessing children between dataset rebuilds."""
    baseline = baseline_pids or set()
    children = []
    for child in multiprocessing.active_children():
        pid = _child_pid(child)
        if pid in baseline and pid not in _COGNEE_CHILD_PROCESS_PIDS:
            continue
        if pid is not None:
            _COGNEE_CHILD_PROCESS_PIDS.add(pid)
        children.append(child)
    if not children:
        gc.collect()
        return

    child_descriptions = [
        f"{getattr(child, 'name', '?')}:{getattr(child, 'pid', '?')}"
        for child in children
    ]
    for child in children:
        try:
            if child.is_alive():
                child.terminate()
        except Exception:
            pass

    still_alive = []
    for child in children:
        try:
            child.join(timeout=5)
            if child.is_alive():
                still_alive.append(child)
        except Exception:
            pass

    for child in still_alive:
        try:
            child.kill()
            child.join(timeout=2)
        except Exception:
            pass

    for child in children:
        pid = _child_pid(child)
        if pid is None:
            continue
        try:
            if not child.is_alive():
                _COGNEE_CHILD_PROCESS_PIDS.discard(pid)
        except Exception:
            pass

    PrintStyle.warning(
        "Cleaned up Cognee child process(es) after dataset rebuild "
        f"{label}: {child_descriptions}"
    )
    gc.collect()


def _positive_int_or_none(value) -> int | None:
    if value in (None, ""):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _embedding_config_rebuild_needed() -> bool:
    try:
        from .cognee_init import _embedding_config_rebuild_needed as rebuild_needed

        return bool(rebuild_needed())
    except Exception as e:
        PrintStyle.warning(f"Could not read Cognee embedding rebuild state: {e}")
        return False


async def _dataset_has_existing_graph(dataset: str) -> bool:
    """Return true when Cognee already has graph rows for a dataset."""
    try:
        return await _get_existing_graph_dataset_node_count(dataset) > 0
    except Exception as e:
        PrintStyle.warning(
            f"Could not inspect existing Cognee graph for dataset {dataset}: {e}"
        )
        return False


async def _prepare_vector_rebuild_manifest(
    dataset: str,
    seen_path: str | None,
    *,
    page_size: int = 5000,
) -> int:
    """Precompute final vector rows so rebuild writes each vector id once."""
    if not seen_path:
        return 0

    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.engine.utils.generate_edge_id import generate_edge_id

    conn = _open_seen_manifest(seen_path)
    if conn is None:
        return 0

    prepared_entries = 0
    try:
        db_engine = get_relational_engine()
        async with db_engine.get_async_session() as session:
            dataset_id = await _get_dataset_id_text(session, dataset)
            if dataset_id is None:
                raise RuntimeError(f"Cognee dataset not found: {dataset}")

            offset = 0
            while True:
                rows = await _fetch_graph_node_rows(
                    session,
                    dataset_id,
                    offset=offset,
                    limit=page_size,
                )
                if not rows:
                    break

                by_collection: dict[str, list[tuple[str, str, list[str]]]] = {}
                for row in rows:
                    for collection_name, row_id, text, belongs_to_set in _node_vector_entries(row):
                        by_collection.setdefault(collection_name, []).append(
                            (row_id, text, belongs_to_set)
                        )
                for collection_name, entries in by_collection.items():
                    _upsert_manifest_entries(conn, collection_name, entries)
                    prepared_entries += len(entries)
                offset += len(rows)

            edge_rows = await _fetch_edge_relationship_counts(session, dataset_id)
            edge_entries = [
                (
                    generate_edge_id(edge_id=str(relationship)),
                    str(relationship),
                    [],
                )
                for relationship, count in edge_rows
                if relationship
            ]
            _upsert_manifest_entries(
                conn,
                "EdgeType_relationship_name",
                edge_entries,
            )
            prepared_entries += len(edge_entries)

        _set_seen_manifest_prepared(conn)
        return prepared_entries
    finally:
        conn.close()


async def _get_existing_graph_dataset_node_count(dataset: str) -> int:
    from sqlalchemy import text
    from cognee.infrastructure.databases.relational import get_relational_engine

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        dataset_id = await _get_dataset_id_text(session, dataset)
        if dataset_id is None:
            return 0
        count = (
            await session.execute(
                text(
                    "SELECT COUNT(*) FROM nodes "
                    "WHERE CAST(dataset_id AS TEXT) = :dataset_id"
                ),
                {"dataset_id": dataset_id},
            )
        ).scalar_one()
        return int(count or 0)


async def _rebuild_embedding_vectors_from_existing_graph(
    dataset: str,
    *,
    batch_size: int = 128,
    process_chunk_size: int = 5000,
    progress_callback: Callable[[dict[str, int | str]], None] | None = None,
) -> dict[str, int]:
    """Recreate vector indexes from persisted Cognee graph rows.

    Embedding-model changes do not require extracting the graph with the LLM again.
    Re-indexing nodes and relationship types keeps the business flow on
    GRAPH_COMPLETION while avoiding a full cognify pass over large production graphs.
    """
    node_count = await _get_existing_graph_dataset_node_count(dataset)
    if node_count <= 0:
        raise RuntimeError(f"Cognee dataset has no graph nodes: {dataset}")

    process_chunk_size = max(batch_size, process_chunk_size)
    indexed_nodes = 0
    skipped_nodes = 0
    offset = 0
    PrintStyle.standard(
        "Cognee vector rebuild started from existing graph for "
        f"{dataset}: rows={node_count}, batch_size={batch_size}, "
        f"process_chunk_size={process_chunk_size}"
    )

    def emit_progress(progress: dict[str, int | str]) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(progress)
        except Exception as e:
            PrintStyle.warning(f"Cognee vector rebuild progress callback failed: {e}")

    seen_path = os.path.join(
        tempfile.gettempdir(),
        f"a0-cognee-vector-rebuild-seen-{uuid.uuid4().hex}.sqlite3",
    )
    try:
        manifest_entries = await _prepare_vector_rebuild_manifest(dataset, seen_path)
        PrintStyle.standard(
            "Cognee vector rebuild manifest prepared for "
            f"{dataset}: vector_entries={manifest_entries}"
        )
        emit_progress(
            {
                "phase": "manifest",
                "rows_done": 0,
                "rows_total": node_count,
                "indexed_vectors": 0,
                "skipped_rows": 0,
            }
        )
        while offset < node_count:
            limit = min(process_chunk_size, node_count - offset)
            counts = await _run_vector_rebuild_chunk_subprocess(
                dataset,
                offset=offset,
                limit=limit,
                batch_size=batch_size,
                include_edges=False,
                seen_path=seen_path,
            )
            rows = int(counts.get("rows", 0) or 0)
            if rows <= 0:
                raise RuntimeError(
                    "Cognee vector rebuild made no progress for "
                    f"{dataset} at offset {offset}"
                )
            indexed_nodes += int(counts.get("nodes", 0) or 0)
            skipped_nodes += int(counts.get("skipped_nodes", 0) or 0)
            offset += rows
            PrintStyle.standard(
                "Cognee vector rebuild progress for "
                f"{dataset}: rows={offset}/{node_count}, "
                f"indexed_vectors={indexed_nodes}, "
                f"skipped_rows={skipped_nodes}"
            )
            emit_progress(
                {
                    "phase": "nodes",
                    "rows_done": offset,
                    "rows_total": node_count,
                    "indexed_vectors": indexed_nodes,
                    "skipped_rows": skipped_nodes,
                }
            )

        edge_counts = await _run_vector_rebuild_chunk_subprocess(
            dataset,
            offset=0,
            limit=0,
            batch_size=batch_size,
            include_edges=True,
            seen_path=seen_path,
        )
        emit_progress(
            {
                "phase": "edges",
                "rows_done": offset,
                "rows_total": node_count,
                "indexed_vectors": indexed_nodes,
                "skipped_rows": skipped_nodes,
                "edge_types": int(edge_counts.get("edge_types", 0) or 0),
            }
        )
    finally:
        for suffix in ("", "-wal", "-shm"):
            try:
                os.remove(f"{seen_path}{suffix}")
            except FileNotFoundError:
                pass

    return {
        "nodes": indexed_nodes,
        "skipped_nodes": skipped_nodes,
        "edge_types": int(edge_counts.get("edge_types", 0) or 0),
    }


async def _run_vector_rebuild_chunk_subprocess(
    dataset: str,
    *,
    offset: int,
    limit: int,
    batch_size: int,
    include_edges: bool,
    seen_path: str | None = None,
) -> dict[str, int]:
    result_path = os.path.join(
        tempfile.gettempdir(),
        f"a0-cognee-vector-rebuild-{uuid.uuid4().hex}.json",
    )
    args = [
        sys.executable,
        "-m",
        "usr.plugins.memory_cognee.helpers.cognee_background",
        "vector-rebuild-chunk",
        dataset,
        str(offset),
        str(limit),
        str(batch_size),
        "1" if include_edges else "0",
        result_path,
    ]
    if seen_path is not None:
        args.append(seen_path)
    env = os.environ.copy()
    # The chunk process is already the lifecycle boundary. Running another
    # LanceDB subprocess inside it leaves orphaned child processes under PID 1
    # after each chunk in Docker.
    env["VECTOR_DB_SUBPROCESS_ENABLED"] = "false"

    try:
        chunk_timeout = float(get_cognee_setting("cognee_operation_timeout_seconds", 1800))
        proc = await asyncio.to_thread(
            subprocess.run,
            args,
            cwd=os.getcwd(),
            env=env,
            text=True,
            capture_output=True,
            timeout=chunk_timeout,
        )
        result: dict[str, Any] = {}
        if os.path.exists(result_path):
            with open(result_path, "r", encoding="utf-8") as result_file:
                result = json.load(result_file)
        if proc.returncode != 0:
            detail = result.get("error") if isinstance(result, dict) else ""
            output_tail = _tail_subprocess_output(proc.stdout, proc.stderr)
            raise RuntimeError(
                "Cognee vector rebuild chunk subprocess failed "
                f"for {dataset} offset={offset} limit={limit} "
                f"include_edges={include_edges} exit={proc.returncode}: "
                f"{detail or output_tail or 'no child output'}"
            )
        if result.get("error"):
            raise RuntimeError(str(result["error"]))
        return {
            "rows": int(result.get("rows", 0) or 0),
            "nodes": int(result.get("nodes", 0) or 0),
            "skipped_nodes": int(result.get("skipped_nodes", 0) or 0),
            "edge_types": int(result.get("edge_types", 0) or 0),
        }
    except subprocess.TimeoutExpired as e:
        output_tail = _tail_subprocess_output(e.stdout, e.stderr)
        raise RuntimeError(
            "Cognee vector rebuild chunk subprocess timed out "
            f"for {dataset} offset={offset} limit={limit} "
            f"include_edges={include_edges} after {e.timeout}s: "
            f"{output_tail or 'no child output'}"
        ) from e
    finally:
        try:
            os.remove(result_path)
        except FileNotFoundError:
            pass


def _tail_subprocess_output(stdout: str | None, stderr: str | None, limit: int = 2000) -> str:
    output = "\n".join(part for part in [stdout or "", stderr or ""] if part)
    return output[-limit:]


async def _rebuild_embedding_vectors_chunk_in_current_process(
    dataset: str,
    *,
    offset: int,
    limit: int,
    batch_size: int,
    include_edges: bool,
    seen_path: str | None = None,
) -> dict[str, int]:
    from pydantic import create_model
    from sqlalchemy import select
    from cognee.context_global_variables import set_database_global_context_variables
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.infrastructure.databases.vector import get_vector_engine
    from cognee.infrastructure.engine import DataPoint
    from cognee.modules.data.models import Dataset
    from cognee.modules.engine.utils.generate_edge_id import generate_edge_id

    model_cache: dict[tuple[str, str], type[DataPoint]] = {}

    def make_index_model(type_name: str, field: str) -> type[DataPoint]:
        key = (type_name, field)
        cached = model_cache.get(key)
        if cached is not None:
            return cached
        model = create_model(
            type_name,
            __base__=DataPoint,
            **{
                field: (str, ""),
                "belongs_to_set": (list[str] | None, None),
                "metadata": (dict, {"index_fields": [field]}),
            },
        )
        model_cache[key] = model
        return model

    def normalize_belongs_to_set(attributes: dict[str, Any]) -> list[str]:
        raw = attributes.get("belongs_to_set")
        if not raw:
            raw = attributes.get("source_node_set")
        return _normalize_belongs_to_set(raw)

    def build_node_datapoints(row) -> list[DataPoint]:
        attributes = dict(row.attributes or {})
        type_name = str(row.type or attributes.get("type") or "").strip()
        if not type_name:
            return []
        indexed_fields = row.indexed_fields or attributes.get("metadata", {}).get("index_fields")
        if isinstance(indexed_fields, str):
            try:
                indexed_fields = json.loads(indexed_fields)
            except (TypeError, ValueError, json.JSONDecodeError):
                indexed_fields = [indexed_fields]
        if not isinstance(indexed_fields, list):
            return []

        node_id = attributes.get("id") or getattr(row, "slug", None) or getattr(row, "id", None)
        belongs_to_set = normalize_belongs_to_set(attributes)
        datapoints: list[DataPoint] = []
        for field in indexed_fields:
            field = str(field or "").strip()
            if not field:
                continue
            text = _clean_vector_text(attributes.get(field))
            if not text:
                continue
            model = make_index_model(type_name, field)
            datapoints.append(
                model(
                    id=node_id,
                    **{field: text},
                    belongs_to_set=belongs_to_set,
                    metadata={"index_fields": [field]},
                )
            )
        return datapoints

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        target = (
            await session.execute(select(Dataset).filter(Dataset.name == dataset))
        ).scalar_one_or_none()
        if target is None:
            raise RuntimeError(f"Cognee dataset not found: {dataset}")
        dataset_id = await _get_dataset_id_text(session, dataset)
        if dataset_id is None:
            raise RuntimeError(f"Cognee dataset not found: {dataset}")

        async with set_database_global_context_variables(target.id, target.owner_id):
            vector_engine = get_vector_engine()
            try:
                indexed_nodes = 0
                skipped_nodes = 0
                rows_processed = 0
                if limit > 0:
                    chunk_offset = offset
                    chunk_end = offset + limit
                    while chunk_offset < chunk_end:
                        rows = await _fetch_graph_node_rows(
                            session,
                            dataset_id,
                            offset=chunk_offset,
                            limit=min(batch_size, chunk_end - chunk_offset),
                        )
                        if not rows:
                            break
                        batch: list[DataPoint] = []
                        for row in rows:
                            datapoints = build_node_datapoints(row)
                            if datapoints:
                                batch.extend(datapoints)
                            else:
                                skipped_nodes += 1
                        if batch:
                            await _index_data_points_for_vector_rebuild(
                                batch,
                                vector_engine=vector_engine,
                                max_batch_size=batch_size,
                                seen_path=seen_path,
                            )
                            indexed_nodes += len(batch)
                        chunk_offset += len(rows)
                        rows_processed += len(rows)

                edge_type_count = 0
                if include_edges:
                    edge_rows = await _fetch_edge_relationship_counts(
                        session,
                        dataset_id,
                    )
                    edge_points = [
                        make_index_model("EdgeType", "relationship_name")(
                            id=generate_edge_id(edge_id=str(relationship)),
                            relationship_name=str(relationship),
                            metadata={"index_fields": ["relationship_name"]},
                        )
                        for relationship, count in edge_rows
                        if relationship
                    ]
                    if edge_points:
                        await _index_data_points_for_vector_rebuild(
                            edge_points,
                            vector_engine=vector_engine,
                            max_batch_size=batch_size,
                            seen_path=seen_path,
                        )
                    edge_type_count = len(edge_points)
            finally:
                vector_engine = None
                await _close_cached_vector_engine()

    return {
        "rows": rows_processed,
        "nodes": indexed_nodes,
        "skipped_nodes": skipped_nodes,
        "edge_types": edge_type_count,
    }


def _run_vector_rebuild_chunk_cli(argv: list[str]) -> int:
    if len(argv) not in (8, 9) or argv[1] != "vector-rebuild-chunk":
        return 2

    _, _, dataset, offset, limit, batch_size, include_edges, result_path = argv[:8]
    seen_path = argv[8] if len(argv) == 9 else None
    result: dict[str, Any]
    exit_code = 1
    try:
        from usr.plugins.memory_cognee.helpers.cognee_init import configure_cognee

        configure_cognee()
        result = asyncio.run(
            _rebuild_embedding_vectors_chunk_in_current_process(
                dataset,
                offset=int(offset),
                limit=int(limit),
                batch_size=int(batch_size),
                include_edges=include_edges == "1",
                seen_path=seen_path,
            )
        )
        exit_code = 0
    except BaseException as e:
        result = {"error": f"{type(e).__name__}: {e}"}
        exit_code = 1
    finally:
        _cleanup_cognee_child_processes(
            "vector-rebuild-chunk-cli",
            baseline_pids=set(),
        )

    with open(result_path, "w", encoding="utf-8") as result_file:
        json.dump(result, result_file, ensure_ascii=False)
    return exit_code


async def _close_cached_vector_engine() -> None:
    """Close the current Cognee vector engine cache entry after a bounded rebuild chunk."""
    try:
        from cognee.infrastructure.databases.vector.config import get_vectordb_context_config
        from cognee.infrastructure.databases.vector.create_vector_engine import (
            create_vector_engine,
            evict_vector_engine,
            is_vector_engine_cached,
        )

        config = get_vectordb_context_config()
        if not is_vector_engine_cached(**config):
            gc.collect()
            return

        engine = create_vector_engine(**config)
        evict_vector_engine(**config)
        close = getattr(engine, "close", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                await result
    except Exception as e:
        PrintStyle.warning(f"Could not close Cognee vector engine after rebuild chunk: {e}")
    finally:
        gc.collect()


async def _purge_vector_store_for_rebuild(dataset: str) -> None:
    try:
        from .cognee_init import purge_lancedb_vector_tables_for_dataset_names

        await purge_lancedb_vector_tables_for_dataset_names([dataset])
    except Exception as e:
        PrintStyle.error(
            f"Could not purge LanceDB vector store before rebuild for {dataset}: {e}"
        )
        raise RuntimeError(
            "Could not purge LanceDB vector store before rebuild "
            f"for {dataset}: {e}"
        ) from e


async def _run_cognify_with_corrupt_wal_repair(
    cognee: Any,
    dataset: str,
    *,
    temporal_enabled: bool,
    cognify_kwargs: dict[str, Any],
    operation_timeout: float | None,
    gate_timeout: float,
) -> None:
    await _preflight_graph_store_for_rebuild(
        cognee,
        dataset,
        gate_timeout=gate_timeout,
    )

    async def run_once() -> None:
        await run_cognee_operation(
            "cognee.cognify background",
            cognee.cognify,
            datasets=[dataset],
            temporal_cognify=temporal_enabled,
            **cognify_kwargs,
            timeout=gate_timeout,
            operation_timeout=operation_timeout,
        )

    try:
        await run_once()
    except Exception as error:
        if not _repair_corrupt_kuzu_wal(error):
            raise
        PrintStyle.warning(
            "Retrying Cognee cognify after corrupt graph WAL repair for "
            f"dataset: {dataset}"
        )
        await _reset_pipeline_status_for_rebuild(dataset)
        await run_once()


async def _preflight_graph_store_for_rebuild(
    cognee: Any,
    dataset: str,
    *,
    gate_timeout: float,
) -> None:
    child_baseline_pids = _active_child_pids()
    dataset_graphs = await run_cognee_operation(
        "cognee.graph rebuild preflight",
        read_dataset_graphs,
        cognee,
        [dataset],
        skip_empty_data=False,
        repair_unreadable=True,
        include_graph_data=False,
        timeout=gate_timeout,
    )
    errors = _graph_read_errors(dataset_graphs)
    if _contains_corrupt_wal_error(errors):
        _cleanup_cognee_child_processes(
            f"{dataset}-graph-repair-preflight",
            baseline_pids=child_baseline_pids,
        )
        dataset_graphs = await run_cognee_operation(
            "cognee.graph rebuild preflight retry",
            read_dataset_graphs,
            cognee,
            [dataset],
            skip_empty_data=False,
            repair_unreadable=True,
            include_graph_data=False,
            timeout=gate_timeout,
        )

    errors = _graph_read_errors(dataset_graphs)
    if not errors:
        return

    details = "; ".join(errors[:3])
    if _contains_corrupt_wal_error(errors):
        raise RuntimeError(
            "Cognee graph WAL repair failed before rebuild; "
            f"dataset={dataset}; {details}"
        )

    PrintStyle.warning(
        "Cognee graph preflight reported unreadable graph before rebuild; "
        f"dataset={dataset}; {details}"
    )


def _graph_read_errors(dataset_graphs: list[Any] | None) -> list[str]:
    errors = [
        f"{graph.dataset_name}: {graph.error}"
        for graph in (dataset_graphs or [])
        if getattr(graph, "error", None)
    ]
    return errors


def _contains_corrupt_wal_error(errors: list[str]) -> bool:
    details = "; ".join(errors).lower()
    return _is_repairable_graph_store_error(details)


class CogneeBackgroundWorker:
    _instance = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._dirty_datasets: Set[str] = set()
        self._dirty_versions: dict[str, int] = {}
        self._insert_count: int = 0
        self._last_cognify_time: float = time.monotonic()
        self._last_activity_time: float = self._last_cognify_time
        self._running: bool = False
        self._run_scheduled: bool = False
        self._run_scheduled_force: bool = False
        self._last_error: str | None = None
        self._last_run_datasets: list[str] = []
        self._last_run_success: bool = False
        self._dataset_readiness: dict[str, dict[str, object]] = {}
        self._retry_attempts: dict[str, int] = {}
        self._needs_pipeline_reset: Set[str] = set()
        self._task: DeferredTask | None = None
        self._state_lock = threading.RLock()

    @classmethod
    def get_instance(cls) -> "CogneeBackgroundWorker":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    def mark_dirty(
        self,
        dataset_name: str,
        *,
        reset_retry: bool = True,
        preserve_readable: bool = True,
    ) -> None:
        """Mark a dataset as having new data."""
        with self._state_lock:
            self._last_activity_time = time.monotonic()
            self._dirty_datasets.add(dataset_name)
            self._dirty_versions[dataset_name] = self._dirty_versions.get(dataset_name, 0) + 1
            if reset_retry:
                self._retry_attempts.pop(dataset_name, None)
            self._insert_count += 1
            readable = (
                self._dataset_has_readable_snapshot_locked(dataset_name)
                if preserve_readable
                else False
            )
            self._set_dataset_state_locked(
                dataset_name,
                "dirty",
                "Cognee memory graph rebuild pending",
                readable=readable,
            )
        self._schedule_run_soon(force=not preserve_readable or not readable)

    def mark_activity(self) -> None:
        """Record user-path memory activity so background rebuild waits for idle."""
        with self._state_lock:
            self._last_activity_time = time.monotonic()

    def mark_datasets_readable(
        self,
        dataset_names: list[str],
        reason: str = "Cognee memory graph is readable",
    ) -> None:
        """Record that existing graph/vector data can serve search while rebuild catches up."""
        now = time.monotonic()
        with self._state_lock:
            for dataset_name in dataset_names:
                if not dataset_name:
                    continue
                readiness = dict(self._dataset_readiness.get(dataset_name, {}))
                state = str(readiness.get("state") or "")
                if not state:
                    state = "ready"
                readiness.update(
                    {
                        "state": state,
                        "reason": readiness.get("reason") or reason,
                        "readable": True,
                        "last_ready_at": now,
                        "last_ready_reason": reason,
                        "updated_at": readiness.get("updated_at") or now,
                    }
                )
                self._dataset_readiness[dataset_name] = readiness

    def get_status(self) -> dict:
        """Return current status for dashboard."""
        self._refresh_stale_rebuilds()
        with self._state_lock:
            return {
                "running": self._running,
                "dirty_datasets": list(self._dirty_datasets),
                "insert_count": self._insert_count,
                "last_cognify_time": self._last_cognify_time,
                "last_run_datasets": self._last_run_datasets,
                "last_run_success": self._last_run_success,
                "last_error": self._last_error,
                "dataset_readiness": {
                    dataset: dict(state)
                    for dataset, state in self._dataset_readiness.items()
                },
                "retry_attempts": dict(self._retry_attempts),
                "pipeline_reset_datasets": sorted(self._needs_pipeline_reset),
            }

    def get_search_block_reason(self, datasets: list[str]) -> str | None:
        """Return why graph search must wait, or None when datasets are searchable."""
        self._refresh_stale_rebuilds()
        clean_datasets = [dataset for dataset in datasets if dataset]
        if not clean_datasets:
            return None

        pending: list[str] = []
        rebuilding: list[str] = []
        failed: list[str] = []
        with self._state_lock:
            for dataset in clean_datasets:
                readiness = self._dataset_readiness.get(dataset, {})
                state = str(readiness.get("state") or "")
                if not state or state == "ready":
                    continue
                if self._dataset_has_readable_snapshot_locked(dataset):
                    continue
                if state == "rebuilding":
                    rebuilding.append(dataset)
                elif state == "failed":
                    failed.append(dataset)
                else:
                    pending.append(dataset)

        if rebuilding:
            return f"Cognee memory graph rebuild running for dataset(s): {rebuilding}"
        if pending:
            return f"Cognee memory graph rebuild pending for dataset(s): {pending}"
        if failed:
            failure_reasons = []
            with self._state_lock:
                for dataset in failed[:3]:
                    reason = self._dataset_readiness.get(dataset, {}).get("reason")
                    if reason:
                        failure_reasons.append(f"{dataset}: {reason}")
            detail = f". Reason: {'; '.join(failure_reasons)}" if failure_reasons else ""
            return f"Cognee memory graph rebuild failed for dataset(s): {failed}{detail}"
        return None

    def _set_dataset_state_locked(
        self,
        dataset_name: str,
        state: str,
        reason: str | None = None,
        *,
        readable: bool | None = None,
    ) -> None:
        now = time.monotonic()
        previous = self._dataset_readiness.get(dataset_name, {})
        if readable is None:
            readable = bool(previous.get("readable")) or state == "ready"
        last_ready_at = previous.get("last_ready_at")
        last_ready_reason = previous.get("last_ready_reason")
        if state == "ready":
            readable = True
            last_ready_at = now
            last_ready_reason = reason or "Cognee memory graph rebuild completed"
        self._dataset_readiness[dataset_name] = {
            "state": state,
            "reason": reason,
            "updated_at": now,
            "readable": bool(readable),
            "last_ready_at": last_ready_at,
            "last_ready_reason": last_ready_reason,
        }

    def _dataset_has_readable_snapshot_locked(self, dataset_name: str) -> bool:
        readiness = self._dataset_readiness.get(dataset_name, {})
        return bool(readiness.get("readable")) or readiness.get("state") == "ready"

    def _touch_dataset_rebuild_progress(
        self,
        dataset_name: str,
        progress: dict[str, int | str] | None = None,
    ) -> None:
        with self._state_lock:
            readiness = self._dataset_readiness.get(dataset_name)
            if not readiness or readiness.get("state") != "rebuilding":
                return
            readiness["updated_at"] = time.monotonic()
            if progress is not None:
                readiness["progress"] = dict(progress)

    def _refresh_stale_rebuilds(self) -> None:
        """Expire rebuild states that outlived the worker operation timeout path."""
        config = self._get_config()
        stale_after = float(config["rebuild_stale_after"])
        if stale_after <= 0:
            return

        stale_rebuilds: list[tuple[str, str]] = []
        should_schedule_retry = False
        with self._state_lock:
            now = time.monotonic()
            for dataset, readiness in list(self._dataset_readiness.items()):
                if readiness.get("state") != "rebuilding":
                    continue

                updated_at = readiness.get("updated_at")
                if not isinstance(updated_at, (int, float)):
                    continue

                age_seconds = now - float(updated_at)
                if age_seconds < stale_after:
                    continue

                if self._running:
                    continue
                detail = "retry scheduled"
                self._run_scheduled = False
                should_schedule_retry = True

                reason = (
                    "Cognee memory graph rebuild state is stale "
                    f"after {age_seconds:.0f}s ({detail})"
                )
                self._dirty_datasets.add(dataset)
                self._needs_pipeline_reset.add(dataset)
                self._last_error = reason
                self._last_run_success = False
                self._set_dataset_state_locked(dataset, "failed", reason)
                stale_rebuilds.append((dataset, reason))

        for dataset, reason in stale_rebuilds:
            PrintStyle.warning(
                f"Cognee rebuild state expired for dataset {dataset}: {reason}"
            )

        if should_schedule_retry:
            self._schedule_run_soon(float(config["retry_min_delay"]), force=True)

    def _mark_unfinished_rebuilds_failed_locked(
        self,
        retry_delay: float,
    ) -> list[str]:
        unfinished: list[str] = []
        for dataset in self._last_run_datasets:
            readiness = self._dataset_readiness.get(dataset, {})
            if readiness.get("state") != "rebuilding":
                continue

            reason = (
                "Cognee memory graph rebuild interrupted before readiness update; "
                f"retry scheduled in {retry_delay:.0f}s"
            )
            self._dirty_datasets.add(dataset)
            self._needs_pipeline_reset.add(dataset)
            self._last_error = reason
            self._last_run_success = False
            self._set_dataset_state_locked(dataset, "failed", reason)
            unfinished.append(dataset)
        return unfinished

    def nudge_rebuild_if_unready(self, datasets: list[str], reason: str = "") -> bool:
        """Legacy empty-search hook.

        Empty search results are not enough to prove a graph is stale, so this
        intentionally does not schedule rebuilds. Startup graph checks and
        explicit dirty marks own rebuild state.
        """
        clean_datasets = [dataset for dataset in datasets if dataset]
        if not clean_datasets:
            return False

        with self._state_lock:
            if self._running:
                return False
        return False

    def _get_config(self) -> dict:
        """Load cognee-related settings."""
        return {
            "cognify_interval": get_cognee_setting("cognee_cognify_interval", 5),
            "temporal_enabled": get_cognee_setting("cognee_temporal_enabled", False),
            "memify_enabled": get_cognee_setting("cognee_memify_enabled", True),
            "retry_min_delay": get_cognee_setting("cognee_rebuild_retry_min_seconds", 30),
            "retry_max_delay": get_cognee_setting("cognee_rebuild_retry_max_seconds", 300),
            "operation_timeout": get_cognee_setting("cognee_operation_timeout_seconds", 1800),
            "rebuild_stale_after": get_cognee_setting("cognee_rebuild_stale_after_seconds", 3600),
            "rebuild_data_per_batch": get_cognee_setting("cognee_rebuild_data_per_batch", None),
            "rebuild_chunks_per_batch": get_cognee_setting("cognee_rebuild_chunks_per_batch", None),
        }

    async def _should_run(self) -> bool:
        """Run only when search is unreadable or readable data has been idle."""
        config = self._get_config()
        interval_minutes = config["cognify_interval"]

        with self._state_lock:
            dirty_count = len(self._dirty_datasets)
            last_activity_time = self._last_activity_time
            has_unreadable_dirty = any(
                not self._dataset_has_readable_snapshot_locked(dataset)
                for dataset in self._dirty_datasets
            )

        if not dirty_count:
            return False
        if has_unreadable_dirty:
            return True

        time_elapsed_minutes = (time.monotonic() - last_activity_time) / 60
        return time_elapsed_minutes >= interval_minutes

    def _next_idle_rebuild_delay(self) -> float | None:
        """Return seconds until readable dirty datasets should be checked again."""
        config = self._get_config()
        interval_seconds = float(config["cognify_interval"]) * 60
        if interval_seconds <= 0:
            return 0

        with self._state_lock:
            if not self._dirty_datasets:
                return None
            has_unreadable_dirty = any(
                not self._dataset_has_readable_snapshot_locked(dataset)
                for dataset in self._dirty_datasets
            )
            if has_unreadable_dirty:
                return 0
            elapsed_seconds = time.monotonic() - self._last_activity_time

        remaining = interval_seconds - elapsed_seconds
        if remaining <= 0:
            return 0
        return max(1.0, remaining)

    async def run_pipeline(self) -> None:
        """Run cognify + memify on dirty datasets."""
        should_reschedule = False
        reschedule_delay: float | None = None
        reschedule_force = False
        mark_embedding_config_applied = False
        with self._state_lock:
            if self._running or not self._dirty_datasets:
                return

            self._running = True
            self._last_error = None
            datasets = sorted(self._dirty_datasets)
            dataset_versions = {
                dataset: self._dirty_versions.get(dataset, 0)
                for dataset in datasets
            }
            self._last_run_datasets = datasets
            dataset_states = {
                dataset: dict(self._dataset_readiness.get(dataset, {}))
                for dataset in datasets
            }
            pipeline_reset_datasets = set(self._needs_pipeline_reset)

        config = self._get_config()
        embedding_rebuild_needed = _embedding_config_rebuild_needed()

        try:
            import cognee
        except Exception as e:
            config = self._get_config()
            retry_delay = float(config["retry_min_delay"])
            with self._state_lock:
                error = f"Cognee import failed: {e}"
                self._last_error = error
                self._last_run_success = False
                for dataset in datasets:
                    self._dirty_datasets.add(dataset)
                    self._set_dataset_state_locked(
                        dataset,
                        "failed",
                        f"{error}; retry scheduled in {retry_delay:.0f}s",
                    )
                self._running = False
                should_reschedule = bool(self._dirty_datasets)
                reschedule_force = should_reschedule
            PrintStyle.error(f"Cognee background: cognee import failed: {e}")
            if should_reschedule:
                self._schedule_run_soon(retry_delay, force=reschedule_force)
            return

        try:
            failed_datasets: list[str] = []
            completed_datasets: set[str] = set()
            retry_delays: list[float] = []
            for dataset in datasets:
                dataset_started = False
                child_baseline_pids = _active_child_pids()
                try:
                    with self._state_lock:
                        self._set_dataset_state_locked(
                            dataset,
                            "rebuilding",
                            "Cognee memory graph rebuild running",
                        )
                    dataset_started = True

                    previous_state = str(
                        dataset_states.get(dataset, {}).get("state") or ""
                    )
                    needs_pipeline_reset = (
                        previous_state == "failed"
                        or dataset in pipeline_reset_datasets
                    )
                    rebuilt_vectors_only = False
                    if (
                        embedding_rebuild_needed
                        and not needs_pipeline_reset
                        and await _dataset_has_existing_graph(dataset)
                    ):
                        await _close_cached_vector_engine()
                        await _purge_vector_store_for_rebuild(dataset)
                        counts = await run_cognee_operation(
                            "cognee.vector rebuild from existing graph",
                            _rebuild_embedding_vectors_from_existing_graph,
                            dataset,
                            progress_callback=(
                                lambda progress, dataset_name=dataset: (
                                    self._touch_dataset_rebuild_progress(
                                        dataset_name,
                                        progress,
                                    )
                                )
                            ),
                            operation_timeout=None,
                        )
                        rebuilt_vectors_only = True
                        PrintStyle.standard(
                            "Cognee vector indexes rebuilt from existing graph for "
                            f"{dataset}: nodes={counts.get('nodes', 0)}, "
                            f"skipped_nodes={counts.get('skipped_nodes', 0)}, "
                            f"edge_types={counts.get('edge_types', 0)}"
                        )
                        if int(counts.get("nodes", 0) or 0) <= 0:
                            raise RuntimeError(
                                "Cognee vector rebuild produced no searchable node "
                                f"vectors for dataset: {dataset}"
                            )
                    else:
                        if needs_pipeline_reset or embedding_rebuild_needed:
                            await _reset_pipeline_status_for_rebuild(dataset)
                        if embedding_rebuild_needed:
                            await _close_cached_vector_engine()
                            await _purge_vector_store_for_rebuild(dataset)

                        cognify_kwargs = {}
                        data_per_batch = _positive_int_or_none(config["rebuild_data_per_batch"])
                        chunks_per_batch = _positive_int_or_none(config["rebuild_chunks_per_batch"])
                        if (needs_pipeline_reset or embedding_rebuild_needed) and data_per_batch is None:
                            data_per_batch = 1
                        if (needs_pipeline_reset or embedding_rebuild_needed) and chunks_per_batch is None:
                            chunks_per_batch = 1
                        if data_per_batch is not None:
                            cognify_kwargs["data_per_batch"] = data_per_batch
                        if chunks_per_batch is not None:
                            cognify_kwargs["chunks_per_batch"] = chunks_per_batch
                        cognify_operation_timeout = (
                            None
                            if embedding_rebuild_needed
                            else float(config["operation_timeout"])
                        )
                        PrintStyle.standard(
                            "Cognee rebuild started for dataset: "
                            f"{dataset} "
                            f"(temporal_cognify={bool(config['temporal_enabled'])}, "
                            f"data_per_batch={data_per_batch or 'cognee-default'}, "
                            f"chunks_per_batch={chunks_per_batch or 'cognee-default'})"
                        )

                        gate_timeout = float(config["operation_timeout"])
                        await _run_cognify_with_corrupt_wal_repair(
                            cognee,
                            dataset,
                            temporal_enabled=bool(config["temporal_enabled"]),
                            cognify_kwargs=cognify_kwargs,
                            operation_timeout=cognify_operation_timeout,
                            gate_timeout=gate_timeout,
                        )

                        PrintStyle.standard(f"Cognee cognify completed for dataset: {dataset}")

                    readiness_error = await _verify_cognify_ready(
                        cognee,
                        [dataset],
                        gate_timeout=float(config["operation_timeout"]),
                    )
                    if readiness_error:
                        operation = "vector rebuild" if rebuilt_vectors_only else "cognify"
                        raise RuntimeError(
                            f"Cognee {operation} completed but "
                            f"{readiness_error} for dataset: {dataset}"
                        )

                    if config["memify_enabled"] and not rebuilt_vectors_only:
                        try:
                            await run_cognee_operation(
                                "cognee.improve background",
                                cognee.improve,
                                dataset=dataset,
                                timeout=float(config["operation_timeout"]),
                                operation_timeout=float(config["operation_timeout"]),
                            )
                            PrintStyle.standard(f"Cognee improve completed for dataset: {dataset}")
                        except Exception as e:
                            if _is_empty_graph_improve_error(e):
                                PrintStyle.warning(
                                    f"Cognee improve skipped for {dataset}: graph is empty"
                                )
                            else:
                                raise

                        readiness_error = await _verify_cognify_ready(
                            cognee,
                            [dataset],
                            gate_timeout=float(config["operation_timeout"]),
                        )
                        if readiness_error:
                            raise RuntimeError(
                                "Cognee improve completed but "
                                f"{readiness_error} for dataset: {dataset}"
                            )
                    with self._state_lock:
                        self._retry_attempts.pop(dataset, None)
                        if (
                            self._dirty_versions.get(dataset, 0)
                            == dataset_versions.get(dataset)
                        ):
                            self._dirty_datasets.discard(dataset)
                            self._dirty_versions.pop(dataset, None)
                            self._needs_pipeline_reset.discard(dataset)
                            self._set_dataset_state_locked(
                                dataset,
                                "ready",
                                "Cognee memory graph rebuild completed",
                            )
                        else:
                            self._set_dataset_state_locked(
                                dataset,
                                "dirty",
                                "Cognee memory graph changed during rebuild",
                            )
                        completed_datasets.add(dataset)
                except Exception as e:
                    failed_datasets.append(dataset)
                    with self._state_lock:
                        self._needs_pipeline_reset.add(dataset)
                        attempt = self._retry_attempts.get(dataset, 0) + 1
                        self._retry_attempts[dataset] = attempt
                        retry_delay = min(
                            float(config["retry_max_delay"]),
                            float(config["retry_min_delay"]) * (2 ** (attempt - 1)),
                        )
                        retry_delays.append(retry_delay)
                        self._last_error = str(e)
                        self._set_dataset_state_locked(
                            dataset,
                            "failed",
                            f"{e}; retry scheduled in {retry_delay:.0f}s",
                        )
                    PrintStyle.error(f"Cognee pipeline failed for dataset {dataset}", str(e))
                finally:
                    if dataset_started and embedding_rebuild_needed:
                        try:
                            await _close_cached_vector_engine()
                        except Exception as cleanup_error:
                            PrintStyle.warning(
                                "Cognee vector engine cleanup failed after "
                                f"{dataset}: {cleanup_error}"
                            )
                    if dataset_started:
                        try:
                            _cleanup_cognee_child_processes(
                                dataset,
                                baseline_pids=child_baseline_pids,
                            )
                        except Exception as cleanup_error:
                            PrintStyle.warning(
                                "Cognee child process cleanup failed after "
                                f"{dataset}: {cleanup_error}"
                            )

            with self._state_lock:
                for dataset in datasets:
                    if dataset in failed_datasets:
                        continue
                    if dataset in completed_datasets:
                        continue
                    if self._dirty_versions.get(dataset, 0) == dataset_versions.get(dataset):
                        self._dirty_datasets.discard(dataset)
                        self._dirty_versions.pop(dataset, None)
                        self._needs_pipeline_reset.discard(dataset)
                        self._set_dataset_state_locked(
                            dataset,
                            "ready",
                            "Cognee memory graph rebuild completed",
                        )
                    else:
                        self._set_dataset_state_locked(
                            dataset,
                            "dirty",
                            "Cognee memory graph changed during rebuild",
                        )

                if failed_datasets:
                    should_reschedule = bool(self._dirty_datasets)
                    reschedule_delay = max(retry_delays) if retry_delays else None
                    reschedule_force = should_reschedule
                    self._last_run_success = False
                elif self._dirty_datasets:
                    should_reschedule = True
                    self._last_run_success = True
                else:
                    should_reschedule = False
                    self._insert_count = 0
                    self._last_run_success = True
                    mark_embedding_config_applied = True
                self._last_cognify_time = time.monotonic()

            if mark_embedding_config_applied:
                try:
                    from .cognee_init import _mark_embedding_config_rebuild_completed

                    _mark_embedding_config_rebuild_completed()
                except Exception as e:
                    PrintStyle.warning(
                        f"Could not mark Cognee embedding config rebuild completed: {e}"
                    )
        except Exception as e:
            with self._state_lock:
                self._last_error = str(e)
                self._last_run_success = False
                for dataset in self._last_run_datasets:
                    self._needs_pipeline_reset.add(dataset)
                    self._set_dataset_state_locked(dataset, "failed", str(e))
                should_reschedule = False
                reschedule_delay = None
            PrintStyle.error("Cognee pipeline failed", str(e))
        finally:
            unfinished_datasets: list[str] = []
            with self._state_lock:
                retry_delay = float(config["retry_min_delay"])
                unfinished_datasets = self._mark_unfinished_rebuilds_failed_locked(
                    retry_delay,
                )
                if unfinished_datasets:
                    should_reschedule = True
                    reschedule_force = True
                    reschedule_delay = max(
                        retry_delay,
                        float(reschedule_delay or 0),
                    )
                self._running = False
                if self._dirty_datasets and not should_reschedule:
                    should_reschedule = True
            for dataset in unfinished_datasets:
                PrintStyle.error(
                    "Cognee pipeline did not complete readiness update",
                    f"dataset={dataset}",
                )
            if should_reschedule and not reschedule_force:
                idle_delay = self._next_idle_rebuild_delay()
                if idle_delay is not None:
                    reschedule_delay = idle_delay
                    reschedule_force = idle_delay <= 0
            self._log_rebuild_readiness(
                retry_scheduled=should_reschedule,
                retry_delay=reschedule_delay,
            )
            if should_reschedule:
                self._schedule_run_soon(reschedule_delay, force=reschedule_force)

    async def maybe_run_pipeline(self) -> None:
        """Run pipeline only when dataset readability or idle timing allows it."""
        if await self._should_run():
            await self.run_pipeline()

    async def run_loop(self) -> None:
        """Main background loop. Checks every 60 seconds if pipeline should run."""
        PrintStyle.standard("Cognee background worker started")
        while True:
            try:
                await self.maybe_run_pipeline()
            except Exception as e:
                with self._state_lock:
                    self._last_error = str(e)
                PrintStyle.error("Cognee background worker error", str(e))
            await asyncio.sleep(60)

    def start(self) -> DeferredTask:
        """Start the background worker using DeferredTask. Returns the task for optional cleanup."""
        with self._state_lock:
            if self._task and self._task.is_alive():
                return self._task

            task = DeferredTask(thread_name=THREAD_BACKGROUND)
            task.start_task(self.run_loop)
            self._task = task
            return task

    def _schedule_run_soon(
        self,
        delay: float | None = None,
        *,
        force: bool = False,
    ) -> None:
        """Debounce a near-immediate rebuild on the background worker loop."""
        with self._state_lock:
            if self._run_scheduled:
                self._run_scheduled_force = self._run_scheduled_force or force
                return
            if self._running:
                return
            self._run_scheduled = True
            self._run_scheduled_force = force
            if delay is None:
                delay = float(get_cognee_setting("cognee_cognify_debounce_seconds", 2))
            task = self._task

        loop = getattr(getattr(task, "event_loop_thread", None), "loop", None)
        if loop and loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self._run_after_delay(delay, force=force),
                loop,
            )
            return

        with self._state_lock:
            self._run_scheduled = False
            self._run_scheduled_force = False

    async def _run_after_delay(self, delay: float, *, force: bool = False) -> None:
        reschedule_delay: float | None = None
        try:
            if delay > 0:
                await asyncio.sleep(delay)
            with self._state_lock:
                force = force or self._run_scheduled_force
                self._run_scheduled = False
                self._run_scheduled_force = False
            if force:
                await self.run_pipeline()
            elif await self._should_run():
                await self.run_pipeline()
            else:
                reschedule_delay = self._next_idle_rebuild_delay()
        finally:
            with self._state_lock:
                self._run_scheduled = False
                self._run_scheduled_force = False
        if reschedule_delay is not None:
            self._schedule_run_soon(reschedule_delay)

    def _log_rebuild_readiness(
        self,
        *,
        retry_scheduled: bool,
        retry_delay: float | None,
    ) -> None:
        with self._state_lock:
            datasets = list(self._last_run_datasets)
            dirty = sorted(self._dirty_datasets)
            running = self._running
            last_error = self._last_error
            blocked_states = []
            ready = []
            for dataset_name, state in self._dataset_readiness.items():
                state_name = str(state.get("state") or "")
                if state_name == "ready":
                    ready.append(dataset_name)
                elif state_name:
                    blocked_states.append(f"{dataset_name}:{state_name}")

        if not datasets:
            return

        if running or dirty or blocked_states:
            PrintStyle.warning(
                "Cognee rebuild readiness: BLOCKED; recall may be unavailable. "
                f"last_run_datasets={datasets}; ready={_short_list(sorted(ready))}; "
                f"dirty={dirty}; blocked_states={_short_list(sorted(blocked_states))}; "
                f"retry_scheduled={retry_scheduled}; retry_delay={retry_delay}; "
                f"last_error={last_error}"
            )
            return

        PrintStyle.standard(
            "Cognee rebuild readiness: READY; recall enabled. "
            f"last_run_datasets={datasets}; ready={_short_list(sorted(ready))}"
        )


def _is_empty_graph_improve_error(error: Exception) -> bool:
    message = str(error).lower()
    return "entitynotfounderror" in message and "empty graph projected" in message


async def _verify_cognify_ready(
    cognee,
    datasets: list[str],
    *,
    gate_timeout: float | None = None,
) -> str:
    """Return an error reason if cognify did not produce a readable graph."""
    operation_kwargs = {}
    if gate_timeout is not None:
        operation_kwargs["timeout"] = gate_timeout
    dataset_graphs = await run_cognee_operation(
        "cognee.graph readiness",
        read_dataset_graphs,
        cognee,
        datasets,
        skip_empty_data=False,
        repair_unreadable=True,
        include_graph_data=False,
        **operation_kwargs,
    )
    if not dataset_graphs:
        return "graph readiness could not be verified"

    errors = [graph for graph in dataset_graphs if graph.error]
    if errors:
        details = "; ".join(
            f"{graph.dataset_name}: {graph.error}" for graph in errors[:3]
        )
        PrintStyle.warning(f"Could not read Cognee dataset graph(s): {details}")
        return "graph data could not be read"

    relevant_graphs = [
        graph
        for graph in dataset_graphs
        if graph.data_count is None or graph.data_count > 0
    ]
    if not relevant_graphs:
        return ""

    if any(graph.graph_empty is False for graph in relevant_graphs):
        return ""

    PrintStyle.warning(
        f"Cognee graph is still empty after cognify. "
        f"{_describe_graph_dataset_results(relevant_graphs)}"
    )
    return "graph is still empty"


async def _reset_pipeline_status_for_rebuild(dataset: str) -> None:
    """Clear Cognee pipeline status before retrying a failed rebuild."""
    try:
        from .cognee_init import reset_cognify_status_for_dataset_names

        reset = await reset_cognify_status_for_dataset_names([dataset])
        if reset:
            PrintStyle.standard(
                f"Reset Cognee pipeline status before rebuild retry: {reset}"
            )
    except Exception as e:
        PrintStyle.warning(
            f"Could not reset Cognee pipeline status before rebuild retry for {dataset}: {e}"
        )


def _describe_graph_dataset_results(dataset_graphs: list) -> str:
    non_empty_datasets = [
        graph.dataset_name
        for graph in dataset_graphs
        if graph.data_count is not None and graph.data_count > 0
    ]
    unknown_datasets = [
        graph.dataset_name
        for graph in dataset_graphs
        if graph.data_count is None
    ]

    details = []
    if non_empty_datasets:
        details.append(f"Non-empty dataset(s): {non_empty_datasets}")
    if unknown_datasets:
        details.append(f"unverified dataset(s): {unknown_datasets}")

    return "; ".join(details)


def _short_list(items: list[str], limit: int = 12) -> list[str]:
    if len(items) <= limit:
        return items
    return [*items[:limit], f"... +{len(items) - limit} more"]


async def _describe_non_empty_or_unverified_datasets(cognee, datasets: list[str]) -> str:
    non_empty_datasets = []
    unknown_datasets = []
    for dataset in datasets:
        data_count = await _get_dataset_data_count(cognee, dataset)
        if data_count is None:
            unknown_datasets.append(dataset)
        elif data_count > 0:
            non_empty_datasets.append(dataset)

    details = []
    if non_empty_datasets:
        details.append(f"Non-empty dataset(s): {non_empty_datasets}")
    if unknown_datasets:
        details.append(f"unverified dataset(s): {unknown_datasets}")

    return "; ".join(details)


async def _get_dataset_data_count(cognee, dataset_name: str) -> int | None:
    datasets_api = getattr(cognee, "datasets", None)
    if datasets_api is None:
        return None

    try:
        datasets = await datasets_api.list_datasets()
    except Exception as e:
        PrintStyle.warning(f"Could not list Cognee datasets for readiness check: {e}")
        return None

    target = None
    for dataset in datasets:
        if getattr(dataset, "name", None) == dataset_name:
            target = dataset
            break

    if target is None:
        return 0

    try:
        data_items = await datasets_api.list_data(target.id)
        return len(data_items or [])
    except Exception as e:
        PrintStyle.warning(
            f"Could not list Cognee data for readiness check ({dataset_name}): {e}"
        )
        return None


if __name__ == "__main__":
    raise SystemExit(_run_vector_rebuild_chunk_cli(sys.argv))
