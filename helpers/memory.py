from datetime import datetime
from typing import Any, List, Optional

import os
import json
import asyncio
import hashlib


from helpers.print_style import PrintStyle
from helpers import files
from langchain_core.documents import Document
from . import knowledge_import
from helpers.log import Log, LogItem
from enum import Enum
from agent import Agent, AgentContext
import models
import logging
from .cognee_init import get_cognee_setting


def _get_cognee():
    from .cognee_init import get_cognee
    return get_cognee()


def parse_node_set_area(raw_node_set) -> str:
    """Extract area string from Cognee Data.node_set field.

    Cognee double-serializes: json.dumps() into a Column(JSON), so the value
    comes back as a JSON string like '["main"]' rather than a list.
    """
    if not raw_node_set:
        return "main"
    ns = raw_node_set
    if isinstance(ns, str):
        try:
            ns = json.loads(ns)
        except (json.JSONDecodeError, ValueError):
            return ns.strip().lower()
    if isinstance(ns, list) and ns:
        return str(ns[0]).strip().lower()
    return str(ns).strip().lower()


def content_hash_id(content: str, dataset_name: str = "") -> str:
    """Deterministic ID derived from content text. Used for matching data items and feedback."""
    h = hashlib.sha256()
    h.update(str(dataset_name).encode("utf-8", errors="replace"))
    h.update(b"\0")
    h.update(content[:8000].encode("utf-8", errors="replace"))
    return "syn_" + h.hexdigest()[:32]


class Memory:

    class Area(Enum):
        MAIN = "main"
        FRAGMENTS = "fragments"
        SOLUTIONS = "solutions"

    _initialized_subdirs: set[str] = set()
    _datasets_cache: dict[str, str] = {}
    _existing_datasets_cache: set[str] | None = None
    _existing_datasets_ts: float = 0
    _DATASETS_CACHE_TTL = 30
    SEARCH_TIMEOUT = 15

    @staticmethod
    async def get(agent: Agent) -> "Memory":
        memory_subdir = get_agent_memory_subdir(agent)
        dataset_name = _subdir_to_dataset(memory_subdir)
        mem = Memory(dataset_name=dataset_name, memory_subdir=memory_subdir)
        if memory_subdir not in Memory._initialized_subdirs:
            Memory._initialized_subdirs.add(memory_subdir)
            knowledge_subdirs = get_knowledge_subdirs_by_memory_subdir(
                memory_subdir, agent.config.knowledge_subdirs or []
            )
            if knowledge_subdirs:
                log_item = agent.context.log.log(
                    type="util",
                    heading=f"Initializing Cognee memory in '{memory_subdir}'",
                )
                await mem.preload_knowledge(log_item, knowledge_subdirs, memory_subdir)
        return mem

    @staticmethod
    async def get_by_subdir(
        memory_subdir: str,
        log_item: LogItem | None = None,
        preload_knowledge: bool = True,
    ) -> "Memory":
        dataset_name = _subdir_to_dataset(memory_subdir)
        mem = Memory(dataset_name=dataset_name, memory_subdir=memory_subdir)
        if preload_knowledge and memory_subdir not in Memory._initialized_subdirs:
            Memory._initialized_subdirs.add(memory_subdir)
            import initialize
            agent_config = initialize.initialize_agent()
            knowledge_subdirs = get_knowledge_subdirs_by_memory_subdir(
                memory_subdir, agent_config.knowledge_subdirs or []
            )
            if knowledge_subdirs:
                await mem.preload_knowledge(log_item, knowledge_subdirs, memory_subdir)
        return mem

    @staticmethod
    async def reload(agent: Agent) -> "Memory":
        Memory._initialized_subdirs.clear()
        Memory._datasets_cache.clear()
        return await Memory.get(agent)

    def __init__(self, dataset_name: str, memory_subdir: str):
        self.dataset_name = dataset_name
        self.memory_subdir = memory_subdir

    def get_search_datasets(self) -> list[str]:
        """Always search in 'default' + current project dataset (if any)."""
        ds = ["default"]
        if self.dataset_name != "default" and self.dataset_name not in ds:
            ds.append(self.dataset_name)
        return ds

    @staticmethod
    async def _get_existing_dataset_names() -> set[str]:
        import time as _t
        now = _t.monotonic()
        if (Memory._existing_datasets_cache is not None
                and now - Memory._existing_datasets_ts < Memory._DATASETS_CACHE_TTL):
            return Memory._existing_datasets_cache
        try:
            cognee, _ = _get_cognee()
            all_ds = await cognee.datasets.list_datasets()
            Memory._existing_datasets_cache = {ds.name for ds in all_ds}
            Memory._existing_datasets_ts = now
        except Exception:
            if Memory._existing_datasets_cache is not None:
                return Memory._existing_datasets_cache
            return set()
        return Memory._existing_datasets_cache

    @staticmethod
    def _invalidate_datasets_cache():
        Memory._existing_datasets_cache = None

    async def preload_knowledge(
        self, log_item: LogItem | None, kn_dirs: list[str], memory_subdir: str
    ):
        cognee, _ = _get_cognee()

        if log_item:
            log_item.update(heading="Preloading knowledge...")

        state_dir = _state_dir(memory_subdir)
        os.makedirs(state_dir, exist_ok=True)
        index_path = os.path.join(state_dir, "knowledge_import.json")

        index: dict[str, knowledge_import.KnowledgeImport] = {}
        if os.path.exists(index_path):
            with open(index_path, "r") as f:
                index = json.load(f)

        if index:
            try:
                datasets = await cognee.datasets.list_datasets()
                if not datasets:
                    PrintStyle.warning("Cognee DB is empty but index exists — forcing full re-import")
                    if log_item:
                        log_item.stream(progress="\nCognee DB empty, re-importing all knowledge...")
                    index = {}
            except Exception:
                PrintStyle.warning("Cannot check cognee datasets — forcing full re-import")
                index = {}

        index = self._preload_knowledge_folders(log_item, kn_dirs, index)

        all_ids_to_delete: set[str] = set()
        for entry in index.values():
            if entry["state"] in ["changed", "removed"] and entry.get("ids", []):
                all_ids_to_delete.update(entry["ids"])

        if all_ids_to_delete:
            await _batch_delete_by_ids(self.dataset_name, all_ids_to_delete)

        for file_key in index:
            entry = index[file_key]
            if entry["state"] == "changed" and entry.get("documents"):
                new_ids = []
                area = entry.get("metadata", {}).get("area", "main")
                for doc in entry["documents"]:
                    content = doc.page_content if hasattr(doc, "page_content") else str(doc)
                    try:
                        await cognee.add(
                            content,
                            dataset_name=self.dataset_name,
                            node_set=[area],
                        )
                        new_ids.append(content_hash_id(content, self.dataset_name))
                    except Exception as e:
                        PrintStyle.error(f"Failed to import knowledge: {e}")
                entry["ids"] = new_ids

        index = {k: v for k, v in index.items() if v["state"] != "removed"}

        for file_key in index:
            if "documents" in index[file_key]:
                del index[file_key]["documents"]
            if "state" in index[file_key]:
                del index[file_key]["state"]
        with open(index_path, "w") as f:
            json.dump(index, f)

    def _preload_knowledge_folders(
        self,
        log_item: LogItem | None,
        kn_dirs: list[str],
        index: dict[str, knowledge_import.KnowledgeImport],
    ):
        for kn_dir in kn_dirs:
            index = knowledge_import.load_knowledge(
                log_item,
                abs_knowledge_dir(kn_dir),
                index,
                {"area": Memory.Area.MAIN.value},
                filename_pattern="*",
                recursive=False,
            )
            for area in Memory.Area:
                index = knowledge_import.load_knowledge(
                    log_item,
                    abs_knowledge_dir(kn_dir, area.value),
                    index,
                    {"area": area.value},
                    recursive=True,
                )
        return index

    def get_document_by_id(self, id: str) -> Document | None:
        return None

    async def search_similarity_threshold(
        self, query: str, limit: int, threshold: float, filter: str = "",
        include_default: bool = True, session_id: str | None = None,
    ) -> list[Document]:
        cognee, SearchType = _get_cognee()
        from cognee.modules.engine.models.node_set import NodeSet

        node_names = _parse_filter_to_node_names(filter)
        datasets = self.get_search_datasets() if include_default else [self.dataset_name]

        try:
            results = await cognee.search(
                query_text=query,
                top_k=limit,
                datasets=datasets,
                node_type=NodeSet,
                node_name=node_names if node_names else None,
                session_id=session_id,
                only_context=True,
                verbose=True,
            )
        except Exception as e:
            PrintStyle.error(f"cognee.search failed: {e}")
            return []

        return _results_to_documents(results or [], limit)

    async def delete_documents_by_query(
        self, query: str, threshold: float, filter: str = ""
    ) -> list[Document]:
        docs = await self.search_similarity_threshold(
            query=query, limit=100, threshold=threshold, filter=filter,
            include_default=False,
        )
        if docs:
            await _delete_matching_data_items(self.dataset_name, docs)
            _invalidate_dashboard_cache()
        return docs

    async def delete_documents_by_ids(self, ids: list[str]) -> list[Document]:
        if not ids:
            return []

        cognee, _ = _get_cognee()
        removed = []
        id_set = set(ids)

        try:
            target = await _find_dataset(self.dataset_name)
            if not target:
                return []

            for data_id in list(id_set):
                if await _try_delete_direct(cognee, target, data_id):
                    removed.append(Document(page_content="", metadata={"id": data_id}))
                    id_set.discard(data_id)

            if id_set:
                data_items = await cognee.datasets.list_data(target.id)
                for item in data_items:
                    if not id_set:
                        break
                    content = await read_data_item_content_async(item)
                    item_hash = content_hash_id(content, self.dataset_name)
                    for data_id in list(id_set):
                        if item_hash == data_id:
                            await cognee.forget(data_id=item.id, dataset=target.id)
                            removed.append(Document(page_content="", metadata={"id": data_id}))
                            id_set.discard(data_id)
                            break
        except Exception as e:
            PrintStyle.error(f"Failed to delete from {self.dataset_name}: {e}")

        if removed:
            _invalidate_dashboard_cache()
        return removed

    async def insert_text(self, text: str, metadata: dict = {}) -> str:
        doc = Document(text, metadata=metadata)
        ids = await self.insert_documents([doc])
        if not ids:
            preview = (text or "")[:120].replace("\n", " ")
            area = metadata.get("area", Memory.Area.MAIN.value) if metadata else Memory.Area.MAIN.value
            raise RuntimeError(
                "Memory.insert_text: cognee.add returned no IDs "
                f"(dataset={self.dataset_name!r}, area={area!r}, "
                f"text_len={len(text)}, metadata_keys={list(metadata.keys()) if metadata else []}, "
                f"preview={preview!r}). "
                "See prior 'Cognee insert failed' log lines for the underlying exception."
            )
        return ids[0]

    async def insert_documents(self, docs: list[Document]) -> list[str]:
        cognee, _ = _get_cognee()
        ids = []
        from .cognee_background import CogneeBackgroundWorker

        for doc in docs:
            area = doc.metadata.get("area", Memory.Area.MAIN.value)
            if not area:
                area = Memory.Area.MAIN.value

            try:
                await cognee.add(
                    doc.page_content,
                    dataset_name=self.dataset_name,
                    node_set=[area],
                )
                content_id = content_hash_id(doc.page_content, self.dataset_name)
                ids.append(content_id)
                CogneeBackgroundWorker.get_instance().mark_dirty(self.dataset_name)
            except Exception as e:
                preview = (doc.page_content or "")[:120].replace("\n", " ")
                PrintStyle.error(
                    f"Cognee insert failed: {type(e).__name__}: {e} "
                    f"(dataset={self.dataset_name!r}, area={area!r}, "
                    f"text_len={len(doc.page_content)}, preview={preview!r})"
                )

        _invalidate_dashboard_cache()
        return ids

    async def update_documents(self, docs: list[Document]) -> list:
        ids = [doc.metadata.get("id", "") for doc in docs if doc.metadata.get("id")]
        if ids:
            await self.delete_documents_by_ids(ids)
        result = await self.insert_documents(docs)
        return result

    @staticmethod
    def format_docs_plain(docs: list[Document]) -> list[str]:
        result = []
        for doc in docs:
            text = ""
            for k, v in doc.metadata.items():
                text += f"{k}: {v}\n"
            text += f"Content: {doc.page_content}"
            result.append(text)
        return result

    @staticmethod
    def get_timestamp():
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _subdir_to_dataset(memory_subdir: str) -> str:
    return memory_subdir.replace("/", "_").replace(" ", "_").lower()


def _state_dir(memory_subdir: str) -> str:
    if memory_subdir.startswith("projects/"):
        from helpers.projects import get_project_meta
        return files.get_abs_path(get_project_meta(memory_subdir[9:]), "cognee_state")
    return files.get_abs_path("usr/cognee_state", memory_subdir)


def _parse_filter_to_node_names(filter_str: str) -> list[str]:
    if not filter_str:
        return []
    node_names = []
    for area in Memory.Area:
        if area.value in filter_str:
            node_names.append(area.value)
    return node_names


def recall_text_and_feedback_items(
    answers: Any,
    limit: int,
    *,
    context_id: str,
    fallback_dataset: str,
    kind: str,
) -> tuple[list[str], list[dict[str, Any]]]:
    """
    Plain recall lines for prompts plus rows the UI can POST to /memory_feedback.
    Each row: text, memory_id, dataset, context_id, kind ('memory' | 'solution').
    """
    docs = _results_to_documents(answers or [], limit)
    texts: list[str] = []
    items: list[dict[str, Any]] = []
    for doc in docs:
        content = (doc.page_content or "").strip()
        if not content:
            continue
        ds = str(doc.metadata.get("dataset") or fallback_dataset or "default")
        mid = str(doc.metadata.get("id") or content_hash_id(content, ds))
        texts.append(content)
        items.append(
            {
                "text": content,
                "memory_id": mid,
                "dataset": ds,
                "context_id": str(context_id or ""),
                "kind": kind,
            }
        )
    return texts, items


def _results_to_documents(results: Any, limit: int) -> list[Document]:
    docs = []
    if not results:
        return docs

    flat = _flatten_search_results(results)

    for item, dataset_name in flat:
        if len(docs) >= limit:
            break

        content = ""
        metadata: dict[str, Any] = {}

        if isinstance(item, str):
            content = item
        elif isinstance(item, dict):
            content = item.get("text", item.get("content", ""))
            if item.get("id"):
                metadata["id"] = str(item["id"])
        elif hasattr(item, "text"):
            content = str(item.text)
        elif hasattr(item, "page_content"):
            content = item.page_content
            metadata = getattr(item, "metadata", {})
        else:
            content = str(item)

        if dataset_name:
            metadata.setdefault("dataset", dataset_name)

        if not content or not content.strip():
            continue

        if not metadata.get("id"):
            ds = str(metadata.get("dataset") or "")
            metadata["id"] = content_hash_id(content, ds)

        docs.append(Document(page_content=content, metadata=metadata))

    return docs


def _flatten_search_results(results: Any) -> list[tuple[Any, str]]:
    """Flatten verbose Cognee results into (node_text, dataset_name) pairs."""
    flat: list[tuple[Any, str]] = []
    if not results:
        return flat

    for result in results:
        ds = ""
        objects = None

        if isinstance(result, dict):
            ds = result.get("dataset_name", "") or ""
            objects = result.get("objects_result")
        elif hasattr(result, "dataset_name"):
            ds = str(getattr(result, "dataset_name", "") or "")
            objects = (getattr(result, "objects_result", None)
                       or getattr(result, "result_object", None))

        if objects and isinstance(objects, list):
            _extract_nodes_to_flat(objects, str(ds), flat)
            continue

        sr = None
        if isinstance(result, dict):
            sr = result.get("search_result") or result.get("context_result")
            if sr is None:
                sr = result.get("text")
        elif hasattr(result, "search_result"):
            sr = result.search_result

        if sr is None:
            sr = result

        if isinstance(sr, str) and sr.strip():
            flat.append((sr.strip(), str(ds)))
        elif isinstance(sr, list):
            joined = "\n".join(str(item).strip() for item in sr if item)
            if joined.strip():
                flat.append((joined.strip(), str(ds)))

    return flat


def _extract_nodes_to_flat(
    objects: list, dataset_name: str, flat: list[tuple[Any, str]]
) -> None:
    """Extract unique node texts from a list of Cognee Edge objects.

    Passes Cognee native node IDs through as dicts so _results_to_documents
    can use them instead of generating synthetic IDs.
    """
    seen_ids: set = set()
    for obj in objects:
        nodes = []
        if hasattr(obj, "node1") and hasattr(obj, "node2"):
            nodes = [obj.node1, obj.node2]
        elif hasattr(obj, "attributes") and hasattr(obj, "id"):
            nodes = [obj]

        for node in nodes:
            node_id = getattr(node, "id", None)
            if node_id and node_id in seen_ids:
                continue
            if node_id:
                seen_ids.add(node_id)

            attrs = getattr(node, "attributes", {}) or {}
            text = attrs.get("text", "")
            if not text:
                text = attrs.get("description", attrs.get("name", ""))
            if text and text.strip():
                entry: dict[str, Any] = {"text": text.strip()}
                if node_id:
                    entry["id"] = str(node_id)
                flat.append((entry, dataset_name))


def _extract_dataset_name(result: Any) -> str:
    """Pull dataset_name from a Cognee result wrapper (object or dict)."""
    if hasattr(result, "dataset_name") and result.dataset_name:
        return str(result.dataset_name)
    if isinstance(result, dict):
        dn = result.get("dataset_name")
        if dn:
            return str(dn)
    return ""


def _deduplicate_documents(docs: list[Document]) -> list[Document]:
    seen: set[str] = set()
    unique: list[Document] = []
    for doc in docs:
        key = doc.metadata.get("id", "")
        if not key:
            key = doc.page_content[:200]
        if key not in seen:
            seen.add(key)
            unique.append(doc)
    return unique


def read_data_item_content(item) -> str:
    """Read the text content of a Cognee data item, checking the file at raw_data_location."""
    raw_location = getattr(item, "raw_data_location", None)
    if raw_location:
        from urllib.parse import urlparse, unquote
        path = raw_location
        if path.startswith("file://"):
            path = unquote(urlparse(path).path)
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception:
                pass
        return str(raw_location)
    return str(getattr(item, "name", ""))


async def read_data_item_content_async(item) -> str:
    """Async wrapper around read_data_item_content to avoid blocking the event loop."""
    import asyncio
    return await asyncio.to_thread(read_data_item_content, item)


async def _find_dataset(dataset_name: str):
    """Find a Cognee dataset object by name."""
    cognee, _ = _get_cognee()
    try:
        datasets = await cognee.datasets.list_datasets()
        for ds in datasets:
            if ds.name == dataset_name:
                return ds
    except Exception:
        pass
    return None


async def _try_delete_direct(cognee, dataset, data_id: str) -> bool:
    """Try deleting a data item using data_id as a Cognee native UUID."""
    try:
        import uuid
        uuid.UUID(data_id)
        await cognee.forget(data_id=data_id, dataset=dataset.id)
        return True
    except (ValueError, TypeError):
        return False
    except Exception:
        return False


async def _delete_matching_data_items(dataset_name: str, docs: list[Document]) -> int:
    """Delete Cognee data items whose content matches any of the given documents."""
    cognee, _ = _get_cognee()
    deleted = 0
    try:
        target = await _find_dataset(dataset_name)
        if not target:
            return 0

        match_hashes = set()
        for doc in docs:
            content = (doc.page_content or "").strip()
            if content:
                match_hashes.add(content_hash_id(content, dataset_name))

        if not match_hashes:
            return 0

        data_items = await cognee.datasets.list_data(target.id)
        for item in data_items:
            content = await read_data_item_content_async(item)
            item_hash = content_hash_id(content, dataset_name)
            if item_hash in match_hashes:
                try:
                    await cognee.forget(data_id=item.id, dataset=target.id)
                    deleted += 1
                except Exception:
                    pass
    except Exception as e:
        PrintStyle.error(f"Failed to delete matching data from {dataset_name}: {e}")
    return deleted


async def _delete_data_by_id(dataset_name: str, data_id: str):
    """Delete a data item by ID. Tries Cognee native UUID first, falls back to content hash."""
    cognee, _ = _get_cognee()
    try:
        target = await _find_dataset(dataset_name)
        if not target:
            return False

        if await _try_delete_direct(cognee, target, data_id):
            return True

        data_items = await cognee.datasets.list_data(target.id)
        for item in data_items:
            content = await read_data_item_content_async(item)
            item_hash = content_hash_id(content, dataset_name)
            if item_hash == data_id:
                await cognee.forget(data_id=item.id, dataset=target.id)
                return True
    except Exception as e:
        PrintStyle.error(f"Failed to delete data {data_id} from {dataset_name}: {e}")
    return False


async def _batch_delete_by_ids(dataset_name: str, ids: set[str]) -> int:
    """Delete multiple data items in one pass. Single dataset lookup + single list_data call."""
    if not ids:
        return 0
    cognee, _ = _get_cognee()
    deleted = 0
    remaining = set(ids)
    try:
        target = await _find_dataset(dataset_name)
        if not target:
            return 0

        for data_id in list(remaining):
            if await _try_delete_direct(cognee, target, data_id):
                deleted += 1
                remaining.discard(data_id)

        if remaining:
            data_items = await cognee.datasets.list_data(target.id)
            for item in data_items:
                if not remaining:
                    break
                content = await read_data_item_content_async(item)
                item_hash = content_hash_id(content, dataset_name)
                if item_hash in remaining:
                    try:
                        await cognee.forget(data_id=item.id, dataset=target.id)
                        deleted += 1
                        remaining.discard(item_hash)
                    except Exception:
                        pass
    except Exception as e:
        PrintStyle.error(f"Batch delete failed for {dataset_name}: {e}")
    return deleted


def _invalidate_dashboard_cache():
    try:
        from usr.plugins.memory_cognee.api.memory_dashboard import invalidate_dashboard_cache
        invalidate_dashboard_cache()
    except Exception:
        pass


def get_custom_knowledge_subdir_abs(agent: Agent) -> str:
    for dir in agent.config.knowledge_subdirs:
        if dir != "default":
            if dir == "custom":
                return files.get_abs_path("usr/knowledge")
            return files.get_abs_path("usr/knowledge", dir)
    raise Exception("No custom knowledge subdir set")


def reload():
    from . import cognee_init as ci
    ci._configured = False
    ci._cognee_module = None
    ci._search_type_class = None
    Memory._initialized_subdirs.clear()
    Memory._datasets_cache.clear()
    Memory._invalidate_datasets_cache()
    ci.configure_cognee()


def abs_db_dir(memory_subdir: str) -> str:
    return _state_dir(memory_subdir)


def abs_knowledge_dir(knowledge_subdir: str, *sub_dirs: str) -> str:
    if knowledge_subdir.startswith("projects/"):
        from helpers.projects import get_project_meta
        return files.get_abs_path(
            get_project_meta(knowledge_subdir[9:]), "knowledge", *sub_dirs
        )
    if knowledge_subdir == "default":
        return files.get_abs_path("knowledge", *sub_dirs)
    if knowledge_subdir == "custom":
        return files.get_abs_path("usr/knowledge", *sub_dirs)
    return files.get_abs_path("usr/knowledge", knowledge_subdir, *sub_dirs)


def get_memory_subdir_abs(agent: Agent) -> str:
    subdir = get_agent_memory_subdir(agent)
    return _state_dir(subdir)


def get_agent_memory_subdir(agent: Agent) -> str:
    return get_context_memory_subdir(agent.context)


def get_context_memory_subdir(context: AgentContext) -> str:
    from helpers.projects import get_context_project_name, load_project_header
    project_name = get_context_project_name(context)
    if project_name:
        try:
            raw_header = load_project_header(project_name)
            if raw_header.get("memory") == "own":
                return "projects/" + project_name
        except Exception:
            pass
    from helpers import plugins
    cfg = plugins.get_plugin_config("memory_cognee", agent=context.streaming_agent or context.agent0) or {}
    return cfg.get("agent_memory_subdir", "default")


def get_existing_memory_subdirs() -> list[str]:
    try:
        subdirs: set[str] = set()

        from helpers.projects import get_projects_parent_folder
        project_parent = get_projects_parent_folder()
        if os.path.exists(project_parent):
            for name in files.get_subdirectories(project_parent):
                subdirs.add(f"projects/{name}")

        result = sorted(subdirs)
        result.insert(0, "default")
        return result
    except Exception as e:
        PrintStyle.error(f"Failed to get memory subdirectories: {str(e)}")
        return ["default"]


def get_knowledge_subdirs_by_memory_subdir(
    memory_subdir: str, default: list[str]
) -> list[str]:
    result = list(default)
    if memory_subdir.startswith("projects/"):
        from helpers.projects import get_project_meta
        result.append(get_project_meta(memory_subdir[9:], "knowledge"))
    return result


async def insert_with_simple_dedup(
    db: "Memory", text: str, area: str, threshold: float
) -> str | None:
    """Replace near-identical memories above *threshold* with the new version.

    Matches core Agent Zero behavior: delete similar old entries, then insert
    the new one. This ensures memories evolve over time instead of being frozen.
    """
    try:
        if threshold > 0:
            removed = await db.delete_documents_by_query(
                query=text,
                threshold=threshold,
                filter=f"area == '{area}'",
            )
            if removed:
                PrintStyle(font_color="gray").print(
                    f"Replacing {len(removed)} similar memories (area={area}): {text[:80]}..."
                )
    except Exception:
        pass
    return await db.insert_text(text=text, metadata={"area": area})
