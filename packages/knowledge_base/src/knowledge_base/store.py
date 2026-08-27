from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
import threading
import time
from typing import Any
from uuid import uuid4

from gpt4all import Embed4All
import orjson as json
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)
from structlog import get_logger

from .documents import Document
from .loader import load_vault_markdown_dataset_after_modified
from .paths import RAG_PATH

logger = get_logger(__name__)

DEFAULT_COLLECTION_NAME = "doc"

_KB_LOCK = threading.Lock()
_KB_INSTANCES: dict[tuple[str, str], QdrantKnowledgeBase] = {}


def _sanitize_payload(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


class QdrantKnowledgeBase:
    """Local Qdrant store.

    The embedded Qdrant client takes an exclusive lock on its storage directory,
    so opening one per call breaks as soon as two requests overlap. The client is
    therefore created once and every operation runs under ``_client_lock``.
    """

    def __init__(self, path: Path, collection_name: str = DEFAULT_COLLECTION_NAME):
        self.path = path
        self.collection_name = collection_name
        self._embeddings: Embed4All | None = None
        self._embedding_lock = threading.Lock()
        self._client: QdrantClient | None = None
        self._client_lock = threading.RLock()
        self.path.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _session(self) -> Iterator[QdrantClient]:
        """Yield the shared client with exclusive access for the caller."""
        with self._client_lock:
            if self._client is None:
                self._client = QdrantClient(path=str(self.path))
            yield self._client

    def _get_embeddings(self) -> Embed4All:
        if self._embeddings is None:
            self._embeddings = Embed4All()
        return self._embeddings

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        with self._embedding_lock:
            return self._get_embeddings().embed(texts)

    def create(self) -> None:
        vector_size = len(self._embed_texts([" "])[0])
        with self._session() as client:
            if client.collection_exists(collection_name=self.collection_name):
                client.delete_collection(collection_name=self.collection_name)

            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    def rebuild(self, documents: list[Document]) -> None:
        if not documents:
            return

        texts = [doc.page_content for doc in documents]
        vectors = self._embed_texts(texts)
        vector_size = len(vectors[0])

        points = []
        for vector, document in zip(vectors, documents, strict=True):
            points.append(
                PointStruct(
                    id=str(uuid4()),
                    vector=vector,
                    payload=_sanitize_payload(
                        {
                            "page_content": document.page_content,
                            "metadata": document.metadata or {},
                        }
                    ),
                )
            )

        with self._session() as client:
            if client.collection_exists(collection_name=self.collection_name):
                client.delete_collection(collection_name=self.collection_name)

            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            client.upsert(collection_name=self.collection_name, points=points, wait=True)

    def add(self, documents: list[Document]) -> None:
        if not documents:
            return

        texts = [doc.page_content for doc in documents]
        vectors = self._embed_texts(texts)
        vector_size = len(vectors[0])

        points = [
            PointStruct(
                id=str(uuid4()),
                vector=vector,
                payload=_sanitize_payload(
                    {
                        "page_content": document.page_content,
                        "metadata": document.metadata or {},
                    }
                ),
            )
            for vector, document in zip(vectors, documents, strict=True)
        ]

        with self._session() as client:
            self._ensure_collection(client, vector_size)
            client.upsert(collection_name=self.collection_name, points=points, wait=True)

    def delete_by_source(self, source: str) -> None:
        """Remove all document chunks matching the given source path."""
        with self._session() as client:
            if not client.collection_exists(collection_name=self.collection_name):
                return
            client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[FieldCondition(key="metadata.source", match=MatchValue(value=source))]
                ),
            )

    def update_by_source(self, source: str, documents: list[Document]) -> None:
        """Replace all chunks for a source with new documents."""
        self.delete_by_source(source)
        if documents:
            self.add(documents)

    def similarity_search(self, message: str, k: int = 5) -> list[Document]:
        query_vector = self._embed_texts([message])
        with self._session() as client:
            if not client.collection_exists(collection_name=self.collection_name):
                return []
            results = client.query_points(
                collection_name=self.collection_name,
                query=query_vector[0],
                limit=k,
                with_payload=True,
            )

        documents: list[Document] = []
        logger.debug("similarity_search_completed", hits=len(results.points), k=k)
        for result in results.points:
            payload = result.payload or {}
            metadata = payload.get("metadata", {})
            documents.append(
                Document(
                    page_content=str(payload.get("page_content", "")),
                    metadata=_sanitize_payload(metadata),
                )
            )
        return documents

    def _ensure_collection(self, client: QdrantClient, vector_size: int) -> None:
        if client.collection_exists(collection_name=self.collection_name):
            return
        client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    def close(self) -> None:
        with self._client_lock:
            client, self._client = self._client, None
        if client is not None:
            try:
                client.close()
            except Exception as error:
                logger.warning("qdrant_client_close_failed", error=str(error))

        with self._embedding_lock:
            embeddings = self._embeddings
            self._embeddings = None

        if embeddings is None:
            return

        close = getattr(embeddings, "close", None)
        if callable(close):
            close()


def _kb_cache_key(path: Path, collection_name: str) -> tuple[str, str]:
    resolved_path = str(path.expanduser().resolve())
    return resolved_path, collection_name


def open_knowledge_base(
    path: Path, collection_name: str = DEFAULT_COLLECTION_NAME
) -> QdrantKnowledgeBase:
    key = _kb_cache_key(path, collection_name)
    with _KB_LOCK:
        knowledge_base = _KB_INSTANCES.get(key)
        if knowledge_base is None:
            knowledge_base = QdrantKnowledgeBase(
                path=Path(key[0]), collection_name=collection_name
            )
            _KB_INSTANCES[key] = knowledge_base
        return knowledge_base


def close_knowledge_base(
    path: Path | None = None,
    collection_name: str | None = None,
) -> None:
    with _KB_LOCK:
        if path is None:
            knowledge_bases = list(_KB_INSTANCES.values())
            _KB_INSTANCES.clear()
        else:
            key = _kb_cache_key(path, collection_name or DEFAULT_COLLECTION_NAME)
            knowledge_base = _KB_INSTANCES.pop(key, None)
            knowledge_bases = [knowledge_base] if knowledge_base is not None else []

    for knowledge_base in knowledge_bases:
        knowledge_base.close()


def create_knowledge_base(collection_name: str = DEFAULT_COLLECTION_NAME) -> QdrantKnowledgeBase:
    knowledge_base = open_knowledge_base(path=RAG_PATH, collection_name=collection_name)
    knowledge_base.create()
    return knowledge_base


def rebuild_knowledge_base(
    path: Path,
    documents: list[Document],
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> QdrantKnowledgeBase:
    knowledge_base = open_knowledge_base(path=path, collection_name=collection_name)
    knowledge_base.rebuild(documents=documents)
    return knowledge_base


def update_knowledge_base(
    path: Path,
    documents: list[Document],
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> QdrantKnowledgeBase:
    knowledge_base = open_knowledge_base(path=path, collection_name=collection_name)
    knowledge_base.add(documents=documents)
    return knowledge_base


def resync_knowledge_base(
    path: Path,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> QdrantKnowledgeBase:
    _documents = load_vault_markdown_dataset_after_modified(after_modified=time.time())
    return update_knowledge_base(path=path, documents=_documents, collection_name=collection_name)
