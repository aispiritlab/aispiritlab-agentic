"""The embedded Qdrant client is opened once and shared under a lock.

Embedded Qdrant takes an exclusive lock on its storage directory, so a client
per call fails the moment two chat requests overlap. Everything below asserts
the shape that fixes that: one lazily-created client, exclusive access per
operation, and a clean teardown that allows re-opening afterwards.
"""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
import time

import pytest

import knowledge_base.store as store_module


class FakeQdrantClient:
    """Counts how many clients exist and who is inside a session."""

    instances = 0
    live = 0
    closed = 0
    close_error: Exception | None = None

    def __init__(self, path: str) -> None:
        type(self).instances += 1
        type(self).live += 1
        self.path = path

    def close(self) -> None:
        type(self).closed += 1
        type(self).live -= 1
        error = type(self).close_error
        if error is not None:
            raise error


@pytest.fixture(autouse=True)
def fake_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[type[FakeQdrantClient]]:
    FakeQdrantClient.instances = 0
    FakeQdrantClient.live = 0
    FakeQdrantClient.closed = 0
    FakeQdrantClient.close_error = None
    monkeypatch.setattr(store_module, "QdrantClient", FakeQdrantClient)
    store_module.close_knowledge_base()
    yield FakeQdrantClient
    store_module.close_knowledge_base()


def _store(tmp_path: Path) -> store_module.QdrantKnowledgeBase:
    return store_module.QdrantKnowledgeBase(path=tmp_path)


# ---------------------------------------------------------------------------
# Client lifetime
# ---------------------------------------------------------------------------


def test_the_client_is_not_opened_until_it_is_needed(tmp_path: Path) -> None:
    _store(tmp_path)

    assert FakeQdrantClient.instances == 0


def test_the_first_session_opens_exactly_one_client(tmp_path: Path) -> None:
    store = _store(tmp_path)

    with store._session() as client:
        assert isinstance(client, FakeQdrantClient)

    assert FakeQdrantClient.instances == 1
    assert FakeQdrantClient.live == 1, "the client must outlive the session"


def test_repeated_sessions_reuse_the_same_client(tmp_path: Path) -> None:
    store = _store(tmp_path)

    with store._session() as first:
        pass
    with store._session() as second:
        pass

    assert first is second
    assert FakeQdrantClient.instances == 1


def test_the_session_is_reentrant(tmp_path: Path) -> None:
    # Operations nest — similarity_search opens a session and calls
    # _ensure_collection, which needs one too. A plain Lock would deadlock.
    store = _store(tmp_path)

    with store._session() as outer, store._session() as inner:
        assert outer is inner

    assert FakeQdrantClient.instances == 1


def test_the_storage_directory_is_created_eagerly(tmp_path: Path) -> None:
    target = tmp_path / "kb" / "nested"

    store_module.QdrantKnowledgeBase(path=target)

    assert target.is_dir()


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


def test_concurrent_sessions_never_overlap(tmp_path: Path) -> None:
    store = _store(tmp_path)
    lock = threading.Lock()
    inside = 0
    max_inside = 0

    def use_session(_: int) -> None:
        nonlocal inside, max_inside
        with store._session():
            with lock:
                inside += 1
                max_inside = max(max_inside, inside)
            time.sleep(0.01)
            with lock:
                inside -= 1

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(use_session, range(16)))

    assert max_inside == 1, "two operations shared the embedded client at once"


def test_a_concurrent_first_use_still_opens_one_client(tmp_path: Path) -> None:
    # The race that matters: eight chat requests arriving before the store has
    # ever been touched. Two clients here would mean an exclusive-lock crash.
    store = _store(tmp_path)

    def touch(_: int) -> None:
        with store._session():
            pass

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(touch, range(8)))

    assert FakeQdrantClient.instances == 1


# ---------------------------------------------------------------------------
# Teardown
# ---------------------------------------------------------------------------


def test_close_releases_the_client(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with store._session():
        pass

    store.close()

    assert FakeQdrantClient.closed == 1
    assert FakeQdrantClient.live == 0


def test_a_store_can_be_reopened_after_close(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with store._session() as first:
        pass
    store.close()

    with store._session() as second:
        pass

    assert second is not first
    assert FakeQdrantClient.instances == 2


def test_close_is_idempotent(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with store._session():
        pass

    store.close()
    store.close()

    assert FakeQdrantClient.closed == 1


def test_close_on_an_untouched_store_does_nothing(tmp_path: Path) -> None:
    _store(tmp_path).close()

    assert FakeQdrantClient.closed == 0


def test_a_failing_client_close_does_not_propagate(tmp_path: Path) -> None:
    # Shutdown runs from an atexit handler; one bad client must not take the
    # whole teardown down with it.
    store = _store(tmp_path)
    with store._session():
        pass
    FakeQdrantClient.close_error = RuntimeError("storage already released")

    store.close()

    with store._session() as fresh:
        assert isinstance(fresh, FakeQdrantClient)


# ---------------------------------------------------------------------------
# Instance cache
# ---------------------------------------------------------------------------


def test_one_instance_per_path_and_collection(tmp_path: Path) -> None:
    first = store_module.open_knowledge_base(tmp_path)
    second = store_module.open_knowledge_base(tmp_path)

    assert first is second


def test_different_collections_get_different_instances(tmp_path: Path) -> None:
    first = store_module.open_knowledge_base(tmp_path, "notes")
    second = store_module.open_knowledge_base(tmp_path, "sources")

    assert first is not second
    assert first.collection_name == "notes"
    assert second.collection_name == "sources"


def test_different_paths_get_different_instances(tmp_path: Path) -> None:
    first = store_module.open_knowledge_base(tmp_path / "a")
    second = store_module.open_knowledge_base(tmp_path / "b")

    assert first is not second


def test_closing_by_path_only_drops_that_instance(tmp_path: Path) -> None:
    kept = store_module.open_knowledge_base(tmp_path / "kept")
    dropped = store_module.open_knowledge_base(tmp_path / "dropped")

    store_module.close_knowledge_base(tmp_path / "dropped")

    assert store_module.open_knowledge_base(tmp_path / "kept") is kept
    assert store_module.open_knowledge_base(tmp_path / "dropped") is not dropped


def test_closing_without_a_path_drops_everything(tmp_path: Path) -> None:
    first = store_module.open_knowledge_base(tmp_path / "a")
    second = store_module.open_knowledge_base(tmp_path / "b")

    store_module.close_knowledge_base()

    assert store_module.open_knowledge_base(tmp_path / "a") is not first
    assert store_module.open_knowledge_base(tmp_path / "b") is not second


def test_concurrent_opens_return_one_instance(tmp_path: Path) -> None:
    with ThreadPoolExecutor(max_workers=8) as pool:
        instances = list(pool.map(lambda _: store_module.open_knowledge_base(tmp_path), range(8)))

    assert len({id(instance) for instance in instances}) == 1
