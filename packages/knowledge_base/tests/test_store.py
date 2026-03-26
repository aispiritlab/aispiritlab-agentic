from pathlib import Path

import knowledge_base.store as store_module


def test_open_knowledge_base_shares_instance_and_close_releases_embeddings(
    monkeypatch,
    tmp_path: Path,
) -> None:
    created: list[str] = []
    closed: list[str] = []

    class FakeEmbedder:
        def __init__(self) -> None:
            created.append("created")

        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[float(len(text))] for text in texts]

        def close(self) -> None:
            closed.append("closed")

    monkeypatch.setattr(store_module, "Embed4All", FakeEmbedder)
    store_module.close_knowledge_base()

    first = store_module.open_knowledge_base(tmp_path)
    second = store_module.open_knowledge_base(tmp_path)

    assert first is second
    assert created == []

    assert first._embed_texts(["one"]) == [[3.0]]
    assert second._embed_texts(["two"]) == [[3.0]]
    assert created == ["created"]

    store_module.close_knowledge_base(tmp_path)
    assert closed == ["closed"]

    third = store_module.open_knowledge_base(tmp_path)
    assert third is not first
