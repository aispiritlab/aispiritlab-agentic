from personal_assistant.rag import build_context
from knowledge_base.documents import Document


def test_build_context_uses_note_name_from_source_path() -> None:
    context = build_context(
        [
            Document(
                page_content="ala ma kota",
                metadata={
                    "source": "/tmp/vault/bron z tekstem.md",
                    "header": "bron z tekstem.md chunk=1",
                },
            )
        ]
    )

    assert context == "**bron z tekstem.** ala ma kota"


def test_build_context_falls_back_to_header_without_chunk_suffix() -> None:
    context = build_context(
        [
            Document(
                page_content="ania ma kota",
                metadata={"header": "folder/superasna.md chunk=2"},
            )
        ]
    )

    assert context == "**superasna.** ania ma kota"
