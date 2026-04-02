# Knowledge Base

RAG system for note storage and retrieval. Wraps Qdrant vector DB with markdown document loading and chunking.

## Features

- **QdrantKnowledgeBase** — vector search over indexed documents
- **Vault loader** — load markdown files from a directory
- **Tree-sitter splitters** — language-aware chunking for code and markdown
- **Incremental sync** — update or delete individual notes without full reindex

## Usage

```python
from knowledge_base import get_knowledge_base, update_note_in_kb, delete_note_from_kb

kb = get_knowledge_base()
results = kb.search("event sourcing patterns", top_k=5)

update_note_in_kb("notes/architecture.md")
delete_note_from_kb("notes/old-draft.md")
```

### Generate knowledge base

```bash
make generate-kb
```
