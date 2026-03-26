from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Document:
    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)
