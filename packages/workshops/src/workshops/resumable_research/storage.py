from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import replace
import fcntl
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from .model import PendingFileOperation, ResearchDocument

_SAFE_RESEARCH_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class ResearchFileError(RuntimeError):
    pass


class ResearchFileNotFoundError(ResearchFileError):
    pass


class ResearchFileIntegrityError(ResearchFileError):
    pass


class ResearchFileRepository:
    """Atomic, lock-protected storage for research evidence and summaries."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).expanduser().resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    def path_for(self, research_id: str) -> Path:
        if not _SAFE_RESEARCH_ID.fullmatch(research_id):
            raise ValueError("research_id may contain only letters, digits, '.', '_' and '-'")
        return self._root / f"{research_id}.research.json"

    def exists(self, research_id: str) -> bool:
        return self.path_for(research_id).exists()

    def load(self, research_id: str) -> ResearchDocument:
        path = self.path_for(research_id)
        if not path.exists():
            raise ResearchFileNotFoundError(f"Research file does not exist: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, Mapping):
                raise ValueError("Research file root must be an object")
            return ResearchDocument.from_dict(payload)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ResearchFileIntegrityError(f"Invalid research file {path}: {error}") from error

    def stage_operation(
        self,
        research_id: str,
        *,
        operation_id: str,
        event_type: str,
        event_data: Mapping[str, Any],
        mutate: Callable[[ResearchDocument], ResearchDocument],
        initial: Callable[[], ResearchDocument] | None = None,
    ) -> PendingFileOperation | None:
        path = self.path_for(research_id)
        with self._lock(path):
            if path.exists():
                document = self.load(research_id)
            elif initial is not None:
                document = initial()
            else:
                raise ResearchFileNotFoundError(f"Research file does not exist: {path}")

            if operation_id in document.committed_operation_ids:
                return None
            for pending in document.pending_operations:
                if pending.operation_id == operation_id:
                    return pending

            mutated = replace(mutate(document), revision=document.revision + 1)
            data = {
                **event_data,
                "operation_id": operation_id,
                "file_revision": mutated.revision,
                "content_sha256": mutated.content_sha256,
            }
            operation = PendingFileOperation(
                operation_id=operation_id,
                event_type=event_type,
                data=data,
            )
            staged = replace(
                mutated,
                pending_operations=(*mutated.pending_operations, operation),
            )
            self._write_atomic(path, staged)
            return operation

    def mark_committed(self, research_id: str, operation_id: str) -> None:
        path = self.path_for(research_id)
        with self._lock(path):
            document = self.load(research_id)
            if operation_id in document.committed_operation_ids:
                return
            pending = tuple(
                operation
                for operation in document.pending_operations
                if operation.operation_id != operation_id
            )
            if len(pending) == len(document.pending_operations):
                raise ResearchFileIntegrityError(
                    f"Operation {operation_id!r} is neither pending nor committed"
                )
            committed = (*document.committed_operation_ids, operation_id)
            self._write_atomic(
                path,
                replace(
                    document,
                    pending_operations=pending,
                    committed_operation_ids=committed,
                ),
            )

    @contextmanager
    def _lock(self, path: Path) -> Iterator[None]:
        lock_path = path.with_suffix(f"{path.suffix}.lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _write_atomic(path: Path, document: ResearchDocument) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as file:
                json.dump(document.to_dict(), file, ensure_ascii=False, indent=2, sort_keys=True)
                file.write("\n")
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary_path, path)
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
