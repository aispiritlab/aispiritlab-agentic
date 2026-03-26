from typing import Protocol


class Memory(Protocol):
    def retrieve(self, key: str, length: int) -> str:
        ...

    def store(self, key: str, value: str) -> None:
        ...

    def summary(self) -> str:
        """Reserved for future prompt-context integration."""
        ...

class InMemory(Memory):
    def __init__(self):
        self._memory = {}

    def retrieve(self, key: str, length: int) -> str:
        return self._memory.get(key, '')

    def store(self, key: str, value: str) -> None:
        self._memory[key] = value

    def summary(self) -> str:
        # Keep the staged API available even though prompt injection is not wired yet.
        return ""
