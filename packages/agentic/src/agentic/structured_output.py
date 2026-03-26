from typing import Any


class StructuredOutput:
    def __init__(self, csl_output: object):
        self.csl_output = csl_output
        self._is_structured: bool | None = None
    def parse(self, response: str) -> Any:
        return self.csl_output.parse(response)

    def is_structured(self, response: str) -> bool:
        return False