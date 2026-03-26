from typing import Any, Protocol

import orjson

from agentic.message import Message, SystemMessage, UserMessage
from agentic.tools import Toolset, Toolsets


class PromptBuilder(Protocol):
    @property
    def system_prompt(self) -> str: ...
    @system_prompt.setter
    def system_prompt(self, value: str) -> None: ...

    @property
    def external_prompt_name(self) -> str: ...
    @external_prompt_name.setter
    def external_prompt_name(self, value: str) -> None: ...

    def build_prompt(
        self,
        message: str | Message,
        system_prompt: str = "",
        toolsets: Toolsets | None = None,
    ) -> str | list[dict[str, str]]: ...

    def tools_instruction(self, toolset: Toolset) -> str: ...


class CorePromptBuilder:
    """Base class for prompt builders with shared functionality."""

    def __init__(self, *, system_prompt: str | None = None, external_prompt_name: str | None = None) -> None:
        if system_prompt is not None:
            self._system_prompt = system_prompt
        elif external_prompt_name is not None:
            from registry import get_prompt

            self._system_prompt = get_prompt(external_prompt_name)
        else:
            self._system_prompt = ""
        self._external_prompt_name = external_prompt_name

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, value: str) -> None:
        self._system_prompt = value

    @property
    def external_prompt_name(self) -> str:
        return self._external_prompt_name

    @external_prompt_name.setter
    def external_prompt_name(self, value: str) -> None:
        self._external_prompt_name = value

    @staticmethod
    def _map_types(type_name: str) -> str:
        mapping = {
            "str": "string",
            "int": "integer",
            "float": "number",
            "bool": "boolean",
            "dict": "object",
            "list": "array",
        }
        return mapping.get(type_name, "string")

    def _is_full_turn(self, text: str) -> bool:
        """Check if text is already a full turn for this format. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement _is_full_turn")

    def tools_instruction(self, toolset: Toolset) -> str:
        functions: list[dict[str, Any]] = []
        for t in toolset._tools:
            properties: dict[str, Any] = {}
            required: list[str] = []
            for arg in t.args:
                properties[arg["name"]] = {"type": self._map_types(str(arg["type"]))}
                if arg["required"]:
                    required.append(arg["name"])
            functions.append(
                {
                    "name": t.name,
                    "description": t.doc.splitlines(),
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                }
            )

        lines = [
            orjson.dumps(functions, option=orjson.OPT_INDENT_2).decode("utf-8"),
        ]

        return "\n".join(lines)

    @staticmethod
    def _message_text(message: str | Message) -> str:
        if isinstance(message, Message):
            return message.get_text()
        return str(message)

    def build_prompt(
        self,
        message: str | Message,
        system_prompt: str = "",
        toolsets: Toolsets | None = None,
    ) -> str | list[dict[str, str]]:
        raise NotImplementedError


class GemmaPromptBuilder(CorePromptBuilder, PromptBuilder):
    def _is_full_turn(self, text: str) -> bool:
        candidate = text.strip()
        return candidate.startswith("<start_of_turn>") and "<end_of_turn>" in candidate

    def build_prompt(
        self,
        message: str | Message,
        system_prompt: str = "",
        toolsets: Toolsets | None = None,
    ) -> str | list[dict[str, str]]:
        message_text = self._message_text(message)
        toolset_prompt = (
            "\n".join([self.tools_instruction(toolset) for toolset in toolsets if toolset])
            if toolsets
            else ""
        )
        system_template = self.system_prompt or system_prompt
        system_text = system_template.replace("{tools}", toolset_prompt).strip()
        if self._is_full_turn(system_text):
            system_turn = system_text.strip()
        else:
            system_turn = SystemMessage(system_text).as_turn()

        if self._is_full_turn(message_text):
            user_turn = message_text.strip()
        else:
            user_turn = UserMessage(message_text).as_turn()

        return f"{system_turn}\n{user_turn}\n<start_of_turn>model\n"


class QwenPromptBuilder(CorePromptBuilder, PromptBuilder):
    def _is_full_turn(self, text: str) -> bool:
        candidate = text.strip()
        return candidate.startswith("<|im_start|>") and "<|im_end|>" in candidate

    @staticmethod
    def _is_gemma_turn(text: str) -> bool:
        candidate = text.strip()
        return candidate.startswith("<start_of_turn>") and "<end_of_turn>" in candidate

    @staticmethod
    def _wrap_turn(role: str, content: str) -> str:
        return f"<|im_start|>{role}\n{content}\n<|im_end|>"

    @classmethod
    def _gemma_to_qwen_turns(cls, text: str) -> str:
        return (
            text.replace("<start_of_turn>system\n", "<|im_start|>system\n")
            .replace("<start_of_turn>user\n", "<|im_start|>user\n")
            .replace("<start_of_turn>assistant\n", "<|im_start|>assistant\n")
            .replace("<start_of_turn>model\n", "<|im_start|>assistant\n")
            .replace("<end_of_turn>", "<|im_end|>")
        )

    def build_prompt(
        self,
        message: str | Message,
        system_prompt: str = "",
        toolsets: Toolsets | None = None,
    ) -> str | list[dict[str, str]]:
        message_text = self._message_text(message)
        toolset_prompt = (
            "\n".join([self.tools_instruction(toolset) for toolset in toolsets if toolset])
            if toolsets
            else ""
        )
        system_template = self.system_prompt or system_prompt
        system_text = system_template.replace("{tools}", toolset_prompt).strip()

        if self._is_full_turn(system_text):
            system_turn = system_text.strip()
        elif self._is_gemma_turn(system_text):
            system_turn = self._gemma_to_qwen_turns(system_text).strip()
        else:
            system_turn = self._wrap_turn("system", system_text)

        if self._is_full_turn(message_text):
            user_turn = message_text.strip()
        elif self._is_gemma_turn(message_text):
            user_turn = self._gemma_to_qwen_turns(message_text).strip()
        else:
            user_turn = self._wrap_turn("user", message_text)

        return f"{system_turn}\n{user_turn}\n<|im_start|>assistant\n"


class ChatPromptBuilder(CorePromptBuilder, PromptBuilder):
    """Prompt builder that returns OpenAI-compatible messages list."""

    def _is_full_turn(self, text: str) -> bool:
        return False

    def build_prompt(
        self,
        message: str | Message,
        system_prompt: str = "",
        toolsets: Toolsets | None = None,
    ) -> list[dict[str, str]]:
        message_text = self._message_text(message)
        toolset_prompt = (
            "\n".join([self.tools_instruction(toolset) for toolset in toolsets if toolset])
            if toolsets
            else ""
        )
        system_template = self.system_prompt or system_prompt
        system_text = system_template.replace("{tools}", toolset_prompt).strip()

        messages: list[dict[str, str]] = []
        if system_text:
            messages.append({"role": "system", "content": system_text})
        messages.append({"role": "user", "content": message_text})
        return messages


class PromptTemplate:
    def __init__(self, template: str, context_variables: list[str]):
        self._template = template
        self._context_variables = context_variables

    def format(self, **kwargs) -> str:
        return self._template.format(**kwargs)