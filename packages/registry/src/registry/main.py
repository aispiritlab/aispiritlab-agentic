import mlflow

from registry import Prompts
from registry.prompts import (
    DISCOVERY_NOTES_PROMPT,
    DECISION_PROMPT,
    GREETING_PROMPT,
    MANAGE_NOTES_PROMPT,
    ORGANIZER_PROMPT,
    SAGE_PROMPT,
)
from registry.register_prompt import register_prompt, RegisterPrompt


def init_registry_prompt():
    register_prompt(
        RegisterPrompt(name=Prompts.GREETING.value, prompt=GREETING_PROMPT)
    )
    register_prompt(
        RegisterPrompt(
            name=Prompts.MANAGE_NOTES.value, prompt=MANAGE_NOTES_PROMPT
        )
    )
    register_prompt(
        RegisterPrompt(
            name=Prompts.DISCOVERY_NOTES.value, prompt=DISCOVERY_NOTES_PROMPT
        )
    )
    register_prompt(
        RegisterPrompt(
            name=Prompts.ORGANIZER.value, prompt=ORGANIZER_PROMPT
        )
    )
    register_prompt(
        RegisterPrompt(name=Prompts.SAGE.value, prompt=SAGE_PROMPT)
    )
    register_prompt(
        RegisterPrompt(name=Prompts.DECISION.value, prompt=DECISION_PROMPT)
    )

def main():
    init_registry_prompt()


if __name__ == "__main__":
    main()
