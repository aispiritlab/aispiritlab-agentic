import mlflow
from pydantic import BaseModel

from core import settings


class RegisterPrompt(BaseModel):
    name: str
    prompt: str
    commit_message: str = "Initial commit"
    tags: dict[str, str] = {
        "author": "John Doe"
    }

def register_prompt(prompt: RegisterPrompt):
    mlflow.set_registry_uri(settings.mlflow_registry_uri)
    mlflow.genai.register_prompt(
        name=prompt.name,
        template=prompt.prompt,
        commit_message=prompt.commit_message,
        tags=prompt.tags
    )