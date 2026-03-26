"""Core settings shared across packages."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Core application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # MLflow registry URI for prompt management
    mlflow_registry_uri: str = "http://127.0.0.1:5001"

    debug: bool = False


settings = Settings()
