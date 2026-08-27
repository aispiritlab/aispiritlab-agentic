"""Core settings shared across packages."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Core application settings.

    Field names map to environment variables case-insensitively, so
    ``mlflow_tracking_uri`` is populated from ``MLFLOW_TRACKING_URI``.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # MLflow. Tracking and registry share one server by default (see `make mlflow-ui`).
    mlflow_tracking_uri: str = "http://127.0.0.1:5001"
    mlflow_registry_uri: str = "http://127.0.0.1:5001"
    mlflow_experiment_name: str = "AI Spirit"
    mlflow_evaluation_experiment_name: str = "AI Spirit/evaluation"

    debug: bool = False


settings = Settings()
