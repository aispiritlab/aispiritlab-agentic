from datetime import datetime
import os
from pathlib import Path
from uuid import uuid4

from agentic.observability import LLMTracer, build_tracer
import mlflow
from core import settings as core_settings

_INITIALIZED = False
_RUN_ID: str | None = None


def init_tracing() -> str:
    """Initialize MLflow tracing with experiment and run ID.

    Returns:
        The run ID for the current tracing session.
    """
    global _INITIALIZED, _RUN_ID
    if _INITIALIZED and _RUN_ID is not None:
        return _RUN_ID

    data_path = Path(__file__).parents[4] / "data"
    data_path.mkdir(parents=True, exist_ok=True)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", core_settings.mlflow_registry_uri)
    run_id = str(uuid4())[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"AI Spirit - {timestamp} ({run_id})"

    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.autolog()
        mlflow.set_experiment(experiment_name)
    except Exception:
        # Tracing must never block app startup.
        pass

    _RUN_ID = run_id
    _INITIALIZED = True
    return run_id


def create_tracer(enabled: bool = True) -> LLMTracer:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", core_settings.mlflow_registry_uri)
    return build_tracer(enabled=enabled, backend="mlflow", tracking_uri=tracking_uri)
