import os
import threading
from pathlib import Path
from uuid import uuid4

from agentic.observability import LLMTracer, build_tracer
import mlflow
from core import settings as core_settings

_LOCK = threading.Lock()
_INITIALIZED = False
_RUN_ID: str | None = None
_EXPERIMENT_ID: str | None = None


def get_tracking_uri() -> str:
    return os.getenv("MLFLOW_TRACKING_URI", core_settings.mlflow_registry_uri)


def get_experiment_name(*, evaluation: bool = False) -> str:
    if evaluation:
        return os.getenv(
            "MLFLOW_EVALUATION_EXPERIMENT_NAME",
            core_settings.mlflow_evaluation_experiment_name,
        )
    return os.getenv("MLFLOW_EXPERIMENT_NAME", core_settings.mlflow_experiment_name)


def ensure_experiment(*, evaluation: bool = False) -> str | None:
    tracking_uri = get_tracking_uri()
    experiment_name = get_experiment_name(evaluation=evaluation)
    try:
        mlflow.set_tracking_uri(tracking_uri)
        experiment = mlflow.set_experiment(experiment_name)
    except Exception:
        return None
    experiment_id = getattr(experiment, "experiment_id", None)
    if experiment_id is None:
        return None
    return str(experiment_id)


def init_tracing() -> str:
    """Initialize MLflow tracing with experiment and run ID.

    Returns:
        The run ID for the current tracing session.
    """
    global _INITIALIZED, _RUN_ID, _EXPERIMENT_ID
    with _LOCK:
        if _INITIALIZED and _RUN_ID is not None:
            return _RUN_ID

        data_path = Path(__file__).parents[4] / "data"
        data_path.mkdir(parents=True, exist_ok=True)

        run_id = str(uuid4())[:8]
        tracking_uri = get_tracking_uri()

        try:
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.autolog()
            _EXPERIMENT_ID = ensure_experiment()
        except Exception:
            # Tracing must never block app startup.
            pass

        _RUN_ID = run_id
        _INITIALIZED = True
        return run_id


def create_tracer(enabled: bool = True) -> LLMTracer:
    tracking_uri = get_tracking_uri()
    return build_tracer(enabled=enabled, backend="mlflow", tracking_uri=tracking_uri)


def get_tracing_run_id() -> str | None:
    return _RUN_ID


def get_experiment_id(*, evaluation: bool = False) -> str | None:
    global _EXPERIMENT_ID
    if evaluation:
        return ensure_experiment(evaluation=True)
    with _LOCK:
        if _EXPERIMENT_ID is not None:
            return _EXPERIMENT_ID
        _EXPERIMENT_ID = ensure_experiment()
        return _EXPERIMENT_ID
