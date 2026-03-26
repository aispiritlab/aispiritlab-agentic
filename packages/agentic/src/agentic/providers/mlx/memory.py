from __future__ import annotations

from structlog import get_logger

logger = get_logger(__name__)


def clear_mlx_cache() -> None:
    """Best-effort release of cached MLX/Metal memory."""
    try:
        import mlx.core as mx
    except Exception:
        return

    try:
        mx.clear_cache()
    except Exception as error:
        logger.debug("mlx_cache_clear_failed", target="mx", error=str(error))

    try:
        mx.metal.clear_cache()
    except Exception as error:
        logger.debug("mlx_cache_clear_failed", target="mx.metal", error=str(error))
