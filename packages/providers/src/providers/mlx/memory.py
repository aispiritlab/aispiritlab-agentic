from __future__ import annotations

from structlog import get_logger

logger = get_logger(__name__)


def clear_mlx_cache() -> None:
    """Best-effort release of cached MLX/Metal memory.

    ``mx.metal.clear_cache()`` was the old spelling and now emits a deprecation
    warning on every call; ``mx.clear_cache()`` covers the same buffers.
    """
    try:
        import mlx.core as mx
    except ImportError:
        return

    try:
        mx.clear_cache()
    except Exception as error:
        logger.debug("mlx_cache_clear_failed", error=str(error))
