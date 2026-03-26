"""Local text-to-image generation using MFLUX."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import secrets
import tempfile
from threading import Lock
import time
from uuid import uuid4

from agentic.providers.mlx.memory import clear_mlx_cache


@dataclass(frozen=True, slots=True)
class ImageGenerationResult:
    image_path: str
    prompt: str
    seed: int
    width: int
    height: int
    steps: int
    latency_ms: float = 0.0


class MfluxImageCall:
    """Direct text-to-image generation via the `mflux` Python API."""

    def __init__(
        self,
        model_name: str = "z-image-turbo",
        *,
        quantize: int = 8,
        width: int = 1024,
        height: int = 1024,
        steps: int = 4,
        output_dir: str | Path | None = None,
    ) -> None:
        self._model_name = model_name
        self._quantize = quantize
        self._width = width
        self._height = height
        self._steps = steps
        self._output_dir = Path(output_dir) if output_dir is not None else (
            Path(tempfile.gettempdir()) / "aispiritagent-generated-images"
        )
        self._load_lock = Lock()
        self._inference_lock = Lock()
        self._model: object | None = None

    def _build_model(self) -> object:

        if self._model_name != "z-image-turbo" and self._model_name != "flux2-klein-9b":
            raise ValueError(
                f"Unsupported MFLUX model '{self._model_name}'. Supported models: z-image-turbo, flux2-klein-9b."
            )
        if self._model_name == "flux2-klein-9b":
            try:
                from mflux.models.flux2.variants import Flux2Klein
            except ModuleNotFoundError as error:
                raise RuntimeError(
                    "MFLUX is not installed. Install it with `uv add --prerelease=allow mflux`."
                ) from error

            return Flux2Klein(quantize=self._quantize)

        try:
            from mflux.models.z_image import ZImageTurbo
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "MFLUX is not installed. Install it with `uv add --prerelease=allow mflux`."
            ) from error

        return ZImageTurbo(quantize=self._quantize)

    def _get_model(self) -> object:
        with self._load_lock:
            if self._model is None:
                self._model = self._build_model()
            return self._model

    def generate_image(
        self,
        prompt: str,
        *,
        seed: int | None = None,
        width: int | None = None,
        height: int | None = None,
        steps: int | None = None,
    ) -> ImageGenerationResult:
        resolved_prompt = prompt.strip()
        if not resolved_prompt:
            raise ValueError("Prompt for image generation cannot be empty.")

        resolved_seed = seed if seed is not None else secrets.randbits(32)
        resolved_width = width if width is not None else self._width
        resolved_height = height if height is not None else self._height
        resolved_steps = steps if steps is not None else self._steps
        model = self._get_model()

        with self._inference_lock:
            started = time.monotonic()
            image = model.generate_image(
                prompt=resolved_prompt,
                seed=resolved_seed,
                width=resolved_width,
                height=resolved_height,
                num_inference_steps=resolved_steps,
            )
            latency_ms = round((time.monotonic() - started) * 1000, 2)

        self._output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self._output_dir / f"generated-{uuid4().hex}.png"
        image.save(output_path)
        return ImageGenerationResult(
            image_path=str(output_path),
            prompt=resolved_prompt,
            seed=resolved_seed,
            width=resolved_width,
            height=resolved_height,
            steps=resolved_steps,
            latency_ms=latency_ms,
        )

    def close(self) -> None:
        with self._load_lock:
            self._model = None
        clear_mlx_cache()
