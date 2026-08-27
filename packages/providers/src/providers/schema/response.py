"""Inference response.

``InferenceResponse`` is the client-layer name for the same payload the model
layer calls ``ModelResponse``; it is an alias rather than a copy so values cross
layer boundaries without conversion.
"""

from __future__ import annotations

from providers.models.response import ModelResponse

__all__ = ["InferenceResponse", "ModelResponse"]

# A plain alias, not a `type` statement: this name is also used as a constructor.
InferenceResponse = ModelResponse
