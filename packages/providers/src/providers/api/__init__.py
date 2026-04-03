"""OpenAI-compatible API provider."""

__all__ = ["OpenAIProvider"]


def __getattr__(name: str) -> object:
    if name == "OpenAIProvider":
        from providers.api.provider import OpenAIProvider

        return OpenAIProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
