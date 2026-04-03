"""SGLang model provider with configurable inference strategies."""

__all__ = ["SGLangProvider"]


def __getattr__(name: str) -> object:
    if name == "SGLangProvider":
        from providers.sglang.provider import SGLangProvider

        return SGLangProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
