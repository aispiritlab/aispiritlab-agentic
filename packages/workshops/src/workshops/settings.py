"""Workshop-specific settings.

Lab configuration lives here rather than in ``agentic_runtime.Settings``: the
framework should not carry knobs that only a teaching lab uses.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["WorkshopSettings", "settings"]


class WorkshopSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Lab 6 — web search via LangSearch
    langsearch_api_key: str | None = None
    langsearch_base_url: str = "https://api.langsearch.com"
    langsearch_timeout: float = 20.0
    lab6_search_results_per_query: int = 5
    lab6_summary_max_results: int = 6
    lab6_summary_snippet_chars: int = 400
    lab6_summary_total_chars: int = 4000


settings = WorkshopSettings()
