"""Fetch release information from the GitHub Releases API."""

from __future__ import annotations

from dataclasses import dataclass

import httpx
from structlog import get_logger

logger = get_logger(__name__)

_GITHUB_API = "https://api.github.com"


class ReleaseNotFoundError(Exception):
    """Raised when a release cannot be found."""


@dataclass(frozen=True, slots=True)
class ReleaseAsset:
    name: str
    download_url: str
    size: int
    digest: str


@dataclass(frozen=True, slots=True)
class Release:
    tag: str
    assets: tuple[ReleaseAsset, ...]

    def find_asset(self, name_contains: str) -> ReleaseAsset | None:
        for asset in self.assets:
            if name_contains in asset.name:
                return asset
        return None


def _parse_release(data: dict) -> Release:
    assets = tuple(
        ReleaseAsset(
            name=a["name"],
            download_url=a["browser_download_url"],
            size=a.get("size", 0),
            digest=a.get("digest", ""),
        )
        for a in data.get("assets", [])
    )
    return Release(tag=data["tag_name"], assets=assets)


def fetch_latest_release(repo: str) -> Release:
    url = f"{_GITHUB_API}/repos/{repo}/releases/latest"
    logger.info("fetching_latest_release", repo=repo)
    with httpx.Client(timeout=30.0) as client:
        response = client.get(url, headers={"Accept": "application/vnd.github+json"})
        if response.status_code == 404:
            raise ReleaseNotFoundError(f"No releases found for {repo}")
        response.raise_for_status()
        return _parse_release(response.json())


def fetch_release(repo: str, tag: str) -> Release:
    url = f"{_GITHUB_API}/repos/{repo}/releases/tags/{tag}"
    logger.info("fetching_release", repo=repo, tag=tag)
    with httpx.Client(timeout=30.0) as client:
        response = client.get(url, headers={"Accept": "application/vnd.github+json"})
        if response.status_code == 404:
            raise ReleaseNotFoundError(f"Release {tag} not found for {repo}")
        response.raise_for_status()
        return _parse_release(response.json())
