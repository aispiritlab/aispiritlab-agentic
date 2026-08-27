"""Launch guards for the chat app.

The app switches user identity from the UI and reads per-user vaults, so an
unauthenticated non-local bind hands every profile to anyone reaching the port.
``launch`` is the only place that decision is made.
"""

from __future__ import annotations

from typing import Any, cast

import gradio as gr
import pytest

from chat.app import (
    LOOPBACK_HOSTS,
    ChatAppConfig,
    InsecureExposureError,
    launch,
    parse_auth,
)


class FakeBlocks:
    """Stands in for ``gr.Blocks``; records what ``launch`` would have done."""

    def __init__(self) -> None:
        self.queued = 0
        self.launch_kwargs: dict[str, Any] | None = None

    def queue(self) -> None:
        self.queued += 1

    def launch(self, **kwargs: Any) -> None:
        self.launch_kwargs = kwargs


def _launch(blocks: FakeBlocks, config: ChatAppConfig) -> None:
    """Call the real ``launch`` with a stand-in for ``gr.Blocks``."""
    launch(cast("gr.Blocks", blocks), config)


# ---------------------------------------------------------------------------
# parse_auth
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw", ["", "   ", ",", " , , "])
def test_parse_auth_returns_none_for_an_empty_value(raw: str) -> None:
    # None is what ChatAppConfig.auth expects for "no credentials configured".
    assert parse_auth(raw) is None


def test_parse_auth_reads_a_single_credential() -> None:
    assert parse_auth("jan:tajne") == [("jan", "tajne")]


def test_parse_auth_reads_several_credentials() -> None:
    assert parse_auth("jan:tajne,ola:inne") == [("jan", "tajne"), ("ola", "inne")]


def test_parse_auth_skips_blank_entries() -> None:
    assert parse_auth("jan:tajne,,ola:inne,") == [("jan", "tajne"), ("ola", "inne")]


def test_parse_auth_trims_the_entry_and_the_user_but_not_inside_the_password() -> None:
    # The whole entry is stripped first, so only whitespace *inside* the entry
    # survives. A comma-separated env var cannot represent a password with
    # leading/trailing spaces or a comma; that is the format's limit, not a bug.
    assert parse_auth("  jan  : s3cret ") == [("jan", " s3cret")]
    assert parse_auth("jan:has space") == [("jan", "has space")]


def test_parse_auth_keeps_colons_inside_a_password() -> None:
    assert parse_auth("jan:pass:with:colons") == [("jan", "pass:with:colons")]


@pytest.mark.parametrize(
    "raw",
    [
        "nopassword",  # no separator at all
        "jan:",  # empty password
        ":tajne",  # empty user
        "   :tajne",  # whitespace-only user
        "jan:tajne,broken",  # one bad entry poisons the whole value
    ],
)
def test_parse_auth_rejects_malformed_entries(raw: str) -> None:
    with pytest.raises(ValueError, match="expected 'user:password'"):
        parse_auth(raw)


# ---------------------------------------------------------------------------
# ChatAppConfig
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("host", sorted(LOOPBACK_HOSTS))
def test_loopback_hosts_are_recognised(host: str) -> None:
    assert ChatAppConfig(server_name=host).is_loopback_only is True


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.10", "example.com", ""])
def test_non_loopback_hosts_are_recognised(host: str) -> None:
    assert ChatAppConfig(server_name=host).is_loopback_only is False


def test_config_defaults_to_loopback_without_sharing() -> None:
    config = ChatAppConfig()

    assert config.is_loopback_only is True
    assert config.share is False
    assert config.auth is None


# ---------------------------------------------------------------------------
# launch
# ---------------------------------------------------------------------------


def test_launch_allows_an_unauthenticated_loopback_bind() -> None:
    blocks = FakeBlocks()

    _launch(blocks, ChatAppConfig(server_name="127.0.0.1"))

    assert blocks.queued == 1
    assert blocks.launch_kwargs is not None
    assert blocks.launch_kwargs["server_name"] == "127.0.0.1"


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.10", "example.com"])
def test_launch_refuses_an_unauthenticated_non_local_bind(host: str) -> None:
    blocks = FakeBlocks()

    with pytest.raises(InsecureExposureError, match="CHAT_AUTH"):
        _launch(blocks, ChatAppConfig(server_name=host))

    assert blocks.launch_kwargs is None


def test_launch_refuses_an_unauthenticated_public_share_even_from_loopback() -> None:
    blocks = FakeBlocks()

    with pytest.raises(InsecureExposureError, match="public share link"):
        _launch(blocks, ChatAppConfig(server_name="127.0.0.1", share=True))

    assert blocks.launch_kwargs is None


def test_launch_names_the_offending_host_in_the_error() -> None:
    with pytest.raises(InsecureExposureError, match="0.0.0.0"):
        _launch(FakeBlocks(), ChatAppConfig(server_name="0.0.0.0"))


def test_launch_allows_a_non_local_bind_once_authenticated() -> None:
    blocks = FakeBlocks()
    config = ChatAppConfig(server_name="0.0.0.0", auth=[("jan", "tajne")])

    _launch(blocks, config)

    assert blocks.launch_kwargs is not None
    assert blocks.launch_kwargs["auth"] == [("jan", "tajne")]


def test_launch_accepts_a_callable_authenticator() -> None:
    blocks = FakeBlocks()
    checker = lambda user, password: user == "jan"

    _launch(blocks, ChatAppConfig(server_name="0.0.0.0", auth=checker))

    assert blocks.launch_kwargs is not None
    assert blocks.launch_kwargs["auth"] is checker


def test_launch_refuses_when_parse_auth_found_no_credentials() -> None:
    # The realistic misconfiguration: CHAT_SERVER_NAME=0.0.0.0 with CHAT_AUTH unset.
    config = ChatAppConfig(server_name="0.0.0.0", auth=parse_auth(""))

    with pytest.raises(InsecureExposureError):
        _launch(FakeBlocks(), config)


def test_launch_forwards_the_whole_configuration_to_gradio() -> None:
    blocks = FakeBlocks()
    config = ChatAppConfig(
        title="Asystent",
        server_name="127.0.0.1",
        server_port=7999,
        allowed_paths=["/tmp/uploads"],
        pwa=False,
        auth=[("jan", "tajne")],
        auth_message="Zaloguj się",
    )

    _launch(blocks, config)

    assert blocks.launch_kwargs == {
        "pwa": False,
        "share": False,
        "allowed_paths": ["/tmp/uploads"],
        "server_name": "127.0.0.1",
        "server_port": 7999,
        "auth": [("jan", "tajne")],
        "auth_message": "Zaloguj się",
    }


def test_insecure_exposure_error_is_a_runtime_error() -> None:
    assert issubclass(InsecureExposureError, RuntimeError)
