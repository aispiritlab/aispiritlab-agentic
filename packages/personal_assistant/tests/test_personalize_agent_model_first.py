from personal_assistant.agents.personalize.personalize_agent import extract_posix_token


def test_extract_posix_token_returns_none_without_posix_path() -> None:
    assert extract_posix_token("brak sciezki") is None


def test_extract_posix_token_keeps_absolute_path_with_many_segments() -> None:
    assert extract_posix_token("vault to /Users/me/Obsidian/Vault") == "/Users/me/Obsidian/Vault"


def test_extract_posix_token_strips_leading_slash_for_single_segment() -> None:
    assert extract_posix_token("sciezka /vault") == "vault"


def test_extract_posix_token_preserves_trailing_slash_for_multi_separator_path() -> None:
    assert extract_posix_token("sciezka /vault/") == "/vault/"
