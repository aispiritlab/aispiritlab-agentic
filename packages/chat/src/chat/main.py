"""Backward-compatible entry point.

The chat UI has moved to the personal_assistant package.
Use `personal-assistant` command instead.
"""


def main() -> None:
    """Launch the Personal Assistant chat UI (backward compat)."""
    try:
        from personal_assistant.ui.app import launch_app
    except ImportError:
        raise SystemExit(
            "The chat UI has moved to the personal_assistant package.\n"
            "Install it with: uv pip install -e packages/personal_assistant\n"
            "Then run: personal-assistant"
        )
    launch_app()


if __name__ == "__main__":
    main()
