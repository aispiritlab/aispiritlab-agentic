FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim AS builder

WORKDIR /app
ENV UV_LINK_MODE=copy

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY packages ./packages

RUN uv sync --no-dev --package chat
RUN uv sync --no-dev --package workshops --inexact

# Same Debian release as the builder, so wheels compiled above keep matching glibc.
FROM python:3.14-slim-bookworm AS runtime

WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app

COPY --from=builder --chown=app:app /app /app

USER app

EXPOSE 7860

HEALTHCHECK --interval=10s --timeout=3s --start-period=40s --retries=6 \
    CMD curl -fsS "http://127.0.0.1:${CHAT_SERVER_PORT:-7860}/" || exit 1

CMD ["lab6-chat"]
