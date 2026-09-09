"""Settings for the agentic system."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    model_name: str = (
        "Qwen/Qwen3.5-4B"  # "google/gemma-3-4b-it" # google/gemma-3-1b-it for fine tuning
    )
    orchestration_model_name: str = (
        "Qwen/Qwen3.5-2B"  # "google/gemma-3-4b-it" # google/gemma-3-1b-it for fine tuning
    )
    thinking_model: str = "Qwen/Qwen3.5-9B"
    visual_model: str = "lmstudio-community/Qwen3-VL-4B-Thinking-MLX-8bit"
    image_model_name: str = "flux2-klein-9b"
    image_model_quantize: int = 8
    image_width: int = 1024
    image_height: int = 1024
    image_steps: int = 4
    image_output_dir: str | None = None
    event_store_path: str | None = None
    message_store_path: str | None = None
    message_store_batch_size: int = 64
    message_store_flush_interval_seconds: float = 0.05
    message_stream_inline_bytes: int = 4096
    message_stream_chunk_bytes: int = 4096
    api_base_url: str = "http://localhost:1234"
    api_key: str | None = None
    api_timeout: float = 120.0
    chat_model_name: str = ""
    chat_server_name: str = "127.0.0.1"
    chat_server_port: int = 7860
    # "user:password" pairs, comma-separated. Required when chat_server_name is
    # not a loopback address — see chat.launch().
    chat_auth: str = ""
    agentic_transport: str = "in_memory"
    # `user:password@host:port`; the Laser SDK supplies Iggy's TCP scheme.
    laser_connection_string: str = "iggy:iggy@127.0.0.1:8090"
    # One Iggy stream per deployment. Every topic below is inside it.
    laser_stream_prefix: str = "agentic"
    # Kept at 1 on purpose: a target's messages are one ordered sequence, and
    # the chat client tails partition 0. See distributed/transport.py.
    laser_partitions: int = 1
    # Iggy trims by time, not by entry count. "server_default" keeps whatever
    # the broker is configured for; "none" keeps everything.
    laser_message_expiry: str = "server_default"
    # The health topic only, and it must stay well above the liveness TTL.
    laser_health_expiry: str = "10m"
    distributed_require_event_store: bool = False
    distributed_max_delivery_attempts: int = 3
    distributed_retry_min_idle_ms: int = 5_000
    chat_entry_agent: str = "planner"
    distributed_chat_source: str = "chat"
    distributed_chat_timeout_seconds: float = 60.0
    agent_service_name: str = ""
    agent_heartbeat_seconds: float = 5.0
    agent_liveness_ttl_seconds: float = 20.0


settings = Settings()
