from __future__ import annotations

import os
import shutil
import subprocess
import sys


def main() -> None:
    uv_binary = shutil.which("uv")

    if uv_binary is None:
        raise RuntimeError("`uv` must be installed to run the DeepFabric proxy.")

    env = os.environ.copy()

    if env.get("OPENROUTER_API_KEY"):
        env["OPENAI_API_KEY"] = env["OPENROUTER_API_KEY"]
    elif env.get("MINIMAX_API_KEY"):
        env["OPENAI_API_KEY"] = env["MINIMAX_API_KEY"]

    command = [
        uv_binary,
        "tool",
        "run",
        "--from",
        "deepfabric",
        "deepfabric",
        *sys.argv[1:],
    ]
    completed_process = subprocess.run(command, check=False, env=env)

    raise SystemExit(completed_process.returncode)
