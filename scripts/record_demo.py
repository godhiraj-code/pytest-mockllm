"""Record actual pytest output as asciinema v2, without an asciinema dependency.

Run from an environment with pytest-mockllm[openai] installed. The recording uses
the real elapsed times of pipe output (not a terminal capture); no output is invented.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("docs/demo.cast"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "COLUMNS": "100"}
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"):
        env.pop(key, None)
    command = [
        sys.executable,
        "-m",
        "pytest",
        "examples/test_demo.py",
        "-v",
        "--color=yes",
        "--no-header",
        "-o",
        "addopts=",
    ]
    with output.open("w", encoding="utf-8", newline="\n") as recording:
        recording.write(
            json.dumps(
                {
                    "version": 2,
                    "width": 100,
                    "height": 24,
                    "timestamp": int(time.time()),
                    "title": "pytest-mockllm: streams, tools, retries",
                }
            )
            + "\n"
        )
        started = time.monotonic()
        with subprocess.Popen(
            command,
            cwd=root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
        ) as process:
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                recording.write(
                    json.dumps(
                        [round(time.monotonic() - started, 6), "o", line.replace("\n", "\r\n")]
                    )
                    + "\n"
                )
            return process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
