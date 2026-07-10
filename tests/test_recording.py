"""Safety contracts for unfinished recording and replay support."""

from pathlib import Path

import pytest

from pytest_mockllm.recording import LLMRecorder


@pytest.mark.parametrize("mode", ["auto", "record", "replay"])
def test_recorder_fails_closed_before_provider_calls(tmp_path: Path, mode: str) -> None:
    recorder = LLMRecorder(tmp_path / "cassette.yaml", mode=mode)

    with pytest.raises(RuntimeError, match="refuses to fall through to a live API"):
        recorder.__enter__()


def test_explicit_none_mode_allows_pass_through(tmp_path: Path) -> None:
    with LLMRecorder(tmp_path / "cassette.yaml", mode="none") as recorder:
        assert recorder.is_recording is False
