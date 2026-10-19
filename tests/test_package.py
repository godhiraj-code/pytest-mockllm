"""Published metadata must agree with the importable package."""

from importlib.metadata import version
from pathlib import Path

import pytest_mockllm


def test_version_and_typing_marker():
    assert version("pytest-mockllm") == pytest_mockllm.__version__ == "0.4.0"
    assert Path(pytest_mockllm.__file__).with_name("py.typed").exists()
