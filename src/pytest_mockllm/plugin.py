"""
pytest-mockllm plugin registration.

This module is the entry point for pytest's plugin system.
It registers all fixtures and configuration options.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from pytest_mockllm.fixtures import (
    mock_llm,
    mock_openai,
    mock_anthropic,
    mock_gemini,
    mock_langchain,
    llm_recorder,
)

if TYPE_CHECKING:
    from _pytest.config import Config
    from _pytest.config.argparsing import Parser


def pytest_addoption(parser: Parser) -> None:
    """Add pytest-mockllm configuration options."""
    group = parser.getgroup("mockllm", "LLM mocking options")
    
    group.addoption(
        "--llm-record",
        action="store_true",
        default=False,
        help="Record LLM API responses for replay (creates cassettes)",
    )
    
    group.addoption(
        "--llm-cassette-dir",
        action="store",
        default="tests/llm_cassettes",
        help="Directory to store LLM response cassettes (default: tests/llm_cassettes)",
    )
    
    group.addoption(
        "--llm-strict",
        action="store_true",
        default=False,
        help="Fail tests if no mock response is configured (strict mode)",
    )


def pytest_configure(config: Config) -> None:
    """Configure pytest-mockllm plugin."""
    # Register markers
    config.addinivalue_line(
        "markers",
        "llm_mock(response=None, provider=None): Configure LLM mock for this test",
    )
    config.addinivalue_line(
        "markers",
        "llm_record: Record real LLM responses for this test (requires API keys)",
    )
    config.addinivalue_line(
        "markers",
        "llm_replay: Replay recorded LLM responses (fails if cassette missing)",
    )


# Export fixtures for pytest discovery
__all__ = [
    "mock_llm",
    "mock_openai", 
    "mock_anthropic",
    "mock_gemini",
    "mock_langchain",
    "llm_recorder",
    "pytest_addoption",
    "pytest_configure",
]
