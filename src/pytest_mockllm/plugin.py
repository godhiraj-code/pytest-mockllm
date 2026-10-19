"""
pytest-mockllm plugin registration.

This module is the entry point for pytest's plugin system.
It registers all fixtures and configuration options.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pytest_mockllm.fixtures import (
    llm_recorder,
    mock_anthropic,
    mock_gemini,
    mock_langchain,
    mock_llm,
    mock_openai,
)

if TYPE_CHECKING:
    from _pytest.config import Config
    from _pytest.config.argparsing import Parser


def pytest_addoption(parser: Parser) -> None:
    """Add pytest-mockllm configuration options."""
    group = parser.getgroup("mockllm", "LLM mocking options")

    group.addoption(
        "--llm-record",
        action="store",
        nargs="?",
        const="all",
        default="none",
        choices=("none", "once", "new_episodes", "all"),
        help="Cassette mode: none (offline default), once, new_episodes, all; bare flag means all",
    )

    group.addoption(
        "--llm-cassette-dir",
        action="store",
        default=None,
        help="Directory to store LLM response cassettes (default: tests/llm_cassettes)",
    )

    group.addoption(
        "--llm-strict",
        action="store_true",
        default=False,
        help="Fail tests if no mock response is configured (strict mode)",
    )

    group.addoption(
        "--llm-stats",
        action="store_true",
        default=False,
        help="Show estimated token usage and cost savings (off by default)",
    )
    group.addoption(
        "--no-llm-stats",
        action="store_true",
        default=False,
        help="Hide token/cost statistics even when llm_stats is enabled in configuration",
    )
    parser.addini("llm_stats", "Show estimated token/cost statistics", type="bool", default=False)

    parser.addini(
        "llm_cassette_dir",
        "Directory to store LLM response cassettes",
        default="tests/llm_cassettes",
    )
    parser.addini(
        "llm_strict",
        "Fail tests if no mock response is configured",
        type="bool",
        default=False,
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
        "llm_record(mode=None): Activate recording/replay; CLI mode defaults to offline",
    )
    config.addinivalue_line(
        "markers",
        "llm_replay: Activate offline cassette replay (overrides recording CLI flags)",
    )


def pytest_runtest_setup(item: Any) -> None:
    """Markers activate the recorder without requiring a fixture parameter."""
    if item.get_closest_marker("llm_record") or item.get_closest_marker("llm_replay"):
        if "llm_recorder" not in item.fixturenames:
            item.fixturenames.insert(0, "llm_recorder")


def pytest_sessionstart(session: Any) -> None:
    """Initialize global statistics."""
    from pytest_mockllm.stats import GLOBAL_STATS

    GLOBAL_STATS.reset()


def pytest_terminal_summary(terminalreporter: Any, exitstatus: int, config: Any) -> None:
    """Display LLM usage and cost savings summary."""
    from pytest_mockllm.stats import GLOBAL_STATS

    enabled = config.getoption("--llm-stats") or config.getini("llm_stats")
    if not enabled or config.getoption("--no-llm-stats") or GLOBAL_STATS.total_calls == 0:
        return

    terminalreporter.section("pytest-mockllm stats")

    # Header
    terminalreporter.write_line(
        f"{'Model':<30} {'Calls':<8} {'Prompt':<10} {'Completion':<12} {'Total':<10}"
    )
    terminalreporter.write_line("-" * 75)

    # Detailed rows (sorted by calls)
    sorted_models = sorted(GLOBAL_STATS.model_counts.items(), key=lambda x: x[1], reverse=True)
    for model, count in sorted_models:
        # Note: We don't currently track per-model tokens in GLOBAL_STATS easily,
        # but we can improve GLOBAL_STATS later. For now, show what we have.
        terminalreporter.write_line(f"{model:<30} {count:<8}")

    terminalreporter.write_line("-" * 75)
    terminalreporter.write_line(
        f"{'TOTAL':<30} {GLOBAL_STATS.total_calls:<8} "
        f"{GLOBAL_STATS.total_prompt_tokens:<10} "
        f"{GLOBAL_STATS.total_completion_tokens:<12} "
        f"{GLOBAL_STATS.total_prompt_tokens + GLOBAL_STATS.total_completion_tokens:<10}"
    )

    # Estimated cost avoided by configured mock responses
    cost_str = f"${GLOBAL_STATS.total_cost_saved:,.4f}"
    terminalreporter.write_line("")
    terminalreporter.write_sep("=", f"Total Estimated Cost Saved: {cost_str}", bold=True, blue=True)
    terminalreporter.write_line("")


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
