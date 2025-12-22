"""
pytest-mockllm: Zero-config LLM API mocking for pytest.

The ultimate pytest plugin for testing LLM-powered applications without
hitting real APIs, spending money, or dealing with non-deterministic responses.

Quick Start:
    ```python
    def test_my_chatbot(mock_openai):
        mock_openai.add_response("Hello! I'm here to help.")

        result = my_chatbot.chat("Hi!")
        assert "help" in result.lower()
    ```

For more examples and documentation, visit:
https://github.com/pytest-mockllm/pytest-mockllm
"""

from pytest_mockllm.core import MockLLM, MockResponse
from pytest_mockllm.providers.anthropic import AnthropicMock
from pytest_mockllm.providers.gemini import GeminiMock
from pytest_mockllm.providers.openai import OpenAIMock
from pytest_mockllm.recording import LLMRecorder

__version__ = "0.2.1"
__all__ = [
    # Core
    "MockLLM",
    "MockResponse",
    # Provider mocks
    "OpenAIMock",
    "AnthropicMock",
    "GeminiMock",
    # Recording
    "LLMRecorder",
    # Version
    "__version__",
]
