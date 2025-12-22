"""
Pytest fixtures for LLM mocking.

These fixtures are automatically available when pytest-mockllm is installed.
No configuration or imports needed - just use them in your tests!
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generator, Optional

import pytest

from pytest_mockllm.providers.openai import OpenAIMock
from pytest_mockllm.providers.anthropic import AnthropicMock
from pytest_mockllm.providers.gemini import GeminiMock
from pytest_mockllm.recording import LLMRecorder

if TYPE_CHECKING:
    from _pytest.fixtures import FixtureRequest


@pytest.fixture
def mock_openai() -> Generator[OpenAIMock, None, None]:
    """
    Mock OpenAI API for testing.
    
    Automatically patches the OpenAI client to return mock responses.
    No API key or network access required.
    
    Example:
        >>> def test_my_chatbot(mock_openai):
        ...     mock_openai.add_response("Hello! How can I help?")
        ...     
        ...     from openai import OpenAI
        ...     client = OpenAI(api_key="fake")
        ...     response = client.chat.completions.create(
        ...         model="gpt-4o",
        ...         messages=[{"role": "user", "content": "Hi!"}]
        ...     )
        ...     
        ...     assert "help" in response.choices[0].message.content.lower()
        ...     assert mock_openai.call_count == 1
    
    Advanced Example (streaming):
        >>> def test_streaming(mock_openai):
        ...     mock_openai.add_response("This is a streamed response")
        ...     
        ...     client = OpenAI(api_key="fake")
        ...     stream = client.chat.completions.create(
        ...         model="gpt-4o",
        ...         messages=[{"role": "user", "content": "Hi!"}],
        ...         stream=True,
        ...     )
        ...     
        ...     full_response = ""
        ...     for chunk in stream:
        ...         if chunk.choices[0].delta.content:
        ...             full_response += chunk.choices[0].delta.content
        ...     
        ...     assert "streamed" in full_response
    """
    with OpenAIMock() as mock:
        yield mock


@pytest.fixture
def mock_anthropic() -> Generator[AnthropicMock, None, None]:
    """
    Mock Anthropic Claude API for testing.
    
    Automatically patches the Anthropic client to return mock responses.
    No API key or network access required.
    
    Example:
        >>> def test_claude_bot(mock_anthropic):
        ...     mock_anthropic.add_response("I'd be happy to help!")
        ...     
        ...     from anthropic import Anthropic
        ...     client = Anthropic(api_key="fake")
        ...     response = client.messages.create(
        ...         model="claude-3-5-sonnet-20241022",
        ...         max_tokens=1024,
        ...         messages=[{"role": "user", "content": "Hello!"}]
        ...     )
        ...     
        ...     assert "happy" in response.content[0].text.lower()
    """
    with AnthropicMock() as mock:
        yield mock


@pytest.fixture
def mock_gemini() -> Generator[GeminiMock, None, None]:
    """
    Mock Google Gemini API for testing.
    
    Automatically patches the Google GenerativeAI client to return mock responses.
    No API key or network access required.
    
    Example:
        >>> def test_gemini_bot(mock_gemini):
        ...     mock_gemini.add_response("Here's what I found...")
        ...     
        ...     import google.generativeai as genai
        ...     model = genai.GenerativeModel("gemini-1.5-pro")
        ...     response = model.generate_content("Tell me about AI")
        ...     
        ...     assert "found" in response.text.lower()
    """
    with GeminiMock() as mock:
        yield mock


@pytest.fixture
def mock_llm(request: "FixtureRequest") -> Generator[OpenAIMock, None, None]:
    """
    Universal LLM mock - defaults to OpenAI.
    
    Use the @pytest.mark.llm_mock(provider="anthropic") marker to switch providers.
    
    Example:
        >>> def test_with_default(mock_llm):
        ...     mock_llm.add_response("Works with OpenAI by default")
        ...     # ... your test code
        
        >>> @pytest.mark.llm_mock(provider="anthropic")
        ... def test_with_anthropic(mock_llm):
        ...     mock_llm.add_response("Now uses Anthropic!")
        ...     # ... your test code
    """
    # Check for marker to determine provider
    marker = request.node.get_closest_marker("llm_mock")
    provider = "openai"
    
    if marker:
        provider = marker.kwargs.get("provider", "openai")
    
    mock_classes = {
        "openai": OpenAIMock,
        "anthropic": AnthropicMock,
        "gemini": GeminiMock,
    }
    
    mock_class = mock_classes.get(provider, OpenAIMock)
    
    with mock_class() as mock:
        yield mock


@pytest.fixture
def mock_langchain() -> Generator["LangChainMock", None, None]:
    """
    Mock LangChain LLM and ChatModel for testing.
    
    Provides a mock that works with LangChain's ChatModel interface,
    including chains, agents, and other LangChain components.
    
    Example:
        >>> def test_langchain_chain(mock_langchain):
        ...     mock_langchain.add_response("The answer is 42.")
        ...     
        ...     from langchain_core.prompts import ChatPromptTemplate
        ...     from langchain_openai import ChatOpenAI
        ...     
        ...     prompt = ChatPromptTemplate.from_template("Question: {question}")
        ...     llm = ChatOpenAI(model="gpt-4o", api_key="fake")
        ...     chain = prompt | llm
        ...     
        ...     result = chain.invoke({"question": "What is the meaning of life?"})
        ...     assert "42" in result.content
    """
    from pytest_mockllm.integrations.langchain import LangChainMock
    
    with LangChainMock() as mock:
        yield mock


@pytest.fixture
def llm_recorder(request: "FixtureRequest") -> Generator[LLMRecorder, None, None]:
    """
    Record and replay LLM API responses (VCR-style).
    
    On first run, makes real API calls and saves responses to cassettes.
    On subsequent runs, replays saved responses without network access.
    
    Requires actual API keys for recording mode.
    
    Example:
        >>> @pytest.mark.llm_record
        ... def test_with_recording(llm_recorder):
        ...     # First run: hits real API, saves response
        ...     # Subsequent runs: uses saved response
        ...     
        ...     from openai import OpenAI
        ...     client = OpenAI()  # Uses real API key
        ...     response = client.chat.completions.create(
        ...         model="gpt-4o-mini",
        ...         messages=[{"role": "user", "content": "Say hello!"}]
        ...     )
        ...     
        ...     assert response.choices[0].message.content
    """
    # Get cassette directory from config or use default
    cassette_dir = request.config.getoption("--llm-cassette-dir", default="tests/llm_cassettes")
    record_mode = request.config.getoption("--llm-record", default=False)
    
    # Use test name as cassette name
    test_name = request.node.name
    cassette_path = Path(cassette_dir) / f"{test_name}.yaml"
    
    # Check for llm_record or llm_replay markers
    record_marker = request.node.get_closest_marker("llm_record")
    replay_marker = request.node.get_closest_marker("llm_replay")
    
    mode = "auto"
    if record_marker or record_mode:
        mode = "record"
    elif replay_marker:
        mode = "replay"
    
    recorder = LLMRecorder(cassette_path=cassette_path, mode=mode)
    
    with recorder:
        yield recorder


# Type alias for the LangChain mock (defined here to avoid circular import)
class LangChainMock:
    """Placeholder - real implementation in integrations/langchain.py"""
    pass
