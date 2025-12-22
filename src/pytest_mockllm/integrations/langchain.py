"""
LangChain integration for pytest-mockllm.

Provides seamless mocking for LangChain's LLM and ChatModel interfaces,
enabling easy testing of chains, agents, and other LangChain components.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Iterator
from typing import Any
from unittest.mock import MagicMock, patch

from pytest_mockllm.core import MockError, MockLLM, MockResponse


class LangChainMock(MockLLM):
    """
    Mock for LangChain ChatModel and LLM interfaces.

    Works with:
        - ChatOpenAI, ChatAnthropic, ChatGoogleGenerativeAI
        - LangChain chains (LCEL, legacy)
        - LangChain agents
        - Any component using BaseChatModel

    Example:
        >>> def test_langchain_chain(mock_langchain):
        ...     mock_langchain.add_response("The capital of France is Paris.")
        ...
        ...     from langchain_openai import ChatOpenAI
        ...     from langchain_core.prompts import ChatPromptTemplate
        ...
        ...     llm = ChatOpenAI(model="gpt-4o", api_key="fake")
        ...     prompt = ChatPromptTemplate.from_template("What is the capital of {country}?")
        ...     chain = prompt | llm
        ...
        ...     result = chain.invoke({"country": "France"})
        ...     assert "Paris" in result.content

    Example with agents:
        >>> def test_agent(mock_langchain):
        ...     mock_langchain.add_responses(
        ...         "I need to search for this.",  # Agent thought
        ...         "The answer is 42.",           # Final answer
        ...     )
        ...
        ...     # Your agent code here
        ...     result = agent.invoke({"input": "What is the meaning of life?"})
        ...     assert "42" in result["output"]
    """

    def __init__(self) -> None:
        super().__init__()
        self._default_model: str = "gpt-4o"

    def _raise_provider_error(self, error: MockError) -> None:
        """Raise an error compatible with LangChain's error handling."""
        raise RuntimeError(f"LLM Error ({error.error_type}): {error.message}")

    def _create_message(self, messages: Any = None, **kwargs: Any) -> Any:
        """Create a mock LangChain message response."""
        self._record_call(type="invoke", messages=messages, **kwargs)
        response = self._get_next_response()
        self._simulate_delay(self._get_delay_ms(response))
        return self._build_langchain_message(response)

    async def _create_async_message(self, messages: Any = None, **kwargs: Any) -> Any:
        """Create a mock async LangChain message response."""
        self._record_call(type="ainvoke", messages=messages, **kwargs)
        response = self._get_next_response()
        await self._simulate_delay_async(self._get_delay_ms(response))
        return self._build_langchain_message(response)

    def _build_langchain_message(self, response: MockResponse) -> Any:
        """Build a LangChain AIMessage response."""
        try:
            from langchain_core.messages import AIMessage

            message = AIMessage(
                content=response.content,
                response_metadata={
                    "token_usage": {
                        "prompt_tokens": response.token_usage.prompt_tokens
                        if response.token_usage
                        else 10,
                        "completion_tokens": response.token_usage.completion_tokens
                        if response.token_usage
                        else 50,
                        "total_tokens": response.token_usage.total_tokens
                        if response.token_usage
                        else 60,
                    },
                    "model_name": response.model,
                    "finish_reason": response.finish_reason,
                },
                id=response.id,
            )

            # Handle tool calls if present
            if response.tool_calls:
                message.tool_calls = [
                    {
                        "id": tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                        "name": tc["function"]["name"],
                        "args": tc["function"].get("arguments", {}),
                    }
                    for tc in response.tool_calls
                ]

            return message

        except ImportError:
            # Return MagicMock if langchain not installed
            mock = MagicMock()
            mock.content = response.content
            mock.response_metadata = {
                "token_usage": {
                    "prompt_tokens": response.token_usage.prompt_tokens
                    if response.token_usage
                    else 10,
                    "completion_tokens": response.token_usage.completion_tokens
                    if response.token_usage
                    else 50,
                    "total_tokens": response.token_usage.total_tokens
                    if response.token_usage
                    else 60,
                },
            }
            mock.id = response.id
            mock.tool_calls = response.tool_calls or []
            return mock

    async def _create_async_streaming_response(self, **kwargs: Any) -> AsyncIterator[Any]:
        """Create an async streaming response for LangChain."""
        self._record_call(type="stream", **kwargs)
        response = self._get_next_response()
        await self._simulate_delay_async(self._get_delay_ms(response))

        if response.stream_chunks:
            chunks = response.stream_chunks
        else:
            words = response.content.split()
            chunks = [word + " " for word in words[:-1]] + [words[-1]] if words else [""]

        for i, chunk_content in enumerate(chunks):
            yield self._build_stream_chunk(
                chunk_content,
                is_first=(i == 0),
                is_last=(i == len(chunks) - 1),
                response_id=response.id,
            )

    def _create_streaming_response(self, **kwargs: Any) -> Iterator[Any]:
        """Create a streaming response for LangChain."""
        self._record_call(type="stream", **kwargs)
        response = self._get_next_response()
        self._simulate_delay(self._get_delay_ms(response))

        # Split content into chunks
        if response.stream_chunks:
            chunks = response.stream_chunks
        else:
            words = response.content.split()
            chunks = [word + " " for word in words[:-1]] + [words[-1]] if words else [""]

        for i, chunk_content in enumerate(chunks):
            yield self._build_stream_chunk(
                chunk_content,
                is_first=(i == 0),
                is_last=(i == len(chunks) - 1),
                response_id=response.id,
            )

    def _build_stream_chunk(
        self,
        content: str,
        is_first: bool,
        is_last: bool,
        response_id: str,
    ) -> Any:
        """Build a LangChain streaming chunk."""
        try:
            from langchain_core.messages import AIMessageChunk

            return AIMessageChunk(
                content=content,
                id=response_id if is_first else None,
            )
        except ImportError:
            mock = MagicMock()
            mock.content = content
            mock.id = response_id if is_first else None
            return mock

    def _create_mock_chat_model(self) -> MagicMock:
        """Create a mock ChatModel that can be used in LangChain chains."""
        mock_model = MagicMock()

        # invoke() - main method for LCEL
        def invoke(messages: Any, *args: Any, **kwargs: Any) -> Any:
            return self._create_message(messages=messages, **kwargs)

        mock_model.invoke = invoke

        # stream() - for streaming
        def stream(messages: Any, *args: Any, **kwargs: Any) -> Iterator[Any]:
            return self._create_streaming_response(messages=messages, **kwargs)

        mock_model.stream = stream

        # ainvoke() - async version
        async def ainvoke(messages: Any, *args: Any, **kwargs: Any) -> Any:
            return await self._create_async_message(messages=messages, **kwargs)

        mock_model.ainvoke = ainvoke

        # astream() - async version
        async def astream(messages: Any, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
            async for chunk in self._create_async_streaming_response(messages=messages, **kwargs):
                yield chunk

        mock_model.astream = astream

        # Make it work with LCEL's | operator
        mock_model.__or__ = lambda self, other: mock_model
        mock_model.__ror__ = lambda self, other: MagicMock(
            invoke=invoke, stream=stream, ainvoke=ainvoke, astream=astream
        )

        # bind_tools for function calling
        def bind_tools(tools: list[Any], **kwargs: Any) -> MagicMock:
            return mock_model

        mock_model.bind_tools = bind_tools

        # with_structured_output for structured responses
        mock_model.with_structured_output = lambda schema, **kw: mock_model

        return mock_model

    def __enter__(self) -> LangChainMock:
        """Start mocking LangChain LLM calls."""
        mock_model = self._create_mock_chat_model()
        self._mock_model = mock_model

        # Patch common LangChain LLM imports
        providers_to_patch = [
            "langchain_openai.ChatOpenAI",
            "langchain_openai.AzureChatOpenAI",
            "langchain_anthropic.ChatAnthropic",
            "langchain_google_genai.ChatGoogleGenerativeAI",
            "langchain_community.chat_models.ChatOpenAI",
        ]

        for provider_path in providers_to_patch:
            try:
                # Create a mock class that returns our mock model
                def mock_class(*args: Any, **kwargs: Any) -> MagicMock:
                    return mock_model

                patcher = patch(provider_path, mock_class)
                self._patches.append(patcher)
                patcher.start()
            except Exception:
                # Provider not installed or can't be patched
                pass

        return self

    @property
    def model(self) -> MagicMock:
        """Access the mock model directly for advanced usage."""
        return self._mock_model

    def add_tool_call(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        content: str = "",
    ) -> LangChainMock:
        """
        Add a response with a tool call (for agent testing).

        Example:
            >>> mock_langchain.add_tool_call("search", {"query": "weather"})
            >>> mock_langchain.add_response("The weather is sunny.")
        """
        response = MockResponse(
            content=content,
            tool_calls=[
                {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "function": {
                        "name": name,
                        "arguments": arguments or {},
                    },
                }
            ],
        )
        self._responses.append(response)
        return self
