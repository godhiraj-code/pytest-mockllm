"""
Anthropic Claude API mock for pytest-mockllm.

Provides comprehensive mocking for the Anthropic Python SDK including:
- Messages API (sync, async, streaming)
- Tool use
- Vision (images in messages)
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Iterator
from typing import Any, cast
from unittest.mock import MagicMock, patch

from pytest_mockllm.core import MockError, MockLLM, MockResponse


class AnthropicMock(MockLLM):
    """
    Mock for Anthropic Python SDK.

    Supports:
        - client.messages.create()
        - Streaming responses
        - Tool use
        - Async variants

    Example:
        >>> def test_claude_bot(mock_anthropic):
        ...     mock_anthropic.add_response("I'd be happy to help!")
        ...
        ...     client = Anthropic()
        ...     response = client.messages.create(
        ...         model="claude-3-5-sonnet-20241022",
        ...         max_tokens=1024,
        ...         messages=[{"role": "user", "content": "Hello!"}]
        ...     )
        ...     assert "happy" in response.content[0].text.lower()
    """

    def __init__(self) -> None:
        super().__init__()
        self._default_model: str = "claude-3-5-sonnet-20241022"

    def _raise_provider_error(self, error: MockError) -> None:
        """Raise an Anthropic-style API error."""
        try:
            from anthropic import (
                APIError,
                APITimeoutError,
                AuthenticationError,
                BadRequestError,
                RateLimitError,
            )

            error_classes = {
                "rate_limit": RateLimitError,
                "auth": AuthenticationError,
                "timeout": APITimeoutError,
                "server": APIError,
                "invalid_request": BadRequestError,
            }

            error_class = error_classes.get(error.error_type, APIError)

            mock_response = MagicMock()
            mock_response.status_code = {
                "rate_limit": 429,
                "auth": 401,
                "timeout": 408,
                "server": 500,
                "invalid_request": 400,
            }.get(error.error_type, 500)
            mock_response.headers = {}

            raise error_class(
                message=error.message,
                response=mock_response,
                body={"error": {"message": error.message}},
            )
        except ImportError:
            raise RuntimeError(
                f"Anthropic API Error ({error.error_type}): {error.message}"
            ) from None

    def _create_message(self, **kwargs: Any) -> Any:
        """Create a mock message response."""
        self._record_call(type="messages.create", **kwargs)
        model = kwargs.get("model", self._default_model)
        response = self._get_next_response(model=model)
        return self._build_message_response(response, model)

    def _build_message_response(self, response: MockResponse, model: str) -> Any:
        """Build an Anthropic-style Message response object."""
        try:
            from anthropic.types import (
                Message,
                TextBlock,
                ToolUseBlock,
                Usage,
            )

            content_blocks: list[Any] = []

            # Add text content
            if response.content:
                content_blocks.append(TextBlock(type="text", text=response.content))

            # Add tool use if present
            if response.tool_calls:
                for tc in response.tool_calls:
                    content_blocks.append(
                        ToolUseBlock(
                            type="tool_use",
                            id=tc.get("id", f"toolu_{uuid.uuid4().hex[:8]}"),
                            name=tc["function"]["name"],
                            input=tc["function"].get("arguments", {}),
                        )
                    )

            usage = Usage(
                input_tokens=response.token_usage.prompt_tokens if response.token_usage else 10,
                output_tokens=response.token_usage.completion_tokens
                if response.token_usage
                else 50,
            )

            stop_reason = "tool_use" if response.tool_calls else "end_turn"

            return Message(
                id=response.id,
                type="message",
                role="assistant",
                content=content_blocks,
                model=model,
                stop_reason=cast(Any, stop_reason),
                stop_sequence=None,
                usage=usage,
            )
        except ImportError:
            return self._build_mock_message(response, model)

    def _build_mock_message(self, response: MockResponse, model: str) -> MagicMock:
        """Build a MagicMock message when anthropic package not installed."""
        mock = MagicMock()
        mock.id = response.id
        mock.type = "message"
        mock.role = "assistant"
        mock.model = model
        mock.stop_reason = "end_turn"
        mock.stop_sequence = None

        # Content blocks
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = response.content
        mock.content = [text_block]

        # Usage
        usage = MagicMock()
        usage.input_tokens = response.token_usage.prompt_tokens if response.token_usage else 10
        usage.output_tokens = response.token_usage.completion_tokens if response.token_usage else 50
        mock.usage = usage

        return mock

    async def _create_async_message(self, **kwargs: Any) -> Any:
        """Create a mock async message response."""
        self._record_call(type="messages.create", **kwargs)
        model = kwargs.get("model", self._default_model)
        response = self._get_next_response(model=model)
        return self._build_message_response(response, model)

    async def _create_async_streaming_message(self, **kwargs: Any) -> AsyncIterator[Any]:
        """Create an async streaming message response."""
        self._record_call(type="messages.create", stream=True, **kwargs)
        model = kwargs.get("model", self._default_model)
        response = self._get_next_response(model=model)

        # Split content into chunks
        if response.stream_chunks:
            chunks = response.stream_chunks
        else:
            words = response.content.split()
            chunks = [word + " " for word in words[:-1]] + [words[-1]] if words else [""]

        # Emit message_start
        yield self._build_stream_event(
            "message_start",
            {
                "message": {
                    "id": response.id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": model,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": response.token_usage.prompt_tokens
                        if response.token_usage
                        else 10,
                        "output_tokens": 0,
                    },
                }
            },
        )

        # Emit content_block_start
        yield self._build_stream_event(
            "content_block_start",
            {
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        )

        # Emit content_block_delta for each chunk
        for chunk_content in chunks:
            yield self._build_stream_event(
                "content_block_delta",
                {
                    "index": 0,
                    "delta": {"type": "text_delta", "text": chunk_content},
                },
            )

        # Emit content_block_stop
        yield self._build_stream_event("content_block_stop", {"index": 0})

        # Emit message_delta
        yield self._build_stream_event(
            "message_delta",
            {
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {
                    "output_tokens": response.token_usage.completion_tokens
                    if response.token_usage
                    else 50
                },
            },
        )

        # Emit message_stop
        yield self._build_stream_event("message_stop", {})

    def _create_streaming_message(self, **kwargs: Any) -> Iterator[Any]:
        """Create a streaming message response."""
        self._record_call(type="messages.create", stream=True, **kwargs)
        model = kwargs.get("model", self._default_model)
        response = self._get_next_response(model=model)

        # Split content into chunks
        if response.stream_chunks:
            chunks = response.stream_chunks
        else:
            words = response.content.split()
            chunks = [word + " " for word in words[:-1]] + [words[-1]] if words else [""]

        # Emit message_start
        yield self._build_stream_event(
            "message_start",
            {
                "message": {
                    "id": response.id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": model,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": response.token_usage.prompt_tokens
                        if response.token_usage
                        else 10,
                        "output_tokens": 0,
                    },
                }
            },
        )

        # Emit content_block_start
        yield self._build_stream_event(
            "content_block_start",
            {
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        )

        # Emit content_block_delta for each chunk
        for chunk_content in chunks:
            yield self._build_stream_event(
                "content_block_delta",
                {
                    "index": 0,
                    "delta": {"type": "text_delta", "text": chunk_content},
                },
            )

        # Emit content_block_stop
        yield self._build_stream_event("content_block_stop", {"index": 0})

        # Emit message_delta
        yield self._build_stream_event(
            "message_delta",
            {
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {
                    "output_tokens": response.token_usage.completion_tokens
                    if response.token_usage
                    else 50
                },
            },
        )

        # Emit message_stop
        yield self._build_stream_event("message_stop", {})

    def _build_stream_event(self, event_type: str, data: dict[str, Any]) -> Any:
        """Build a streaming event."""
        # Use MagicMock to avoid complex type construction
        mock = MagicMock()
        mock.type = event_type
        for key, value in data.items():
            setattr(mock, key, value)
        return mock

    def __enter__(self) -> AnthropicMock:
        """Start mocking Anthropic API calls."""
        # Sync client
        mock_client = MagicMock()

        def create_message(*args: Any, **kwargs: Any) -> Any:
            if kwargs.get("stream", False):
                return self._create_streaming_message(**kwargs)
            return self._create_message(**kwargs)

        mock_client.messages.create = create_message
        mock_client.beta = MagicMock()
        mock_client.beta.messages = mock_client.messages

        # Async client
        async_mock_client = MagicMock()

        async def create_async_message(*args: Any, **kwargs: Any) -> Any:
            if kwargs.get("stream", False):
                return self._create_async_streaming_message(**kwargs)
            return await self._create_async_message(**kwargs)

        async_mock_client.messages.create = create_async_message
        async_mock_client.beta = MagicMock()
        async_mock_client.beta.messages = async_mock_client.messages

        try:
            patcher = patch("anthropic.Anthropic", return_value=mock_client)
            self._patches.append(patcher)
            patcher.start()

            async_patcher = patch("anthropic.AsyncAnthropic", return_value=async_mock_client)
            self._patches.append(async_patcher)
            async_patcher.start()
        except Exception:
            pass

        self._mock_client = mock_client
        self._async_mock_client = async_mock_client
        return self

    @property
    def client(self) -> MagicMock:
        """Access the mock client directly for advanced usage."""
        return self._mock_client
