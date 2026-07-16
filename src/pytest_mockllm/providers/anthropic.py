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
        self._stream_disconnect_after: int | None = None
        self._stream_disconnect_message: str = "Stream disconnected."

    def simulate_stream_disconnect(
        self, *, after_chunks: int, message: str = "Stream disconnected."
    ) -> AnthropicMock:
        """Disconnect a stream after a deterministic number of content deltas."""
        if after_chunks < 0:
            raise ValueError("after_chunks must be non-negative")
        self._stream_disconnect_after = after_chunks
        self._stream_disconnect_message = message
        return self

    def _raise_provider_error(self, error: MockError) -> None:
        """Raise an Anthropic-style API error."""
        try:
            from anthropic import (
                APIError,
                APITimeoutError,
                AuthenticationError,
                BadRequestError,
                InternalServerError,
                RateLimitError,
            )

            error_classes = {
                "rate_limit": RateLimitError,
                "auth": AuthenticationError,
                "timeout": APITimeoutError,
                "server": APIError,
                "overloaded": InternalServerError,
                "invalid_request": BadRequestError,
            }

            error_class = error_classes.get(error.error_type, APIError)

            import httpx

            status_code = {
                "rate_limit": 429,
                "auth": 401,
                "timeout": 408,
                "server": 500,
                "overloaded": 529,
                "invalid_request": 400,
            }.get(error.error_type, 500)
            mock_response = httpx.Response(
                status_code,
                request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
                json={"error": {"message": error.message}},
            )

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
        self._simulate_delay(self._get_delay_ms(response))
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
        await self._simulate_delay_async(self._get_delay_ms(response))
        return self._build_message_response(response, model)

    def _stream_events(self, response: MockResponse, model: str) -> Iterator[Any]:
        """Build official Anthropic raw stream events."""
        from anthropic.types import (
            ContentBlockDeltaEvent,
            ContentBlockStartEvent,
            ContentBlockStopEvent,
            InputJSONDelta,
            MessageDeltaEvent,
            MessageDeltaUsage,
            MessageStartEvent,
            MessageStopEvent,
            TextBlock,
            TextDelta,
            ToolUseBlock,
            Usage,
        )
        from anthropic.types.raw_message_delta_event import Delta

        usage = Usage(
            input_tokens=response.token_usage.prompt_tokens if response.token_usage else 10,
            output_tokens=0,
        )
        message = self._build_message_response(response, model)
        message.content = []
        message.stop_reason = None
        message.usage = usage
        yield MessageStartEvent(type="message_start", message=message)

        if response.tool_calls and response.tool_call_chunks:
            tool_call = response.tool_calls[0]
            yield ContentBlockStartEvent(
                type="content_block_start",
                index=0,
                content_block=ToolUseBlock(
                    type="tool_use",
                    id=tool_call.get("id", f"toolu_{uuid.uuid4().hex[:8]}"),
                    name=tool_call["function"]["name"],
                    input={},
                ),
            )
            chunks = response.tool_call_chunks
            for index, chunk in enumerate(chunks):
                self._maybe_disconnect_stream(index)
                yield ContentBlockDeltaEvent(
                    type="content_block_delta",
                    index=0,
                    delta=InputJSONDelta(type="input_json_delta", partial_json=chunk),
                )
            stop_reason = "tool_use"
        else:
            yield ContentBlockStartEvent(
                type="content_block_start",
                index=0,
                content_block=TextBlock(type="text", text=""),
            )
            if response.stream_chunks:
                chunks = response.stream_chunks
            else:
                words = response.content.split()
                chunks = [word + " " for word in words[:-1]] + [words[-1]] if words else [""]
            for index, chunk in enumerate(chunks):
                self._maybe_disconnect_stream(index)
                yield ContentBlockDeltaEvent(
                    type="content_block_delta",
                    index=0,
                    delta=TextDelta(type="text_delta", text=chunk),
                )
            stop_reason = "end_turn"

        yield ContentBlockStopEvent(type="content_block_stop", index=0)
        yield MessageDeltaEvent(
            type="message_delta",
            delta=Delta(stop_reason=cast(Any, stop_reason), stop_sequence=None),
            usage=MessageDeltaUsage(
                output_tokens=response.token_usage.completion_tokens
                if response.token_usage
                else 50
            ),
        )
        yield MessageStopEvent(type="message_stop")

    def _maybe_disconnect_stream(self, emitted_chunks: int) -> None:
        """Raise the configured official connection error at the stream boundary."""
        if self._stream_disconnect_after != emitted_chunks:
            return
        if self._calls:
            self._calls[-1]["outcome"] = "stream_disconnect"
        import httpx
        from anthropic import APIConnectionError

        raise APIConnectionError(
            message=self._stream_disconnect_message,
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
        )

    async def _create_async_streaming_message(self, **kwargs: Any) -> AsyncIterator[Any]:
        """Create an async iterator of official streaming events."""
        request = {**kwargs, "stream": True}
        self._record_call(type="messages.create", **request)
        model = kwargs.get("model", self._default_model)
        response = self._get_next_response(model=model)
        for event in self._stream_events(response, model):
            yield event

    def _create_streaming_message(self, **kwargs: Any) -> Iterator[Any]:
        """Create an iterator of official streaming events."""
        request = {**kwargs, "stream": True}
        self._record_call(type="messages.create", **request)
        model = kwargs.get("model", self._default_model)
        response = self._get_next_response(model=model)
        yield from self._stream_events(response, model)

    def __enter__(self) -> AnthropicMock:
        """Patch message methods and block every unhandled Anthropic request."""
        mock_client = MagicMock()

        def create_message(*args: Any, **kwargs: Any) -> Any:
            if kwargs.get("stream", False):
                return self._create_streaming_message(**kwargs)
            return self._create_message(**kwargs)

        mock_client.messages.create = create_message
        mock_client.beta = MagicMock()
        mock_client.beta.messages = mock_client.messages

        async_mock_client = MagicMock()

        async def create_async_message(*args: Any, **kwargs: Any) -> Any:
            if kwargs.get("stream", False):
                return self._create_async_streaming_message(**kwargs)
            return await self._create_async_message(**kwargs)

        async_mock_client.messages.create = create_async_message
        async_mock_client.beta = MagicMock()
        async_mock_client.beta.messages = async_mock_client.messages

        def blocked_request(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("Anthropic network blocked by active mock_anthropic fixture")

        async def blocked_async_request(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("Anthropic network blocked by active mock_anthropic fixture")

        targets = [
            ("anthropic.resources.messages.Messages.create", create_message),
            ("anthropic.resources.messages.AsyncMessages.create", create_async_message),
            ("anthropic._base_client.SyncAPIClient.request", blocked_request),
            ("anthropic._base_client.AsyncAPIClient.request", blocked_async_request),
        ]
        for target, replacement in targets:
            patcher = patch(target, new=replacement)
            self._patches.append(patcher)
            patcher.start()

        self._mock_client = mock_client
        self._async_mock_client = async_mock_client
        return self

    @property
    def client(self) -> MagicMock:
        """Access the mock client directly for advanced usage."""
        return self._mock_client
