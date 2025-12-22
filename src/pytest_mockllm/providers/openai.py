"""
OpenAI API mock for pytest-mockllm.

Provides comprehensive mocking for the OpenAI Python SDK including:
- Chat Completions (sync, async, streaming)
- Embeddings
- Images
- Function/Tool calling
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock, patch

from pytest_mockllm.core import MockError, MockLLM, MockResponse


class OpenAIMock(MockLLM):
    """
    Mock for OpenAI Python SDK.

    Supports:
        - client.chat.completions.create()
        - client.embeddings.create()
        - Streaming responses
        - Function/tool calling
        - Async variants

    Example:
        >>> def test_chatbot(mock_openai):
        ...     mock_openai.add_response("Hello! How can I help?")
        ...
        ...     client = OpenAI()
        ...     response = client.chat.completions.create(
        ...         model="gpt-4o",
        ...         messages=[{"role": "user", "content": "Hi!"}]
        ...     )
        ...     assert "help" in response.choices[0].message.content.lower()
    """

    def __init__(self) -> None:
        super().__init__()
        self._embedding_dimension: int = 1536
        self._default_model: str = "gpt-4o"

    def set_embedding_dimension(self, dim: int) -> OpenAIMock:
        """Set the dimension for mock embeddings (default: 1536)."""
        self._embedding_dimension = dim
        return self

    def _raise_provider_error(self, error: MockError) -> None:
        """Raise an OpenAI-style API error."""
        try:
            from openai import (
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

            # Create mock response for error
            mock_response = MagicMock()
            mock_response.status_code = {
                "rate_limit": 429,
                "auth": 401,
                "timeout": 408,
                "server": 500,
                "invalid_request": 400,
            }.get(error.error_type, 500)
            mock_response.headers = {}
            mock_response.json.return_value = {"error": {"message": error.message}}

            raise error_class(
                message=error.message,
                response=mock_response,
                body={"error": {"message": error.message}},
            )
        except ImportError:
            # If openai not installed, fall back to RuntimeError
            raise RuntimeError(f"OpenAI API Error ({error.error_type}): {error.message}") from None

    def _create_chat_completion(self, **kwargs: Any) -> Any:
        """Create a mock chat completion response."""
        self._record_call(type="chat.completions.create", **kwargs)
        response = self._get_next_response()

        model = kwargs.get("model", self._default_model)

        # Build the response object
        return self._build_completion_response(response, model)

    def _build_completion_response(self, response: MockResponse, model: str) -> Any:
        """Build an OpenAI-style ChatCompletion response object."""
        try:
            from openai.types.chat import ChatCompletion, ChatCompletionMessage
            from openai.types.chat.chat_completion import Choice
            from openai.types.completion_usage import CompletionUsage

            message = ChatCompletionMessage(
                role="assistant",
                content=response.content,
                tool_calls=self._format_tool_calls(response.tool_calls) if response.tool_calls else None,
                function_call=response.function_call,
            )

            choice = Choice(
                index=0,
                message=message,
                finish_reason=response.finish_reason,
            )

            usage = None
            if response.token_usage:
                usage = CompletionUsage(
                    prompt_tokens=response.token_usage.prompt_tokens,
                    completion_tokens=response.token_usage.completion_tokens,
                    total_tokens=response.token_usage.total_tokens,
                )

            return ChatCompletion(
                id=response.id,
                choices=[choice],
                created=int(time.time()),
                model=model,
                object="chat.completion",
                usage=usage,
            )
        except ImportError:
            # Return a MagicMock that behaves like ChatCompletion
            return self._build_mock_completion(response, model)

    def _build_mock_completion(self, response: MockResponse, model: str) -> MagicMock:
        """Build a MagicMock completion when openai package not installed."""
        mock = MagicMock()
        mock.id = response.id
        mock.model = model
        mock.object = "chat.completion"
        mock.created = int(time.time())

        # Message
        message = MagicMock()
        message.role = "assistant"
        message.content = response.content
        message.tool_calls = response.tool_calls
        message.function_call = response.function_call

        # Choice
        choice = MagicMock()
        choice.index = 0
        choice.message = message
        choice.finish_reason = response.finish_reason

        mock.choices = [choice]

        # Usage
        if response.token_usage:
            usage = MagicMock()
            usage.prompt_tokens = response.token_usage.prompt_tokens
            usage.completion_tokens = response.token_usage.completion_tokens
            usage.total_tokens = response.token_usage.total_tokens
            mock.usage = usage
        else:
            mock.usage = None

        return mock

    def _format_tool_calls(self, tool_calls: list[dict[str, Any]]) -> list[Any]:
        """Format tool calls for OpenAI response."""
        try:
            from openai.types.chat.chat_completion_message_tool_call import (
                ChatCompletionMessageToolCall,
                Function,
            )

            return [
                ChatCompletionMessageToolCall(
                    id=tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                    type="function",
                    function=Function(
                        name=tc["function"]["name"],
                        arguments=json.dumps(tc["function"].get("arguments", {})),
                    ),
                )
                for tc in tool_calls
            ]
        except ImportError:
            return tool_calls

    def _create_streaming_completion(self, **kwargs: Any) -> Iterator[Any]:
        """Create a streaming chat completion response."""
        self._record_call(type="chat.completions.create", stream=True, **kwargs)
        response = self._get_next_response()
        model = kwargs.get("model", self._default_model)

        # Split content into chunks
        if response.stream_chunks:
            chunks = response.stream_chunks
        else:
            # Default: split by words
            words = response.content.split()
            chunks = [word + " " for word in words[:-1]] + [words[-1]] if words else [""]

        for i, chunk_content in enumerate(chunks):
            yield self._build_stream_chunk(
                chunk_content,
                model=model,
                is_first=(i == 0),
                is_last=(i == len(chunks) - 1),
                response_id=response.id,
            )

    def _build_stream_chunk(
        self,
        content: str,
        model: str,
        is_first: bool,
        is_last: bool,
        response_id: str,
    ) -> Any:
        """Build a streaming chunk response."""
        try:
            from openai.types.chat import ChatCompletionChunk
            from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta

            delta = ChoiceDelta(
                role="assistant" if is_first else None,
                content=content,
            )

            choice = Choice(
                index=0,
                delta=delta,
                finish_reason="stop" if is_last else None,
            )

            return ChatCompletionChunk(
                id=response_id,
                choices=[choice],
                created=int(time.time()),
                model=model,
                object="chat.completion.chunk",
            )
        except ImportError:
            # MagicMock fallback
            mock = MagicMock()
            mock.id = response_id
            mock.model = model
            mock.object = "chat.completion.chunk"

            delta = MagicMock()
            delta.role = "assistant" if is_first else None
            delta.content = content

            choice = MagicMock()
            choice.index = 0
            choice.delta = delta
            choice.finish_reason = "stop" if is_last else None

            mock.choices = [choice]
            return mock

    def _create_embedding(self, **kwargs: Any) -> Any:
        """Create a mock embedding response."""
        self._record_call(type="embeddings.create", **kwargs)

        # Generate deterministic fake embedding
        input_text = kwargs.get("input", "")
        if isinstance(input_text, list):
            input_text = input_text[0] if input_text else ""

        # Create reproducible embedding based on input hash
        import hashlib
        hash_bytes = hashlib.sha256(input_text.encode()).digest()
        embedding = [
            (b / 255.0 - 0.5) * 2  # Normalize to [-1, 1]
            for b in hash_bytes * (self._embedding_dimension // len(hash_bytes) + 1)
        ][:self._embedding_dimension]

        try:
            from openai.types import CreateEmbeddingResponse, Embedding
            from openai.types.create_embedding_response import Usage

            return CreateEmbeddingResponse(
                data=[Embedding(embedding=embedding, index=0, object="embedding")],
                model=kwargs.get("model", "text-embedding-ada-002"),
                object="list",
                usage=Usage(prompt_tokens=len(input_text.split()), total_tokens=len(input_text.split())),
            )
        except ImportError:
            mock = MagicMock()
            mock.data = [MagicMock(embedding=embedding, index=0)]
            mock.model = kwargs.get("model", "text-embedding-ada-002")
            return mock

    def __enter__(self) -> OpenAIMock:
        """Start mocking OpenAI API calls."""
        # Create the mock client
        mock_client = MagicMock()

        # Set up chat completions
        def create_chat(*args: Any, **kwargs: Any) -> Any:
            if kwargs.get("stream", False):
                return self._create_streaming_completion(**kwargs)
            return self._create_chat_completion(**kwargs)

        mock_client.chat.completions.create = create_chat

        # Set up embeddings
        mock_client.embeddings.create = lambda **kw: self._create_embedding(**kw)

        # Patch the OpenAI client
        try:
            patcher = patch("openai.OpenAI", return_value=mock_client)
            self._patches.append(patcher)
            patcher.start()

            # Also patch AsyncOpenAI
            async_patcher = patch("openai.AsyncOpenAI", return_value=mock_client)
            self._patches.append(async_patcher)
            async_patcher.start()
        except Exception:
            # openai not installed - that's fine, we'll use the mock anyway
            pass

        self._mock_client = mock_client
        return self

    @property
    def client(self) -> MagicMock:
        """Access the mock client directly for advanced usage."""
        return self._mock_client
