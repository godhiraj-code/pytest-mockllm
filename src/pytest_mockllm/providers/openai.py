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
from collections.abc import AsyncIterator, Iterator
from typing import Any, cast
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

            import httpx

            status_code = {
                "rate_limit": 429,
                "auth": 401,
                "timeout": 408,
                "server": 500,
                "invalid_request": 400,
            }.get(error.error_type, 500)
            mock_response = httpx.Response(
                status_code,
                request=httpx.Request("POST", "https://api.openai.com/v1/mock"),
                json={"error": {"message": error.message}},
            )

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
        model = kwargs.get("model", self._default_model)
        response = self._get_next_response(model=model)
        self._simulate_delay(self._get_delay_ms(response))

        model = kwargs.get("model", self._default_model)

        # Build the response object
        return self._build_completion_response(response, model)

    def _create_response(self, *, text_format: type[Any] | None = None, **kwargs: Any) -> Any:
        """Create an official Responses API object, optionally with parsed output."""
        call_type = "responses.parse" if text_format is not None else "responses.create"
        self._record_call(type=call_type, text_format=text_format, **kwargs)
        model = kwargs.get("model", self._default_model)
        response = self._get_next_response(model=model)
        self._simulate_delay(self._get_delay_ms(response))

        from openai.types.responses import (
            ParsedResponse,
            ParsedResponseOutputMessage,
            ParsedResponseOutputText,
            Response,
            ResponseOutputMessage,
            ResponseOutputText,
        )

        common = {
            "id": response.id,
            "created_at": time.time(),
            "model": model,
            "object": "response",
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "status": "completed",
        }
        if text_format is not None:
            from pydantic import TypeAdapter

            parsed = TypeAdapter(text_format).validate_json(response.content)
            parsed_output_text = ParsedResponseOutputText[Any](
                annotations=[], type="output_text", text=response.content, parsed=parsed
            )
            parsed_output_message = ParsedResponseOutputMessage[Any](
                id=f"msg_{response.id}",
                content=[parsed_output_text],
                role="assistant",
                status="completed",
                type="message",
            )
            return ParsedResponse[Any](output=[parsed_output_message], **common)

        output_text = ResponseOutputText(
            annotations=[], type="output_text", text=response.content
        )
        output_message = ResponseOutputMessage(
            id=f"msg_{response.id}",
            content=[output_text],
            role="assistant",
            status="completed",
            type="message",
        )
        return Response(output=[output_message], **common)

    def _build_completion_response(self, response: MockResponse, model: str) -> Any:
        """Build an OpenAI-style ChatCompletion response object."""
        try:
            from openai.types.chat import ChatCompletion, ChatCompletionMessage
            from openai.types.chat.chat_completion import Choice
            from openai.types.completion_usage import CompletionUsage

            message = ChatCompletionMessage(
                role="assistant",
                content=response.content,
                tool_calls=self._format_tool_calls(response.tool_calls)
                if response.tool_calls
                else None,
                function_call=cast(Any, response.function_call),
            )

            choice = Choice(
                index=0,
                message=message,
                finish_reason=cast(Any, response.finish_reason),
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

    async def _create_async_chat_completion(self, **kwargs: Any) -> Any:
        """Create a mock async chat completion response."""
        self._record_call(type="chat.completions.create", **kwargs)
        model = kwargs.get("model", self._default_model)
        response = self._get_next_response(model=model)
        await self._simulate_delay_async(self._get_delay_ms(response))
        return self._build_completion_response(response, model)

    async def _create_async_streaming_completion(self, **kwargs: Any) -> AsyncIterator[Any]:
        """Create an async streaming chat completion response."""
        request = {**kwargs, "stream": True}
        self._record_call(type="chat.completions.create", **request)
        model = kwargs.get("model", self._default_model)
        response = self._get_next_response(model=model)

        if response.tool_calls and response.tool_call_chunks:
            for i, arguments in enumerate(response.tool_call_chunks):
                yield self._build_tool_stream_chunk(
                    response=response,
                    model=model,
                    arguments=arguments,
                    is_first=i == 0,
                    is_last=i == len(response.tool_call_chunks) - 1,
                )
            return

        if response.stream_chunks:
            chunks = response.stream_chunks
        else:
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

    def _create_streaming_completion(self, **kwargs: Any) -> Iterator[Any]:
        """Create a streaming chat completion response."""
        request = {**kwargs, "stream": True}
        self._record_call(type="chat.completions.create", **request)
        model = kwargs.get("model", self._default_model)
        response = self._get_next_response(model=model)

        if response.tool_calls and response.tool_call_chunks:
            for i, arguments in enumerate(response.tool_call_chunks):
                yield self._build_tool_stream_chunk(
                    response=response,
                    model=model,
                    arguments=arguments,
                    is_first=i == 0,
                    is_last=i == len(response.tool_call_chunks) - 1,
                )
            return

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

    def _build_tool_stream_chunk(
        self,
        *,
        response: MockResponse,
        model: str,
        arguments: str,
        is_first: bool,
        is_last: bool,
    ) -> Any:
        """Build an official chunk containing one function-argument fragment."""
        from openai.types.chat import ChatCompletionChunk
        from openai.types.chat.chat_completion_chunk import (
            Choice,
            ChoiceDelta,
            ChoiceDeltaToolCall,
            ChoiceDeltaToolCallFunction,
        )

        if not response.tool_calls:
            raise ValueError("A tool call is required to build a tool stream chunk")
        tool_call = response.tool_calls[0]
        delta_tool_call = ChoiceDeltaToolCall(
            index=0,
            id=tool_call.get("id") if is_first else None,
            type="function" if is_first else None,
            function=ChoiceDeltaToolCallFunction(
                name=tool_call["function"]["name"] if is_first else None,
                arguments=arguments,
            ),
        )
        return ChatCompletionChunk(
            id=response.id,
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(tool_calls=[delta_tool_call]),
                    finish_reason="tool_calls" if is_last else None,
                )
            ],
            created=int(time.time()),
            model=model,
            object="chat.completion.chunk",
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
        ][: self._embedding_dimension]

        try:
            from openai.types import CreateEmbeddingResponse, Embedding
            from openai.types.create_embedding_response import Usage

            return CreateEmbeddingResponse(
                data=[Embedding(embedding=embedding, index=0, object="embedding")],
                model=kwargs.get("model", "text-embedding-ada-002"),
                object="list",
                usage=Usage(
                    prompt_tokens=len(input_text.split()), total_tokens=len(input_text.split())
                ),
            )
        except ImportError:
            mock = MagicMock()
            mock.data = [MagicMock(embedding=embedding, index=0)]
            mock.model = kwargs.get("model", "text-embedding-ada-002")
            return mock

    def __enter__(self) -> OpenAIMock:
        """Patch SDK resource methods and block every unhandled OpenAI request."""
        mock_client = MagicMock()

        def create_chat(*args: Any, **kwargs: Any) -> Any:
            if kwargs.get("stream", False):
                return self._create_streaming_completion(**kwargs)
            return self._create_chat_completion(**kwargs)

        def create_response(*args: Any, **kwargs: Any) -> Any:
            return self._create_response(**kwargs)

        def parse_response(*args: Any, **kwargs: Any) -> Any:
            text_format = kwargs.pop("text_format")
            return self._create_response(text_format=text_format, **kwargs)

        async def create_async_chat(*args: Any, **kwargs: Any) -> Any:
            if kwargs.get("stream", False):
                return self._create_async_streaming_completion(**kwargs)
            return await self._create_async_chat_completion(**kwargs)

        async def create_async_response(*args: Any, **kwargs: Any) -> Any:
            return self._create_response(**kwargs)

        async def parse_async_response(*args: Any, **kwargs: Any) -> Any:
            text_format = kwargs.pop("text_format")
            return self._create_response(text_format=text_format, **kwargs)

        async def create_async_embedding(*args: Any, **kwargs: Any) -> Any:
            return self._create_embedding(**kwargs)

        def blocked_request(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("OpenAI network blocked by active mock_openai fixture")

        async def blocked_async_request(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("OpenAI network blocked by active mock_openai fixture")

        targets = [
            ("openai.resources.chat.completions.Completions.create", create_chat),
            ("openai.resources.chat.completions.AsyncCompletions.create", create_async_chat),
            ("openai.resources.responses.Responses.create", create_response),
            ("openai.resources.responses.Responses.parse", parse_response),
            ("openai.resources.responses.AsyncResponses.create", create_async_response),
            ("openai.resources.responses.AsyncResponses.parse", parse_async_response),
            ("openai.resources.embeddings.Embeddings.create", self._create_embedding),
            ("openai.resources.embeddings.AsyncEmbeddings.create", create_async_embedding),
            ("openai._base_client.SyncAPIClient.request", blocked_request),
            ("openai._base_client.AsyncAPIClient.request", blocked_async_request),
        ]
        for target, replacement in targets:
            patcher = patch(target, new=replacement)
            self._patches.append(patcher)
            patcher.start()

        mock_client.chat.completions.create = create_chat
        mock_client.responses.create = create_response
        mock_client.responses.parse = parse_response
        mock_client.embeddings.create = self._create_embedding

        async_mock_client = MagicMock()
        async_mock_client.chat.completions.create = create_async_chat
        async_mock_client.responses.create = create_async_response
        async_mock_client.responses.parse = parse_async_response
        async_mock_client.embeddings.create = create_async_embedding

        self._mock_client = mock_client
        self._async_mock_client = async_mock_client
        return self

    @property
    def client(self) -> MagicMock:
        """Access the mock client directly for advanced usage."""
        return self._mock_client
