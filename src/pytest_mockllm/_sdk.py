"""Small SDK compatibility boundary for HTTPX and HTTPX2 response types."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any


def http_module(provider: str) -> Any:
    sdk = importlib.import_module(f"{provider}._base_client")
    return getattr(sdk, "httpx2", getattr(sdk, "httpx", None))


def raise_sdk_error(provider: str, error: Any) -> None:
    sdk = importlib.import_module(provider)
    http = http_module(provider)
    request = http.Request("POST", f"https://api.{provider}.com/v1/mock")
    if error.error_type == "timeout":
        raise sdk.APITimeoutError(request=request)
    name, status = {
        "rate_limit": ("RateLimitError", 429),
        "auth": ("AuthenticationError", 401),
        "server": ("InternalServerError", 500),
        "overloaded": ("InternalServerError", 529),
        "invalid_request": ("BadRequestError", 400),
    }.get(error.error_type, ("InternalServerError", 500))
    body = {"error": {"message": error.message}}
    response = http.Response(status, request=request, json=body)
    raise getattr(sdk, name)(message=error.message, response=response, body=body)


def sdk_stream(
    provider: str, events: Any, *, asynchronous: bool = False, cast_to: Any = None
) -> Any:
    """Use the installed SDK's real SSE parser and stream/context-manager classes."""
    sdk = importlib.import_module(provider)
    streaming = importlib.import_module(f"{provider}._streaming")
    http = http_module(provider)
    from pydantic import TypeAdapter

    def process_data(*, data: Any, cast_to: Any, response: Any) -> Any:
        return TypeAdapter(cast_to).validate_python(data)

    client = SimpleNamespace(
        _make_sse_decoder=streaming.SSEDecoder,
        _process_response_data=process_data,
    )

    def encode(event: Any) -> bytes:
        data = event.model_dump_json()
        prefix = f"event: {event.type}\n" if provider == "anthropic" else ""
        return (prefix + "data: " + data + "\n\n").encode()

    sync_base: Any = http.SyncByteStream
    async_base: Any = http.AsyncByteStream

    class SyncBody(sync_base):
        def __iter__(self) -> Any:
            for event in events:
                yield encode(event)
            if provider == "openai":
                yield b"data: [DONE]\n\n"

        def close(self) -> None:
            if hasattr(events, "close"):
                events.close()

    class AsyncBody(async_base):
        async def __aiter__(self) -> Any:
            async for event in events:
                yield encode(event)
            if provider == "openai":
                yield b"data: [DONE]\n\n"

        async def aclose(self) -> None:
            if hasattr(events, "aclose"):
                await events.aclose()

    response = http.Response(
        200,
        headers={"content-type": "text/event-stream"},
        request=http.Request("POST", "https://mock.invalid"),
        stream=AsyncBody() if asynchronous else SyncBody(),
    )
    cls = sdk.AsyncStream if asynchronous else sdk.Stream
    return cls(cast_to=cast_to, response=response, client=client)
