"""Acceptance tests for the private Day 2 Anthropic SDK-contract proof."""

from __future__ import annotations

import json

import httpx
import pytest
from anthropic import (
    Anthropic,
    APIConnectionError,
    AsyncAnthropic,
    InternalServerError,
    RateLimitError,
)
from anthropic.types import (
    ContentBlockDeltaEvent,
    InputJSONDelta,
    Message,
    MessageStartEvent,
    MessageStopEvent,
    TextDelta,
)


class CountingTransport(httpx.BaseTransport, httpx.AsyncBaseTransport):
    """A transport that counts every attempted sync or async HTTP request."""

    def __init__(self) -> None:
        self.requests = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.requests += 1
        return httpx.Response(500, request=request, json={"error": {"message": "egress"}})

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests += 1
        return httpx.Response(500, request=request, json={"error": {"message": "egress"}})


def offline_client() -> tuple[Anthropic, CountingTransport]:
    transport = CountingTransport()
    client = Anthropic(
        api_key="fixture-only",
        base_url="https://provider.invalid",
        http_client=httpx.Client(transport=transport),
    )
    return client, transport


def offline_async_client() -> tuple[AsyncAnthropic, CountingTransport]:
    transport = CountingTransport()
    client = AsyncAnthropic(
        api_key="fixture-only",
        base_url="https://provider.invalid",
        http_client=httpx.AsyncClient(transport=transport),
    )
    return client, transport


def message_request(prompt: str) -> dict[str, object]:
    return {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 128,
        "messages": [{"role": "user", "content": prompt}],
    }


def test_sync_message_is_an_official_anthropic_object(mock_anthropic):
    mock_anthropic.add_response("Hello from Claude")
    client, transport = offline_client()

    response = client.messages.create(**message_request("Hello"))

    assert isinstance(response, Message)
    assert response.content[0].type == "text"
    assert response.content[0].text == "Hello from Claude"
    assert mock_anthropic.request_ledger[0]["outcome"] == "success"
    assert transport.requests == 0


@pytest.mark.asyncio
async def test_async_message_is_an_official_anthropic_object(mock_anthropic):
    mock_anthropic.add_response("Async Claude")
    client, transport = offline_async_client()

    response = await client.messages.create(**message_request("Hello async"))

    assert isinstance(response, Message)
    assert response.content[0].type == "text"
    assert response.content[0].text == "Async Claude"
    assert transport.requests == 0
    await client.close()


def test_text_stream_uses_official_events(mock_anthropic):
    mock_anthropic.add_response("Hello world", stream_chunks=["Hello ", "world"])
    client, transport = offline_client()

    events = list(client.messages.create(**message_request("Stream"), stream=True))

    assert isinstance(events[0], MessageStartEvent)
    deltas = [event for event in events if isinstance(event, ContentBlockDeltaEvent)]
    assert all(isinstance(event.delta, TextDelta) for event in deltas)
    assert "".join(event.delta.text for event in deltas) == "Hello world"
    assert isinstance(events[-1], MessageStopEvent)
    assert transport.requests == 0


@pytest.mark.asyncio
async def test_async_tool_stream_fragments_official_json_deltas(mock_anthropic):
    fragments = ['{"city":', '"Chennai",', '"unit":"c"}']
    mock_anthropic.add_response(
        "",
        tool_calls=[
            {
                "id": "toolu_weather",
                "function": {
                    "name": "get_weather",
                    "arguments": {"city": "Chennai", "unit": "c"},
                },
            }
        ],
        tool_call_chunks=fragments,
    )
    client, transport = offline_async_client()

    stream = await client.messages.create(**message_request("Weather?"), stream=True)
    events = [event async for event in stream]

    deltas = [
        event.delta
        for event in events
        if isinstance(event, ContentBlockDeltaEvent)
        and isinstance(event.delta, InputJSONDelta)
    ]
    assert [delta.partial_json for delta in deltas] == fragments
    assert json.loads("".join(delta.partial_json for delta in deltas)) == {
        "city": "Chennai",
        "unit": "c",
    }
    block_start = events[1]
    assert block_start.content_block.id == "toolu_weather"
    assert block_start.content_block.name == "get_weather"
    assert transport.requests == 0
    await client.close()


def test_one_shot_429_recovers_with_ordered_request_evidence(mock_anthropic):
    mock_anthropic.simulate_error("rate_limit", message="slow down", times=1)
    mock_anthropic.add_response("recovered")
    client, transport = offline_client()

    with pytest.raises(RateLimitError) as error:
        client.messages.create(**message_request("first"))
    response = client.messages.create(**message_request("second"))

    assert error.value.status_code == 429
    assert isinstance(response, Message)
    assert response.content[0].text == "recovered"
    assert [entry["outcome"] for entry in mock_anthropic.request_ledger] == [
        "rate_limit",
        "success",
    ]
    assert [entry["messages"][0]["content"] for entry in mock_anthropic.request_ledger] == [
        "first",
        "second",
    ]
    assert transport.requests == 0


def test_one_shot_529_recovers_with_ordered_request_evidence(mock_anthropic):
    mock_anthropic.simulate_error("overloaded", message="overloaded", times=1)
    mock_anthropic.add_response("capacity restored")
    client, transport = offline_client()

    with pytest.raises(InternalServerError) as error:
        client.messages.create(**message_request("first overload"))
    response = client.messages.create(**message_request("retry overload"))

    assert error.value.status_code == 529
    assert response.content[0].text == "capacity restored"
    assert [entry["outcome"] for entry in mock_anthropic.request_ledger] == [
        "overloaded",
        "success",
    ]
    assert transport.requests == 0


def test_mid_stream_disconnect_is_controllable_and_recorded(mock_anthropic):
    mock_anthropic.add_response("one two three", stream_chunks=["one ", "two ", "three"])
    mock_anthropic.simulate_stream_disconnect(after_chunks=1, message="socket dropped")
    client, transport = offline_client()

    stream = client.messages.create(**message_request("disconnect"), stream=True)
    iterator = iter(stream)
    assert isinstance(next(iterator), MessageStartEvent)
    next(iterator)  # content_block_start
    first_delta = next(iterator)
    assert first_delta.delta.text == "one "
    with pytest.raises(APIConnectionError, match="socket dropped"):
        next(iterator)

    assert mock_anthropic.request_ledger[0]["outcome"] == "stream_disconnect"
    assert transport.requests == 0


def test_fixture_scoped_supported_call_has_zero_transport_egress(mock_anthropic):
    mock_anthropic.add_response("offline")
    client, transport = offline_client()

    response = client.messages.create(**message_request("No network"))

    assert response.content[0].text == "offline"
    assert transport.requests == 0


def test_client_created_before_fixture_is_blocked_without_transport_egress(request):
    client, transport = offline_client()
    request.getfixturevalue("mock_anthropic")

    with pytest.raises(RuntimeError, match="blocked.*mock_anthropic|mock_anthropic.*blocked"):
        client.models.list()

    assert transport.requests == 0
