"""Acceptance tests for the private Day 1 OpenAI SDK-contract proof."""

from __future__ import annotations

import json

import httpx
import pytest
from openai import OpenAI, RateLimitError
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.responses import ParsedResponse
from pydantic import BaseModel


class Contact(BaseModel):
    name: str
    email: str


class CountingTransport(httpx.BaseTransport):
    """A transport that proves whether an SDK call crossed its HTTP boundary."""

    def __init__(self) -> None:
        self.requests = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.requests += 1
        return httpx.Response(500, json={"error": {"message": "egress occurred"}})


def offline_client() -> tuple[OpenAI, CountingTransport]:
    transport = CountingTransport()
    client = OpenAI(
        api_key="fixture-only",
        base_url="https://provider.invalid/v1",
        http_client=httpx.Client(transport=transport),
    )
    return client, transport


def test_responses_parse_returns_official_typed_structured_output(mock_openai):
    mock_openai.add_response('{"name":"Ada Lovelace","email":"ada@example.test"}')
    client, transport = offline_client()

    response = client.responses.parse(
        model="gpt-4o-mini",
        input="Extract the contact",
        text_format=Contact,
    )

    assert isinstance(response, ParsedResponse)
    assert response.output_text == '{"name":"Ada Lovelace","email":"ada@example.test"}'
    assert response.output_parsed == Contact(name="Ada Lovelace", email="ada@example.test")
    assert mock_openai.request_ledger[0]["type"] == "responses.parse"
    assert mock_openai.request_ledger[0]["input"] == "Extract the contact"
    assert mock_openai.request_ledger[0]["text_format"] is Contact
    assert transport.requests == 0


def test_chat_stream_fragments_official_tool_call_arguments(mock_openai):
    fragments = ['{"city":', '"Chennai",', '"unit":"c"}']
    mock_openai.add_response(
        "",
        tool_calls=[
            {
                "id": "call_weather",
                "function": {
                    "name": "get_weather",
                    "arguments": {"city": "Chennai", "unit": "c"},
                },
            }
        ],
        tool_call_chunks=fragments,
    )
    client, transport = offline_client()

    chunks = list(
        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Weather?"}],
            stream=True,
        )
    )

    assert all(isinstance(chunk, ChatCompletionChunk) for chunk in chunks)
    deltas = [chunk.choices[0].delta.tool_calls[0] for chunk in chunks]
    assert [delta.function.arguments for delta in deltas] == fragments
    assert deltas[0].id == "call_weather"
    assert deltas[0].function.name == "get_weather"
    assert all(delta.id is None for delta in deltas[1:])
    assert all(delta.function.name is None for delta in deltas[1:])
    assert json.loads("".join(delta.function.arguments for delta in deltas)) == {
        "city": "Chennai",
        "unit": "c",
    }
    assert chunks[-1].choices[0].finish_reason == "tool_calls"
    assert transport.requests == 0


def test_one_shot_429_scenario_recovers_and_leaves_request_evidence(mock_openai):
    mock_openai.simulate_error("rate_limit", message="retry now", times=1)
    mock_openai.add_response("recovered")
    client, transport = offline_client()

    with pytest.raises(RateLimitError) as error:
        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Try once"}],
        )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Try twice"}],
    )

    assert error.value.status_code == 429
    assert isinstance(response, ChatCompletion)
    assert response.choices[0].message.content == "recovered"
    assert len(mock_openai.request_ledger) == 2
    assert [entry["outcome"] for entry in mock_openai.request_ledger] == [
        "rate_limit",
        "success",
    ]
    assert [entry["messages"][0]["content"] for entry in mock_openai.request_ledger] == [
        "Try once",
        "Try twice",
    ]
    assert transport.requests == 0


def test_fixture_scoped_clients_are_fail_closed_with_zero_transport_egress(mock_openai):
    client, transport = offline_client()
    mock_openai.add_response("offline")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "No network"}],
    )

    assert response.choices[0].message.content == "offline"
    assert transport.requests == 0


def test_client_created_before_fixture_is_blocked_without_transport_egress(request):
    client, transport = offline_client()
    request.getfixturevalue("mock_openai")

    with pytest.raises(RuntimeError, match="blocked.*mock_openai|mock_openai.*blocked"):
        client.models.list()

    assert transport.requests == 0
