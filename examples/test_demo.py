"""Offline tests shown in the launch demo: pytest examples/test_demo.py -v."""

import json

import pytest
from openai import OpenAI, RateLimitError


def test_stream_without_an_api_key(mock_openai):
    mock_openai.add_response("Hello, Ada!", stream_chunks=["Hello, ", "Ada!"])
    with OpenAI(api_key="fixture-only") as client:
        stream = client.chat.completions.create(model="gpt-4o-mini", messages=[], stream=True)
        assert "".join(chunk.choices[0].delta.content or "" for chunk in stream) == "Hello, Ada!"
    assert mock_openai.call_count == 1


def test_weather_tool_arguments(mock_openai):
    mock_openai.add_response(
        "",
        tool_calls=[
            {"function": {"name": "weather", "arguments": {"city": "Paris"}}},
        ],
    )
    with OpenAI(api_key="fixture-only") as client:
        reply = client.chat.completions.create(model="gpt-4o-mini", messages=[])
    tool = reply.choices[0].message.tool_calls[0]
    assert tool.function.name == "weather"
    assert json.loads(tool.function.arguments) == {"city": "Paris"}


def test_recovers_after_rate_limit(mock_openai):
    mock_openai.simulate_error("rate_limit", times=1)
    mock_openai.add_response("Recovered")
    with OpenAI(api_key="fixture-only") as client:
        with pytest.raises(RateLimitError):
            client.chat.completions.create(model="gpt-4o-mini", messages=[])
        reply = client.chat.completions.create(model="gpt-4o-mini", messages=[])
    assert reply.choices[0].message.content == "Recovered"
