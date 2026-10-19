"""Regression contracts for the current SDK models and streaming interfaces."""

import asyncio
import importlib
import json

import pytest
from openai import AsyncOpenAI, OpenAI
from openai.types.responses import Response
from pydantic import BaseModel

from pytest_mockllm.core import TokenUsage


class Answer(BaseModel):
    answer: str


def test_responses_text_usage_and_tools(mock_openai):
    mock_openai.add_response("hello", token_usage=TokenUsage(3, 4))
    mock_openai.add_response(
        "",
        tool_calls=[
            {"id": "call_1", "function": {"name": "weather", "arguments": {"city": "Paris"}}}
        ],
    )
    with OpenAI(api_key="fake") as client:
        response = client.responses.create(model="gpt-4o-mini", input="hi")
        assert isinstance(response, Response)
        assert response.output_text == "hello"
        assert response.usage.total_tokens == 7
        tool = client.responses.create(model="gpt-4o-mini", input="weather?").output[0]
        assert tool.type == "function_call"
        assert tool.call_id == "call_1"
        assert json.loads(tool.arguments) == {"city": "Paris"}


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("tool", [False, True])
@pytest.mark.asyncio
async def test_responses_stream_types_and_context_manager(mock_openai, asynchronous, tool):
    if tool:
        mock_openai.add_response(
            "",
            tool_calls=[
                {"id": "call_1", "function": {"name": "weather", "arguments": {"city": "Paris"}}}
            ],
            tool_call_chunks=['{"city":', '"Paris"}'],
        )
    else:
        mock_openai.add_response("hello", stream_chunks=["hel", "lo"])
    client = (AsyncOpenAI if asynchronous else OpenAI)(api_key="fake")
    if asynchronous:
        async with await client.responses.create(
            model="gpt-4o-mini", input="hi", stream=True
        ) as stream:
            events = [event async for event in stream]
        await client.close()
    else:
        with client.responses.create(model="gpt-4o-mini", input="hi", stream=True) as stream:
            events = list(stream)
        client.close()
    assert events[0].type == "response.created"
    assert events[-1].type == "response.completed"
    assert [e.sequence_number for e in events] == list(range(len(events)))
    assert isinstance(events[-1].response, Response)
    if tool:
        fragments = [e.delta for e in events if e.type == "response.function_call_arguments.delta"]
        assert json.loads("".join(fragments)) == {"city": "Paris"}
    else:
        assert "".join(e.delta for e in events if e.type == "response.output_text.delta") == "hello"
        assert events[-1].response.output_text == "hello"


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_responses_stream_helper_final_parsed_output(mock_openai, asynchronous):
    mock_openai.add_response('{"answer":"hello"}', stream_chunks=['{"answer":', '"hello"}'])
    client = (AsyncOpenAI if asynchronous else OpenAI)(api_key="fake")
    manager = client.responses.stream(model="gpt-4o-mini", input="hi", text_format=Answer)
    if asynchronous:
        async with manager as stream:
            response = await stream.get_final_response()
        await client.close()
    else:
        with manager as stream:
            response = stream.get_final_response()
        client.close()
    assert response.output_parsed == Answer(answer="hello")


@pytest.mark.asyncio
async def test_responses_async_delay_yields_event_loop(mock_openai):
    mock_openai.add_response('{"answer":"hello"}', latency_ms=30)
    advanced = False

    async def other_task():
        nonlocal advanced
        await asyncio.sleep(0)
        advanced = True

    task = asyncio.create_task(other_task())
    async with AsyncOpenAI(api_key="fake") as client:
        response = await client.responses.parse(model="gpt-4o-mini", input="hi", text_format=Answer)
        assert advanced
        assert response.output_parsed == Answer(answer="hello")
    await task


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize(
    "kind,expected",
    [
        ("rate_limit", "RateLimitError"),
        ("auth", "AuthenticationError"),
        ("timeout", "APITimeoutError"),
        ("server", "InternalServerError"),
        ("invalid_request", "BadRequestError"),
    ],
)
def test_official_error_constructors(request, provider, kind, expected):
    mock = request.getfixturevalue("mock_" + provider)
    sdk = importlib.import_module(provider)
    mock.simulate_error(kind)
    with getattr(sdk, "OpenAI" if provider == "openai" else "Anthropic")(api_key="fake") as client:
        with pytest.raises(getattr(sdk, expected)) as error:
            if provider == "openai":
                client.chat.completions.create(model="test", messages=[])
            else:
                client.messages.create(model="test", messages=[], max_tokens=1)
        assert error.value.request.method == "POST"


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_anthropic_stream_helper_text_and_tools(mock_anthropic, asynchronous):
    import anthropic

    mock_anthropic.add_response(
        "Looking up weather",
        tool_calls=[
            {"id": "tool_1", "function": {"name": "weather", "arguments": {"city": "Paris"}}}
        ],
    )
    client = (anthropic.AsyncAnthropic if asynchronous else anthropic.Anthropic)(api_key="fake")
    manager = client.messages.stream(model="test", max_tokens=100, messages=[])
    if asynchronous:
        async with manager as stream:
            result = await stream.get_final_message()
        await client.close()
    else:
        with manager as stream:
            result = stream.get_final_message()
        client.close()
    assert result.content[0].text == "Looking up weather"
    assert result.content[1].type == "tool_use"
    assert result.content[1].input == {"city": "Paris"}
    assert result.stop_reason == "tool_use"


def test_chat_stream_multiple_tools_without_manual_fragments(mock_openai):
    mock_openai.add_response(
        "",
        tool_calls=[
            {"function": {"name": name, "arguments": {"city": "Paris"}}}
            for name in ["weather", "time"]
        ],
    )
    with OpenAI(api_key="fake", base_url="https://compatible.invalid/v1") as client:
        with client.chat.completions.create(
            model="another-provider-model", messages=[], stream=True
        ) as stream:
            chunks = list(stream)
    tools = [c.choices[0].delta.tool_calls[0] for c in chunks]
    assert [t.index for t in tools] == [0, 1]
    assert [t.function.name for t in tools] == ["weather", "time"]
    assert all(json.loads(t.function.arguments) == {"city": "Paris"} for t in tools)
    assert chunks[-1].choices[0].finish_reason == "tool_calls"
