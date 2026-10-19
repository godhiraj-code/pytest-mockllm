"""google-genai contracts using real clients, messages, streams and error types."""

import pytest
from google import genai
from google.genai import errors, types
from pydantic import BaseModel

from pytest_mockllm.core import TokenUsage


def test_client_created_before_fixture_is_intercepted(request):
    with genai.Client(api_key="fixture-only") as client:
        mock = request.getfixturevalue("mock_gemini")
        mock.add_response("hello", token_usage=TokenUsage(3, 5))
        response = client.models.generate_content(model="gemini-2.5-flash", contents="hi")
        assert isinstance(response, types.GenerateContentResponse)
        assert response.text == "hello"
        assert response.model_version == "gemini-2.5-flash"
        assert response.usage_metadata.total_token_count == 8
        assert mock.last_call["contents"] == "hi"
        assert mock.last_call["model"] == "gemini-2.5-flash"
        with pytest.raises(RuntimeError, match="Gemini network blocked"):
            client.models.list()


def test_sync_stream_and_chat_history(mock_gemini):
    mock_gemini.add_response("hello", stream_chunks=["hel", "lo"])
    mock_gemini.add_responses("first", "second")
    with genai.Client(api_key="fixture-only") as client:
        stream = list(
            client.models.generate_content_stream(model="gemini-2.5-flash", contents="hi")
        )
        assert all(isinstance(c, types.GenerateContentResponse) for c in stream)
        assert [c.text for c in stream] == ["hel", "lo"]
        assert stream[0].candidates[0].finish_reason is None
        assert stream[-1].candidates[0].finish_reason == "STOP"
        chat = client.chats.create(model="gemini-2.5-flash")
        assert chat.send_message("one").text == "first"
        assert chat.send_message("two").text == "second"
        assert len(chat.get_history()) == 4
        assert mock_gemini.call_count == 3


@pytest.mark.asyncio
async def test_async_generation_stream_and_chat(mock_gemini):
    mock_gemini.add_responses("hello", "streamed response", "chat reply")
    async with genai.Client(api_key="fixture-only").aio as client:
        assert (
            await client.models.generate_content(model="gemini-2.5-flash", contents="hi")
        ).text == "hello"
        stream = await client.models.generate_content_stream(
            model="gemini-2.5-flash", contents="hi"
        )
        chunks = [chunk async for chunk in stream]
        assert all(isinstance(c, types.GenerateContentResponse) for c in chunks)
        assert "".join(c.text for c in chunks) == "streamed response"
        chat = client.chats.create(model="gemini-2.5-flash")
        assert (await chat.send_message("hi")).text == "chat reply"
        with pytest.raises(RuntimeError, match="Gemini network blocked"):
            await client.models.get(model="unsupported")


@pytest.mark.parametrize(
    "kind,code",
    [
        ("rate_limit", 429),
        ("auth", 401),
        ("timeout", 504),
        ("server", 500),
        ("invalid_request", 400),
    ],
)
def test_error_types_strict_and_recovery(mock_gemini, kind, code):
    mock_gemini.simulate_error(kind, times=1)
    mock_gemini.add_response("recovered")
    with genai.Client(api_key="fixture-only") as client:
        with pytest.raises(errors.APIError) as error:
            client.models.generate_content(model="gemini-2.5-flash", contents="hi")
        assert error.value.code == code
        assert (
            client.models.generate_content(model="gemini-2.5-flash", contents="hi").text
            == "recovered"
        )
        mock_gemini.set_strict_mode()
        with pytest.raises(RuntimeError, match="No mock response"):
            client.models.generate_content(model="gemini-2.5-flash", contents="hi")


def test_function_call_and_structured_output(mock_gemini):
    class Answer(BaseModel):
        answer: int

    mock_gemini.add_response(
        "", tool_calls=[{"function": {"name": "weather", "arguments": {"city": "Paris"}}}]
    )
    mock_gemini.add_response('{"answer":42}')
    with genai.Client(api_key="fixture-only") as client:
        response = client.models.generate_content(model="gemini-2.5-flash", contents="weather?")
        assert response.function_calls[0].name == "weather"
        assert response.function_calls[0].args == {"city": "Paris"}
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="answer?",
            config=types.GenerateContentConfig(
                response_mime_type="application/json", response_schema=Answer
            ),
        )
        assert response.parsed == Answer(answer=42)


def test_chat_stream_uses_real_chat_session(mock_gemini):
    mock_gemini.add_response("hello", stream_chunks=["hel", "lo"])
    with genai.Client(api_key="fixture-only") as client:
        chat = client.chats.create(model="gemini-2.5-flash")
        assert "".join(c.text for c in chat.send_message_stream("hi")) == "hello"
        history = chat.get_history()
        assert history[0].role == "user"
        assert "".join(part.text for content in history[1:] for part in content.parts) == "hello"
