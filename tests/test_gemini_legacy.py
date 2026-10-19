"""Gemini provider integration contracts."""

import pytest

genai = pytest.importorskip("google.generativeai")


def test_module_imported_before_fixture_is_intercepted(mock_gemini):
    """A normal module-level SDK import must not bypass the active fixture."""
    mock_gemini.add_response("offline")

    with pytest.warns(DeprecationWarning, match="google.generativeai support"):
        model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content("Tell me about AI")

    assert response.text == "offline"
    assert mock_gemini.call_count == 1


def test_legacy_stream_and_provider_error(mock_gemini):
    from google.api_core.exceptions import ResourceExhausted

    mock_gemini.add_response("hello world")
    with pytest.warns(DeprecationWarning):
        model = genai.GenerativeModel("gemini-1.5-pro")
    assert "".join(c.text for c in model.generate_content("hi", stream=True)) == "hello world"
    assert mock_gemini.last_call["contents"] == "hi"
    mock_gemini.simulate_error("rate_limit")
    with pytest.raises(ResourceExhausted):
        model.generate_content("hi")


@pytest.mark.asyncio
async def test_legacy_async_generation(mock_gemini):
    mock_gemini.add_responses("hello", "streamed answer")
    with pytest.warns(DeprecationWarning):
        model = genai.GenerativeModel("gemini-1.5-pro")
    assert (await model.generate_content_async("hi")).text == "hello"
    stream = await model.generate_content_async("hi", stream=True)
    assert "".join([c.text async for c in stream]) == "streamed answer"
