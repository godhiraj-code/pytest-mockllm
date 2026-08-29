"""Gemini provider integration contracts."""

import google.generativeai as genai


def test_module_imported_before_fixture_is_intercepted(mock_gemini):
    """A normal module-level SDK import must not bypass the active fixture."""
    mock_gemini.add_response("offline")

    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content("Tell me about AI")

    assert response.text == "offline"
    assert mock_gemini.call_count == 1
