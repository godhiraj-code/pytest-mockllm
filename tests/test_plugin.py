"""
Tests that verify the pytest plugin registration and fixtures work correctly.
These tests use pytester to test the pytest integration itself.
"""

from importlib.metadata import requires

import pytest


def test_fixture_discovery(testdir):
    """Verify that fixtures are discovered automatically by pytest."""
    testdir.makepyfile(
        """
        def test_mock_openai_available(mock_openai):
            assert mock_openai is not None
            
        def test_mock_anthropic_available(mock_anthropic):
            assert mock_anthropic is not None
            
        def test_mock_gemini_available(mock_gemini):
            assert mock_gemini is not None
            
        def test_mock_llm_available(mock_llm):
            assert mock_llm is not None
    """
    )
    
    result = testdir.runpytest("-v")
    result.assert_outcomes(passed=4)


def test_mock_openai_fixture_works(testdir):
    """Verify the mock_openai fixture mocks responses correctly."""
    testdir.makepyfile(
        """
        def test_openai_mock(mock_openai):
            mock_openai.add_response("Hello from the mock!")
            
            response = mock_openai._get_next_response()
            assert response.content == "Hello from the mock!"
            assert mock_openai.call_count == 1
    """
    )
    
    result = testdir.runpytest("-v")
    result.assert_outcomes(passed=1)


def test_multiple_responses_work(testdir):
    """Verify multiple responses are returned in order."""
    testdir.makepyfile(
        """
        def test_multiple_responses(mock_openai):
            mock_openai.add_responses(
                "First",
                "Second",
                "Third",
            )
            
            assert mock_openai._get_next_response().content == "First"
            assert mock_openai._get_next_response().content == "Second"
            assert mock_openai._get_next_response().content == "Third"
    """
    )
    
    result = testdir.runpytest("-v")
    result.assert_outcomes(passed=1)


def test_strict_mode_works(testdir):
    """Verify strict mode raises error when no response configured."""
    testdir.makepyfile(
        """
        import pytest
        
        def test_strict_mode(mock_openai):
            mock_openai.set_strict_mode(True)
            
            with pytest.raises(RuntimeError, match="No mock response"):
                mock_openai._get_next_response()
    """
    )
    
    result = testdir.runpytest("-v")
    result.assert_outcomes(passed=1)


def test_cli_options_registered(testdir):
    """Verify CLI options are available."""
    result = testdir.runpytest("--help")
    
    assert "--llm-record" in result.stdout.str()
    assert "--llm-cassette-dir" in result.stdout.str()
    assert "--llm-strict" in result.stdout.str()


def test_cli_strict_mode_applies_to_provider_fixture(testdir):
    """The global strict option must change fixture behavior, not only appear in help."""
    testdir.makepyfile(
        """
        import pytest

        def test_unconfigured_call_fails(mock_openai):
            with pytest.raises(RuntimeError, match="No mock response configured"):
                mock_openai.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": "hello"}],
                )
        """
    )

    result = testdir.runpytest("--llm-strict", "-q")
    result.assert_outcomes(passed=1)


def test_markers_registered(testdir):
    """Verify markers are registered."""
    result = testdir.runpytest("--markers")
    
    assert "llm_mock" in result.stdout.str()
    assert "llm_record" in result.stdout.str()
    assert "llm_replay" in result.stdout.str()


def test_langchain_extra_installs_documented_provider_package():
    """The langchain extra must make the documented ChatOpenAI import available."""
    requirements = requires("pytest-mockllm") or []

    assert any(
        requirement.startswith("langchain-openai") and "extra == 'langchain'" in requirement
        for requirement in requirements
    )
