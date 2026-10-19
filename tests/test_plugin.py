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


@pytest.mark.parametrize("flag", ["--llm-record=once", "--llm-record"])
def test_recording_cli_and_offline_default(pytester, flag):
    pytester.makepyfile("""
        import httpx
        from openai import OpenAI
        def test_record(llm_recorder):
            def handle(request):
                assert llm_recorder.is_recording, "replay reached transport"
                return httpx.Response(200, json={
                    "id": "chat_1", "object": "chat.completion", "created": 1,
                    "model": "test", "choices": [{"index": 0, "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "hello"}}]})
            with OpenAI(api_key="fake", max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(handle))) as client:
                response = client.chat.completions.create(model="test", messages=[])
                assert response.choices[0].message.content == "hello"
    """)
    pytester.runpytest("-q", flag).assert_outcomes(passed=1)
    files = list((pytester.path / "tests" / "llm_cassettes").glob("*.yaml"))
    assert len(files) == 1
    before = files[0].read_bytes()
    pytester.runpytest("-q").assert_outcomes(passed=1)
    assert files[0].read_bytes() == before


def test_record_marker_activates_offline_recorder_and_replay_overrides_cli(pytester):
    pytester.makepyfile("""
        import pytest
        from openai import OpenAI
        from pytest_mockllm.recording import CassetteMissError
        @pytest.mark.llm_record
        @pytest.mark.llm_replay
        def test_offline():
            with OpenAI(api_key="fake", max_retries=0) as client:
                with pytest.raises(CassetteMissError):
                    client.chat.completions.create(model="test", messages=[])
    """)
    pytester.runpytest("-q", "--llm-record=all").assert_outcomes(passed=1)
    assert not list(pytester.path.rglob("*.yaml"))


def test_recorder_rejects_mixed_mock_fixtures(pytester):
    pytester.makepyfile("def test_conflict(llm_recorder, mock_openai): pass")
    result = pytester.runpytest("-q")
    result.assert_outcomes(errors=1)
    assert "separately" in result.stdout.str()


def test_cassette_names_include_module_and_parameter_ids(pytester):
    source = """
        import pytest
        @pytest.mark.parametrize("value", ["a/b", "a?b"])
        def test_same(llm_recorder, value):
            llm_recorder.record_interaction({"body": {}}, {"body": value}, "test", "test")
    """
    pytester.makepyfile(test_one=source, test_two=source)
    pytester.runpytest("-q", "--llm-record=once").assert_outcomes(passed=4)
    files = list((pytester.path / "tests" / "llm_cassettes").glob("*.yaml"))
    assert len(files) == 4


@pytest.mark.parametrize(
    "args,ini,expected",
    [
        ([], False, False),
        (["--llm-stats"], False, True),
        ([], True, True),
        (["--no-llm-stats"], True, False),
        (["--llm-stats", "--no-llm-stats"], False, False),
    ],
)
def test_stats_are_opt_in(pytester, args, ini, expected):
    pytester.makeini("[pytest]\nllm_stats = " + str(ini).lower())
    pytester.makepyfile("""
        def test_call(mock_openai):
            mock_openai.add_response("hi")
            mock_openai.client.chat.completions.create(model="test", messages=[])
    """)
    result = pytester.runpytest("-q", *args)
    result.assert_outcomes(passed=1)
    assert ("Total Estimated Cost Saved" in result.stdout.str()) is expected


def test_global_strict_applies_to_langchain(pytester):
    pytester.makepyfile("""
        import pytest
        def test_strict(mock_langchain):
            with pytest.raises(RuntimeError, match="No mock response"):
                mock_langchain.model.invoke("hello")
    """)
    pytester.runpytest("-q", "--llm-strict").assert_outcomes(passed=1)
