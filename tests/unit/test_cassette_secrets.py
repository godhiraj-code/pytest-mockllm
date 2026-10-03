from pytest_mockllm.recording import Cassette, PIIRedactor, RecordedInteraction


def test_extended_key_formats_are_fully_redacted():
    for prefix in ("sk-proj-", "sk-svcacct-", "sk-ant-api03-"):
        key = prefix + "example_secret_value_" * 3
        assert key not in PIIRedactor.redact_text("key=" + key)
        assert "example_secret_value" not in PIIRedactor.redact_text(key)


def test_cassette_redacts_nested_tuples_and_short_credentials(tmp_path):
    request = {"headers": ({"password": "123", "refresh_token": "short"},)}
    cassette = Cassette("test", [RecordedInteraction(request, {}, "test", "test")])
    path = tmp_path / "recording.yaml"
    cassette.save(path)
    saved = path.read_text(encoding="utf-8")
    assert "password: '123'" not in saved
    assert "short" not in saved
    assert request["headers"][0]["password"] == "123"
    assert Cassette.load(path).interactions[0].request["headers"][0]["password"] == "[REDACTED]"
