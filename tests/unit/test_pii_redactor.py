from urllib.parse import parse_qs, urlsplit

import pytest

from pytest_mockllm.recording import PIIRedactor


@pytest.mark.parametrize(("prefix", "label"), [
    ("sk-", "OPENAI"),
    ("sk-proj-", "OPENAI"),
    ("sk-ant-api03-", "ANTHROPIC"),
    ("sk-ant-", "ANTHROPIC"),
    ("ant-api-", "ANTHROPIC"),
    ("AIza", "GCP"),
])
def test_known_key_patterns_and_labels(prefix, label):
    key = prefix + "a" * 35
    assert PIIRedactor.redact_text("Key: " + key) == f"Key: [REDACTED_{label}_KEY]"


def test_redact_bearer_token():
    assert PIIRedactor.redact_text("Bearer credential") == "Bearer [REDACTED]"


def test_redact_azure_key():
    assert PIIRedactor.redact_text("api-key: " + "a" * 32) == "api-key: [REDACTED]"


def test_json_field_names_do_not_identify_secrets():
    data = {
        "properties": {name: {"type": "string"} for name in [
            "key", "token", "auth", "api_key", "authorization", "secret", "password",
        ]},
        "nested": [{"headers": {"Authorization": "a schema value"}}],
        "token": "a word in a sentence",
        "auth": "authentication method",
    }
    assert PIIRedactor.redact_dict(data) == data
    assert PIIRedactor.redact_http({"body": data})["body"] == data


def test_secret_values_are_redacted_regardless_of_json_field_name():
    data = {"nested": [{"ordinary": "sk-ant-" + "a" * 35}], "auth": "Bearer credential"}
    assert PIIRedactor.redact_dict(data) == {
        "nested": [{"ordinary": "[REDACTED_ANTHROPIC_KEY]"}], "auth": "Bearer [REDACTED]",
    }


@pytest.mark.parametrize("header", [
    "Authorization", "X-Api-Key", "api-key", "api_key", "X-Goog-Api-Key",
    "Proxy-Authorization", "Cookie", "Set-Cookie",
])
def test_credentials_are_redacted_in_header_context(header):
    assert PIIRedactor.redact_headers({header: "unrecognizable-credential", "X-Request-ID": "req_1"}) == {
        header: "[REDACTED]", "X-Request-ID": "req_1",
    }


def test_google_query_key_is_redacted_without_touching_other_parameters():
    url = PIIRedactor.redact_url(
        "https://generativelanguage.googleapis.com/v1beta/models?key=opaque-credential&alt=sse"
    )
    assert parse_qs(urlsplit(url).query) == {"key": ["[REDACTED]"], "alt": ["sse"]}
