from pytest_mockllm.recording import PIIRedactor

def test_redact_openai_key():
    text = "The key is sk-1234567890abcdef1234567890abcdef."
    redacted = PIIRedactor.redact_text(text)
    assert "[REDACTED_OPENAI_KEY]" in redacted
    assert "sk-" not in redacted

def test_redact_bearer_token():
    text = "Authorization: Bearer my-secret-token-1234567890"
    redacted = PIIRedactor.redact_text(text)
    assert "Bearer [REDACTED]" in redacted

def test_redact_azure_key():
    text = "Connecting with api-key: 1234567890abcdef1234567890abcdef"
    redacted = PIIRedactor.redact_text(text)
    assert "api-key: [REDACTED]" in redacted
    assert "1234567890" not in redacted

def test_redact_gcp_key():
    # Using a fake pattern that matches our regex but doesn't look real to GitHub
    text = "Key: AIza_MOCK_GOOGLE_API_KEY_FOR_TESTING_123"
    redacted = PIIRedactor.redact_text(text)
    assert "[REDACTED_GCP_KEY]" in redacted
    assert "AIza" not in redacted

def test_redact_dict():
    data = {
        "api_key": "sk-12345",
        "api-key": "azure-secret",
        "nested": {
            "token": "secret-token",
            "msg": "Hello sk-1234567890abcdef12345"
        },
        "list": ["AIza_MOCK_GOOGLE_API_KEY_FOR_TESTING_123", {"auth": "Bearer token"}]
    }
    redacted = PIIRedactor.redact_dict(data)
    
    assert redacted["api_key"] == "[REDACTED]"
    assert redacted["api-key"] == "[REDACTED]"
    assert redacted["nested"]["token"] == "[REDACTED]"
    assert "[REDACTED_OPENAI_KEY]" in redacted["nested"]["msg"]
    assert "[REDACTED_GCP_KEY]" in redacted["list"][0]
    assert redacted["list"][1]["auth"] == "[REDACTED]"
    # Note: 'auth' is not in SENSITIVE_KEYS, but it might contain a Bearer token
    # Wait, 'auth' is not in my SENSITIVE_KEYS set. Let me check.
    # PIIRedactor.SENSITIVE_KEYS = {"api_key", "api-key", "authorization", "x-api-key", "token", "access_token", "secret"}
