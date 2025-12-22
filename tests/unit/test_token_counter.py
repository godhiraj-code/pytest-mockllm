from pytest_mockllm.core import TokenCounter
import tiktoken

def test_token_counter_tiktoken():
    text = "Hello, world! This is a test."
    model = "gpt-4o"
    
    # Expected count using tiktoken directly
    enc = tiktoken.encoding_for_model(model)
    expected = len(enc.encode(text))
    
    # Counter count
    actual = TokenCounter.count_tokens(text, model)
    
    print(f"DEBUG: Expected {expected}, Actual {actual}")
    assert actual == expected
    assert actual > 0

def test_token_counter_fallback():
    # Model that doesn't exist in tiktoken mappings should fallback to cl100k_base or char-count
    text = "Hello, world!"
    actual = TokenCounter.count_tokens(text, "non-existent-model")
    # Should be roughly len/4 + 1 if tiktoken fails or cl100k_base if tiktoken handles it via fallback
    assert actual > 0
