from pytest_mockllm.core import TokenCounter
import tiktoken

def test_token_counter_tiktoken(monkeypatch):
    import tiktoken.model
    import tiktoken.registry

    # A real, locally constructed encoder proves preloaded encoders are used
    # without making the unit suite download third-party tokenizer data.
    enc = tiktoken.Encoding(
        name="offline-test", pat_str=r".",
        mergeable_ranks={bytes([i]): i for i in range(256)}, special_tokens={},
    )
    model = "gpt-4o"
    name = tiktoken.model.encoding_name_for_model(model)
    monkeypatch.setitem(tiktoken.registry.ENCODINGS, name, enc)
    text = "Hello, world! This is a test."
    assert TokenCounter.count_tokens(text, model) == len(enc.encode(text))


def test_token_counter_fallback():
    # Model that doesn't exist in tiktoken mappings should fallback to cl100k_base or char-count
    text = "Hello, world!"
    actual = TokenCounter.count_tokens(text, "non-existent-model")
    # Should be roughly len/4 + 1 if tiktoken fails or cl100k_base if tiktoken handles it via fallback
    assert actual > 0


def test_token_counter_does_not_download_an_uncached_tokenizer(monkeypatch):
    import tiktoken.registry
    monkeypatch.setattr(tiktoken.registry, 'ENCODINGS', {})
    def forbidden(*args, **kwargs):
        raise AssertionError('Implicit token counting must not load/download encoders')
    monkeypatch.setattr(tiktoken, 'encoding_for_model', forbidden)
    monkeypatch.setattr(tiktoken, 'get_encoding', forbidden)
    assert TokenCounter.count_tokens('hello', 'gpt-4o') == 2
