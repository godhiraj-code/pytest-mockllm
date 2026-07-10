<h1 align="center">🧪 pytest-mockllm</h1>

<p align="center">
  <strong>🚀 Zero-config LLM mocking for pytest — Test AI apps without the AI bills</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/pytest-mockllm/"><img src="https://img.shields.io/pypi/v/pytest-mockllm?color=blue" alt="PyPI version"></a>
  <a href="https://pypi.org/project/pytest-mockllm/"><img src="https://img.shields.io/pypi/pyversions/pytest-mockllm" alt="Python versions"></a>
  <a href="https://github.com/godhiraj-code/pytest-mockllm/actions"><img src="https://github.com/godhiraj-code/pytest-mockllm/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/godhiraj-code/pytest-mockllm/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
  <a href="https://pypi.org/project/pytest-mockllm/"><img src="https://img.shields.io/pypi/dm/pytest-mockllm" alt="Downloads"></a>
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-features">Features</a> •
  <a href="#-providers">Providers</a> •
  <a href="#-examples">Examples</a> •
  <a href="#-recording--replay">Recording</a> •
  <a href="#-configuration">Configuration</a>
</p>

---

> [!IMPORTANT]
> **Fixture-scoped interception:** Calls made through an active provider fixture such as
> `mock_openai` are intercepted locally. Installing the plugin alone does not block network
> access from tests that do not use a mock fixture. Keep real API keys out of unit-test
> environments and use normal CI egress controls as a second safety layer.

## Why pytest-mockllm?

Testing LLM applications is **painful**:

- 💸 **Expensive** — Every test run burns API credits
- 🐢 **Slow** — API calls add seconds to your test suite  
- 🎲 **Non-deterministic** — Same input, different output = flaky tests
- 🔒 **Requires API keys** — CI needs secrets, local dev needs setup

**pytest-mockllm** fixes all of this with **zero configuration**:

```python
# Just use the fixture — it works immediately!
def test_my_chatbot(mock_openai):
    mock_openai.add_response("Hello! I'm here to help.")
    
    response = my_chatbot.chat("Hi there!")
    
    assert "help" in response.lower()
    assert mock_openai.call_count == 1
```

No setup. No API keys. No costs. **Just fast, reliable tests.**

---

## 🚀 Quick Start

### Installation

```bash
pip install "pytest-mockllm[openai]"
```

Use `pytest-mockllm[anthropic]`, `pytest-mockllm[google]`, or
`pytest-mockllm[langchain]` for the corresponding provider. The plugin is auto-discovered by pytest.

### Your First Test

```python
def test_customer_support_bot(mock_openai):
    # Configure the mock response
    mock_openai.add_response("I can help you with your order. What's your order number?")
    
    # Your actual code that uses OpenAI
    from openai import OpenAI
    client = OpenAI(api_key="fake-key")  # Key doesn't matter!
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "I need help with my order"}]
    )
    
    # Assert on the response
    assert "order number" in response.choices[0].message.content.lower()
    
    # Assert on what was called
    assert mock_openai.call_count == 1
    assert mock_openai.last_call["model"] == "gpt-4o"
```

---

## ✨ Features

### 🎯 Zero Configuration
Fixtures are auto-discovered. Just use them.

### 🤖 Multi-Provider Support
OpenAI, Anthropic, Google Gemini — one consistent API.

### 🌊 Streaming Support
Full support for streaming responses, just like the real APIs.

### 🔧 LangChain
Native integration with LangChain chat models.

### ⚡ Chaos Engineering
Simulate rate limits, timeouts, and random latency jitter to test your app's resilience.

### 💰 Cost & Token Tracking
Professional-grade token counting with `tiktoken` and built-in cost dashboard. See [Live Benchmarks](benchmarks/BENCHMARK.md).

### 🔒 Type Safe
Full type hints and objects that match SDK structures perfectly.

---

## 🆕 What's New in v0.2.3 "Safety Hardening"

- Fixed clean-install plugin loading by declaring the required PyYAML dependency.
- Fixed LangChain structured output for sync, async, Pydantic, and `include_raw=True` callers.
- Recording and replay now fail closed instead of silently falling through to live provider APIs.
- CI now installs the built wheel in a clean environment and runs a pytest smoke test.
- `--llm-strict` and the corresponding pytest configuration now apply to provider fixtures.

---

## 🤖 Providers

### OpenAI

```python
def test_openai(mock_openai):
    mock_openai.add_response("The answer is 42")
    
    # Works with the official OpenAI SDK
    from openai import OpenAI
    client = OpenAI(api_key="fake")
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "What is the meaning of life?"}]
    )
    
    assert response.choices[0].message.content == "The answer is 42"
```

### Anthropic

```python
def test_anthropic(mock_anthropic):
    mock_anthropic.add_response("I'd be happy to help!")
    
    from anthropic import Anthropic
    client = Anthropic(api_key="fake")
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello Claude!"}]
    )
    
    assert "happy" in response.content[0].text
```

### Google Gemini

```python
def test_gemini(mock_gemini):
    mock_gemini.add_response("Here's what I found...")
    
    import google.generativeai as genai
    model = genai.GenerativeModel("gemini-1.5-pro")
    
    response = model.generate_content("Tell me about AI")
    
    assert "found" in response.text
```

### LangChain

```python
def test_langchain(mock_langchain):
    mock_langchain.add_response("Paris is the capital of France.")
    
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    
    llm = ChatOpenAI(model="gpt-4o", api_key="fake")
    prompt = ChatPromptTemplate.from_template("What is the capital of {country}?")
    chain = prompt | llm
    
    result = chain.invoke({"country": "France"})
    
    assert "Paris" in result.content
```

---

## 📚 Examples

### Multiple Responses (Conversation)

```python
def test_conversation(mock_openai):
    mock_openai.add_responses(
        "Hi! How can I help you today?",
        "I can definitely help with that order.",
        "Your order has been updated. Anything else?",
    )
    
    # First call
    response1 = chatbot.send("Hello")
    assert "help" in response1
    
    # Second call
    response2 = chatbot.send("I need to change my order")  
    assert "order" in response2
    
    # Third call
    response3 = chatbot.send("Change quantity to 5")
    assert "updated" in response3
```

### Streaming Responses

```python
def test_streaming(mock_openai):
    mock_openai.add_response("This is a streaming response that comes in chunks")
    
    from openai import OpenAI
    client = OpenAI(api_key="fake")
    
    stream = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Tell me a story"}],
        stream=True,
    )
    
    full_response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content:
            full_response += chunk.choices[0].delta.content
    
    assert "streaming" in full_response
```

### Function/Tool Calling

```python
def test_function_calling(mock_openai):
    from pytest_mockllm.core import MockResponse
    
    mock_openai._responses.append(MockResponse(
        content="",
        tool_calls=[{
            "id": "call_123",
            "function": {
                "name": "get_weather",
                "arguments": {"location": "San Francisco", "unit": "celsius"},
            },
        }],
    ))
    
    # Your function-calling logic here
    # ...
    
    assert mock_openai.last_call is not None
```

### Token Usage & Cost Assertions

```python
def test_stays_within_budget(mock_openai):
    from pytest_mockllm.core import TokenUsage, estimate_cost
    
    mock_openai.add_response(
        "A detailed response...",
        token_usage=TokenUsage(prompt_tokens=500, completion_tokens=1000),
    )
    
    # Your LLM call here
    result = my_function()
    
    # Assert token usage
    assert mock_openai.total_tokens < 2000
    assert mock_openai.total_completion_tokens < 1500
    
    # Assert cost (for gpt-4o)
    cost = estimate_cost(
        "gpt-4o",
        mock_openai.total_prompt_tokens,
        mock_openai.total_completion_tokens,
    )
    assert cost < 0.05  # Less than 5 cents
```

### Error Simulation (Chaos Testing)

```python
def test_handles_rate_limit(mock_openai):
    mock_openai.simulate_error("rate_limit", after_calls=2)
    mock_openai.add_responses("OK", "OK")
    
    # First two calls succeed, third fails
    # ...

def test_handles_jitter(mock_openai):
    # Add up to 500ms random latency to every call
    mock_openai.simulate_jitter(max_ms=500)
    # ...

def test_random_failures(mock_openai):
    # 10% chance of random "server" or "rate_limit" error
    mock_openai.simulate_random_errors(probability=0.1)
    # ...
```

### Strict Mode

```python
def test_catches_unconfigured_calls(mock_openai):
    mock_openai.set_strict_mode(True)
    
    # This will raise an error because no response is configured
    with pytest.raises(RuntimeError, match="No mock response configured"):
        my_function_that_calls_llm()
```

---

## 📼 Recording & Replay

Recording and replay are temporarily unavailable. Earlier releases exposed the interface
without safely intercepting provider calls. Version 0.2.3 fails closed instead of allowing
an apparent replay test to fall through to a live API. Use the deterministic provider fixtures
until recording returns with provider-level behavioral tests.

---

## 🔧 Configuration

### CLI Options

```bash
# Strict mode (fail if any LLM call is unconfigured)
pytest --llm-strict
```

The reserved `--llm-record` and `--llm-cassette-dir` options currently fail closed when the
`llm_recorder` fixture is used.

### Markers

```python
@pytest.mark.llm_mock(provider="anthropic")
def test_with_anthropic(mock_llm):
    # mock_llm is now an AnthropicMock
    pass

```

### pytest.ini / pyproject.toml

```toml
[tool.pytest.ini_options]
# Default cassette directory
llm_cassette_dir = "tests/fixtures/llm"

# Always run in strict mode
llm_strict = true
```

---

## 🆚 Comparison

| Feature | pytest-mockllm | unittest.mock | responses | vcrpy |
|---------|---------------|---------------|-----------|-------|
| Zero config | ✅ | ❌ | ❌ | ❌ |
| pytest fixtures | ✅ | ❌ | ✅ | ✅ |
| Async support | ✅ True Async | 🟡 Complex | ❌ | 🟡 HTTP |
| OpenAI/Anthropic | ✅ Native | 🟡 Manual | ❌ | 🟡 HTTP |
| Gemini support | ✅ Native | 🟡 Manual | ❌ | 🟡 HTTP |
| Token counting | ✅ tiktoken | ❌ | ❌ | ❌ |
| Cost Dashboard | ✅ | ❌ | ❌ | ❌ |
| Recording/Replay | 🚧 Fail-closed | ❌ | ❌ | ✅ |
| Chaos Engineering| ✅ Jitter/Error| 🟡 Manual | 🟡 HTTP | ❌ |

---

## 🛣️ Roadmap

- [x] True Async/await support
- [x] Professional Tokenizers (tiktoken)
- [x] Terminal Cost Dashboard
- [x] Chaos Engineering (Jitter)
- [ ] Safe provider-level recording and replay
- [ ] More providers (Cohere, Mistral)
- [ ] pytest-xdist compatibility
- [ ] Integration with LangSmith

---

## 🤝 Contributing

We'd love your help! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Clone the repo
git clone https://github.com/godhiraj-code/pytest-mockllm.git

# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check src/
mypy src/
```

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <strong>Stop paying for tests. Start shipping faster.</strong>
</p>

<p align="center">
  <a href="https://github.com/godhiraj-code/pytest-mockllm">⭐ Star us on GitHub</a> •
  <a href="https://www.dhirajdas.dev">Built by Dhiraj Das</a>
</p>
