<p align="center">
  <img src="https://raw.githubusercontent.com/godhiraj-code/pytest-mockllm/main/docs/logo.png" alt="pytest-mockllm logo" width="200">
</p>

<h1 align="center">pytest-mockllm</h1>

<p align="center">
  <strong>🚀 Zero-config LLM mocking for pytest — Test AI apps without the AI bills</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/pytest-mockllm/"><img src="https://img.shields.io/pypi/v/pytest-mockllm.svg" alt="PyPI version"></a>
  <a href="https://pypi.org/project/pytest-mockllm/"><img src="https://img.shields.io/pypi/pyversions/pytest-mockllm.svg" alt="Python versions"></a>
  <a href="https://github.com/godhiraj-code/pytest-mockllm/actions"><img src="https://github.com/godhiraj-code/pytest-mockllm/workflows/CI/badge.svg" alt="CI"></a>
  <a href="https://codecov.io/gh/godhiraj-code/pytest-mockllm"><img src="https://codecov.io/gh/godhiraj-code/pytest-mockllm/branch/main/graph/badge.svg" alt="Coverage"></a>
  <a href="https://github.com/godhiraj-code/pytest-mockllm/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-features">Features</a> •
  <a href="#-providers">Providers</a> •
  <a href="#-examples">Examples</a> •
  <a href="#-recording">Recording</a> •
  <a href="#-docs">Docs</a>
</p>

---

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
pip install pytest-mockllm
```

That's it! The plugin is auto-discovered by pytest.

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

### 🔧 LangChain & LlamaIndex
Native integration with popular LLM frameworks.

### 📼 Response Recording
VCR-style recording for golden tests.

### 💰 Cost & Token Tracking
Assert on costs before they become production surprises.

### ⚡ Chaos Testing
Simulate rate limits, timeouts, and API errors.

### 🔒 Type Safe
Full type hints and mypy compliance.

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
    
    # First two calls succeed
    assert my_function() == "OK"
    assert my_function() == "OK"
    
    # Third call hits rate limit  
    with pytest.raises(Exception):
        my_function()

def test_handles_timeout():
    mock_openai.simulate_error("timeout")
    
    # Should trigger your retry logic
    with pytest.raises(TimeoutError):
        my_function()
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

Record real API responses once, replay them forever — like VCR for LLMs.

### Record Mode

```python
# First run: hits real API and saves response
@pytest.mark.llm_record
def test_with_recording(llm_recorder):
    from openai import OpenAI
    client = OpenAI()  # Uses real API key from environment
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Use cheap model for recording
        messages=[{"role": "user", "content": "Say hello!"}]
    )
    
    assert response.choices[0].message.content
```

Run with recording:
```bash
pytest tests/test_example.py --llm-record
```

### Replay Mode

```python
# Subsequent runs: uses saved response (no API key needed!)
@pytest.mark.llm_replay
def test_with_recording(llm_recorder):
    # Same test code — but now uses cached response
    # ...
```

### Cassette Storage

Responses are saved in `tests/llm_cassettes/` as YAML:

```yaml
name: test_with_recording
version: "1.0"
created: 1703270400
interactions:
  - request:
      model: gpt-4o-mini
      messages:
        - role: user
          content: Say hello!
    response:
      content: "Hello! How can I assist you today?"
      model: gpt-4o-mini
    provider: openai
    latency_ms: 523
```

---

## 🔧 Configuration

### CLI Options

```bash
# Enable recording mode
pytest --llm-record

# Custom cassette directory
pytest --llm-cassette-dir=my_cassettes

# Strict mode (fail if any LLM call is unconfigured)
pytest --llm-strict
```

### Markers

```python
@pytest.mark.llm_mock(provider="anthropic")
def test_with_anthropic(mock_llm):
    # mock_llm is now an AnthropicMock
    pass

@pytest.mark.llm_record
def test_records_responses(llm_recorder):
    pass

@pytest.mark.llm_replay  
def test_replays_responses(llm_recorder):
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
| OpenAI support | ✅ Native | 🟡 Manual | ❌ | 🟡 HTTP |
| Anthropic support | ✅ Native | 🟡 Manual | ❌ | 🟡 HTTP |
| Gemini support | ✅ Native | 🟡 Manual | ❌ | 🟡 HTTP |
| LangChain support | ✅ Native | 🟡 Complex | ❌ | 🟡 LimitedComplex |
| Streaming | ✅ | 🟡 Manual | ❌ | 🟡 Complex |
| Token tracking | ✅ | ❌ | ❌ | ❌ |
| Cost estimation | ✅ | ❌ | ❌ | ❌ |
| Recording/Replay | ✅ | ❌ | ❌ | ✅ |
| Error simulation | ✅ | 🟡 Manual | 🟡 HTTP | ❌ |

---

## 🛣️ Roadmap

- [ ] Async/await support improvements
- [ ] More providers (Cohere, Mistral, Together AI)
- [ ] pytest-xdist compatibility
- [ ] Response fuzzing for robustness testing
- [ ] Integration with LangSmith for debugging
- [ ] Automatic prompt regression detection

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
