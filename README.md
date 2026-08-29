<h1 align="center">pytest-mockllm</h1>

<p align="center">
  <strong>Fixture-scoped LLM mocking for pytest</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/pytest-mockllm/"><img src="https://img.shields.io/pypi/v/pytest-mockllm?color=blue" alt="PyPI version"></a>
  <a href="https://pypi.org/project/pytest-mockllm/"><img src="https://img.shields.io/pypi/pyversions/pytest-mockllm" alt="Python versions"></a>
  <a href="https://github.com/godhiraj-code/pytest-mockllm/actions"><img src="https://github.com/godhiraj-code/pytest-mockllm/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/godhiraj-code/pytest-mockllm/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
  <a href="https://pypi.org/project/pytest-mockllm/"><img src="https://img.shields.io/pypi/dm/pytest-mockllm" alt="Downloads"></a>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> •
  <a href="#supported-integrations">Supported Integrations</a> •
  <a href="#examples">Examples</a> •
  <a href="#recording-and-replay">Recording</a> •
  <a href="#configuration">Configuration</a>
</p>

---

> [!IMPORTANT]
> **Interception is fixture-scoped.** Calls made through an active provider fixture such as
> `mock_openai` are intercepted locally. Installing the plugin alone does not block network
> access from tests that do not request a mock fixture. Keep real API keys out of unit-test
> environments and use normal CI egress controls as a second safety layer.

## Why pytest-mockllm?

Provider calls make unit tests slower, non-deterministic, dependent on secrets, and potentially
costly. `pytest-mockllm` supplies auto-discovered pytest fixtures that return configured responses
and record call details without contacting the provider on supported fixture paths.

```python
def test_my_chatbot(mock_openai):
    mock_openai.add_response("Hello! I'm here to help.")

    response = my_chatbot.chat("Hi there!")

    assert "help" in response.lower()
    assert mock_openai.call_count == 1
```

## Quick Start

### Installation

Install the extra for the SDK used by your test:

```bash
pip install "pytest-mockllm[openai]"
```

Available extras are:

```bash
pip install "pytest-mockllm[anthropic]"
pip install "pytest-mockllm[google]"
pip install "pytest-mockllm[langchain]"
```

The `langchain` extra includes `langchain-core` and `langchain-openai`, which are required by the
LangChain example below. Use `pytest-mockllm[all]` to install every supported integration. A base
`pip install pytest-mockllm` installs the pytest plugin and its core runtime dependencies, but not
the optional provider SDKs.

The plugin is discovered automatically through its `pytest11` entry point; no `pytest_plugins`
setting or fixture import is required.

### Your First Test

```python
def test_customer_support_bot(mock_openai):
    mock_openai.add_response("I can help with your order. What's your order number?")

    from openai import OpenAI

    client = OpenAI(api_key="fixture-only")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "I need help with my order"}],
    )

    assert "order number" in response.choices[0].message.content.lower()
    assert mock_openai.call_count == 1
    assert mock_openai.last_call["model"] == "gpt-4o"
```

The placeholder key only satisfies the SDK constructor. The active fixture intercepts the
supported call before transport.

## Supported Integrations

| Fixture | Install extra | Intercepted interfaces |
|---|---|---|
| `mock_openai` | `openai` | Chat Completions, Responses create/parse, embeddings; sync/async; chat streaming and tool-call chunks |
| `mock_anthropic` | `anthropic` | Messages; sync/async; text and tool-use streaming |
| `mock_gemini` | `google` | `GenerativeModel.generate_content`, async generation, streaming, and chat sessions |
| `mock_langchain` | `langchain` | Installed ChatOpenAI-compatible classes plus direct mock model `invoke`, `ainvoke`, `stream`, `astream`, and structured output |
| `mock_llm` | matching provider extra | OpenAI by default; select `openai`, `anthropic`, or `gemini` with `@pytest.mark.llm_mock` |

OpenAI and Anthropic supported paths return official SDK response/event objects when their SDKs
are installed. Their active fixtures also block unhandled requests at the SDK base-request layer,
including requests from clients created before fixture activation. Gemini and LangChain replace
the supported high-level entry points listed above; they are not global network guards for every
function exposed by those libraries.

The `google` extra currently targets the legacy `google-generativeai` package. That upstream SDK is
deprecated; support for its replacement, `google-genai`, has not been implemented yet.

All fixtures support queued deterministic responses, strict mode, call tracking, token/cost
estimates, and deterministic error simulation. Response fidelity is limited to the interfaces in
the table; this package does not claim complete coverage of every provider API.

## Provider Examples

### OpenAI

```python
def test_openai(mock_openai):
    mock_openai.add_response("The answer is 42")

    from openai import OpenAI

    client = OpenAI(api_key="fixture-only")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "What is the meaning of life?"}],
    )

    assert response.choices[0].message.content == "The answer is 42"
```

### Anthropic

```python
def test_anthropic(mock_anthropic):
    mock_anthropic.add_response("I'd be happy to help!")

    from anthropic import Anthropic

    client = Anthropic(api_key="fixture-only")
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello Claude!"}],
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

    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o", api_key="fixture-only")
    prompt = ChatPromptTemplate.from_template("What is the capital of {country}?")
    chain = prompt | llm

    result = chain.invoke({"country": "France"})

    assert "Paris" in result.content
```

## Examples

### Multiple Responses

```python
def test_conversation(mock_openai):
    mock_openai.add_responses(
        "Hi! How can I help you today?",
        "I can help with that order.",
        "Your order has been updated.",
    )

    assert "help" in chatbot.send("Hello")
    assert "order" in chatbot.send("I need to change my order")
    assert "updated" in chatbot.send("Change quantity to 5")
```

### Streaming Responses

```python
def test_streaming(mock_openai):
    mock_openai.add_response("This is a streaming response that comes in chunks")

    from openai import OpenAI

    client = OpenAI(api_key="fixture-only")
    stream = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Tell me a story"}],
        stream=True,
    )

    full_response = "".join(
        chunk.choices[0].delta.content or ""
        for chunk in stream
    )
    assert "streaming" in full_response
```

### Function and Tool Calling

```python
def test_function_calling(mock_openai):
    mock_openai.add_response(
        "",
        tool_calls=[{
            "id": "call_123",
            "function": {
                "name": "get_weather",
                "arguments": {"location": "San Francisco", "unit": "celsius"},
            },
        }],
    )

    from openai import OpenAI

    client = OpenAI(api_key="fixture-only")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "What's the weather?"}],
    )

    tool_call = response.choices[0].message.tool_calls[0]
    assert tool_call.function.name == "get_weather"
    assert mock_openai.last_call["model"] == "gpt-4o"
```

### Token Usage and Cost Assertions

```python
def test_stays_within_budget(mock_openai):
    from pytest_mockllm.core import TokenUsage, estimate_cost

    mock_openai.add_response(
        "A detailed response...",
        token_usage=TokenUsage(prompt_tokens=500, completion_tokens=1000),
    )

    my_function()

    assert mock_openai.total_tokens < 2000
    cost = estimate_cost(
        "gpt-4o",
        mock_openai.total_prompt_tokens,
        mock_openai.total_completion_tokens,
    )
    assert cost < 0.05
```

### Error Simulation

```python
def test_handles_rate_limit(mock_openai):
    mock_openai.simulate_error("rate_limit", times=1)
    mock_openai.add_response("recovered")

    # The first supported provider call raises a provider-style rate-limit error.
    # The next call returns "recovered".


def test_handles_jitter(mock_openai):
    mock_openai.simulate_jitter(max_ms=500)


def test_random_failures(mock_openai):
    mock_openai.simulate_random_errors(probability=0.1)
```

### Strict Mode

```python
import pytest


def test_catches_unconfigured_calls(mock_openai):
    mock_openai.set_strict_mode(True)

    with pytest.raises(RuntimeError, match="No mock response configured"):
        my_function_that_calls_llm()
```

## Recording and Replay

Recording and replay are currently unavailable. Earlier releases exposed the interface without
safely intercepting provider calls. Current `auto`, `record`, and `replay` modes fail before test
code can reach a provider instead of allowing an apparent replay test to fall through to a live
API. Use the deterministic provider fixtures until recording returns with provider-level
behavioral tests.

The `--llm-record`, `--llm-cassette-dir`, `llm_record`, and `llm_replay` names remain reserved for
future compatibility; using them with `llm_recorder` does not create or replay cassettes today.

## Configuration

### Strict Mode

```bash
pytest --llm-strict
```

Strict mode makes a provider fixture fail when a supported call has no configured response. It can
also be enabled in pytest configuration:

```toml
[tool.pytest.ini_options]
llm_strict = true
```

### Universal Fixture

```python
import pytest


@pytest.mark.llm_mock(provider="anthropic")
def test_with_anthropic(mock_llm):
    mock_llm.add_response("Configured through the universal fixture")
    # Calls through the Anthropic Messages API are intercepted here.
```

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

```bash
pip install -e ".[dev,all]"
pytest
ruff check src/
mypy src/
python -m build
```

## Roadmap

- [x] Async provider fixtures
- [x] Token and cost estimates
- [x] Deterministic latency and error simulation
- [ ] Safe provider-level recording and replay
- [ ] More providers
- [ ] Process-safe pytest-xdist aggregation

## License

MIT License — see [LICENSE](LICENSE) for details.
