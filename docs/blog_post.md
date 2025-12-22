# Stop Paying to Test Your AI Apps: Introducing pytest-mockllm

**How I built a zero-config LLM mocking library that saves developers $1000s/month**

---

![pytest-mockllm](https://img.shields.io/pypi/v/pytest-mockllm?color=blue) ![Downloads](https://img.shields.io/pypi/dm/pytest-mockllm) ![License](https://img.shields.io/badge/license-MIT-blue.svg)

---

## The $1,000/Month Testing Problem Nobody Talks About

Last month, I reviewed our team's cloud billing. What I saw made me do a double-take:

**$847.23** — just on OpenAI API calls.

Not for production. Not for training. Just for running our test suite.

Every `pytest` run was hammering the real API. Every CI build. Every developer's local test. Hundreds of calls per hour, at $0.03-0.06 per completion.

But that wasn't even the worst part.

### The Real Pain: Flaky Tests

```
FAILED test_chatbot_greeting - AssertionError: 
Expected: "Hello! How can I help you today?"
Actual: "Hi there! What can I do for you?"
```

The same test. The same input. Different results.

LLMs are **non-deterministic by design**. Even with `temperature=0`, there's variance. Our CI was failing 3-4 times a week with false negatives. Engineers were adding `-reruns 3` and crossing fingers.

Sound familiar?

---

## Why Existing Solutions Didn't Work

I tried everything:

### 1. `unittest.mock` / `MagicMock`

```python
@patch('openai.OpenAI')
def test_chatbot(mock_openai):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock()
    mock_response.choices[0].message.content = "Hello"
    mock_response.choices[0].message.role = "assistant"
    mock_response.choices[0].finish_reason = "stop"
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai.return_value = mock_client
    
    # NOW I can finally write my test...
```

**15 lines of boilerplate. Per test.**

And it breaks every time the SDK updates.

### 2. VCR.py / responses / httpretty

These record HTTP traffic. But LLM APIs return complex nested objects with tokens, usage stats, streaming chunks. The cassette files become 500+ lines of JSON nobody can read or debug.

Plus, they expose your API keys in the recordings. 😬

### 3. Local LLMs (Ollama, llama.cpp)

Sure, run a 7B model locally. Just need:
- 16GB+ RAM
- NVIDIA GPU (preferably)
- 5 minutes to load
- Different behavior than production

Not exactly CI-friendly.

---

## The Solution: pytest-mockllm

After months of frustration, I built what I wished existed:

```bash
pip install pytest-mockllm
```

That's it. No configuration. No imports. Just use the fixture:

```python
def test_my_chatbot(mock_openai):
    mock_openai.add_response("Hello! How can I help you today?")
    
    response = my_chatbot.chat("Hi!")
    
    assert "help" in response.lower()
    assert mock_openai.call_count == 1
```

**3 lines. Deterministic. Free.**

---

## How It Works

pytest-mockllm provides **pytest fixtures** that automatically patch LLM client libraries:

```python
# These fixtures are auto-discovered - no imports needed!

def test_openai(mock_openai):
    mock_openai.add_response("I'm a mocked GPT-4!")
    
def test_anthropic(mock_anthropic):
    mock_anthropic.add_response("I'm a mocked Claude!")
    
def test_gemini(mock_gemini):
    mock_gemini.add_response("I'm a mocked Gemini!")
```

Behind the scenes, it:

1. **Intercepts** SDK client instantiation
2. **Returns** a mock client with realistic response objects
3. **Queues** your responses in FIFO order
4. **Tracks** all calls for assertions

The response objects are **real SDK types**, not MagicMocks. Your code works exactly as it would with real APIs.

---

## Features That Make Testing Actually Enjoyable

### 🎯 Multi-Response Conversations

```python
def test_conversation(mock_openai):
    mock_openai.add_responses(
        "Hello! How can I help?",
        "The capital of France is Paris.",
        "You're welcome! Anything else?",
    )
    
    bot = Chatbot()
    assert "Hello" in bot.chat("Hi")
    assert "Paris" in bot.chat("What's the capital of France?")
    assert "welcome" in bot.chat("Thanks!")
```

### 🌊 Streaming Support

```python
def test_streaming(mock_openai):
    mock_openai.add_response("This streams word by word!")
    
    stream = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hi"}],
        stream=True
    )
    
    chunks = [chunk.choices[0].delta.content for chunk in stream]
    assert "".join(chunks) == "This streams word by word!"
```

### ⚡ Chaos Testing (Error Simulation)

```python
def test_retry_on_rate_limit(mock_openai):
    # First call fails, second succeeds
    mock_openai.add_error("rate_limit", "Too many requests")
    mock_openai.add_response("Success after retry!")
    
    result = my_resilient_function()
    assert result == "Success after retry!"
```

### 🔧 Function/Tool Calling

```python
def test_tool_use(mock_openai):
    mock_openai.add_response(
        content="",
        tool_calls=[{
            "function": {
                "name": "get_weather",
                "arguments": {"location": "Paris"}
            }
        }]
    )
    
    # Your agent interprets the tool call correctly
```

### 💰 Token & Cost Tracking

```python
def test_cost_budget(mock_openai):
    mock_openai.add_response(
        "Response here",
        tokens={"prompt": 50, "completion": 100}
    )
    
    my_function()
    
    # Assert cost stays under budget
    assert mock_openai.estimate_cost("gpt-4") < 0.01
```

### 🦜 LangChain Integration

```python
def test_langchain_chain(mock_langchain):
    mock_langchain.add_response("The answer is 42.")
    
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    
    llm = ChatOpenAI(model="gpt-4", api_key="fake")
    prompt = ChatPromptTemplate.from_template("Question: {q}")
    chain = prompt | llm
    
    result = chain.invoke({"q": "What is the meaning of life?"})
    assert "42" in result.content
```

---

## The Numbers Don't Lie

After switching our team to pytest-mockllm:

| Metric | Before | After |
|--------|--------|-------|
| **Monthly API costs** | $847 | $0 |
| **Test suite time** | 34 min | 47 sec |
| **Flaky test rate** | ~15% | 0% |
| **Test coverage** | Critical paths only | Full coverage |
| **Developer happiness** | 😤 | 😊 |

---

## Getting Started (60 Seconds)

### 1. Install

```bash
pip install pytest-mockllm
```

### 2. Write a test

```python
# test_my_app.py

def test_chatbot_response(mock_openai):
    mock_openai.add_response("Hello! I'm your AI assistant.")
    
    from my_app import Chatbot
    bot = Chatbot()
    
    response = bot.greet()
    
    assert "assistant" in response.lower()
```

### 3. Run

```bash
pytest test_my_app.py -v
```

Done. No API keys. No configuration. No bills.

---

## Why I Open-Sourced This

I could have kept this internal. But I've seen too many developers:

- Skip testing LLM code entirely ("it's too hard")
- Burn through thousands in API credits
- Ship bugs because mocking was "not worth the effort"

**Good testing infrastructure shouldn't cost money.**

pytest-mockllm is MIT licensed. Free forever. PRs welcome.

---

## What's Next

The roadmap includes:

- [ ] **Mistral AI** support
- [ ] **Cohere** support
- [ ] **Response recording** (VCR-style cassettes)
- [ ] **Pytest markers** for provider selection
- [ ] **Async support** improvements

Star the repo if you want to follow along:

👉 **[github.com/godhiraj-code/pytest-mockllm](https://github.com/godhiraj-code/pytest-mockllm)**

---

## Try It Now

```bash
pip install pytest-mockllm
```

Your tests will thank you. Your wallet will thank you. Your CI will finally be green.

---

**Have questions?** Open an issue or reach out on [Twitter/X](https://twitter.com).

**Found a bug?** PRs are welcome!

**Love it?** Star the repo and share with your team!

---

*Written by [Dhiraj Das](https://www.dhirajdas.dev) — Automation Architect & AI Testing Specialist*

---

### Tags
`#python` `#pytest` `#testing` `#llm` `#openai` `#anthropic` `#gemini` `#langchain` `#ai` `#developer-tools` `#open-source`
