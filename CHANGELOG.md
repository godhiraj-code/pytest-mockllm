# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-08-21

### Added

- **OpenAI Responses API support**: mocked `responses.create` and `responses.parse` with official typed objects, including structured output via Pydantic models.
- **Fragmented tool-call streaming**: `tool_call_chunks` streams function arguments across chunks exactly like the real API, sync and async.
- **Deterministic error budgets**: `simulate_error(..., times=N)` raises a fixed number of times before recovering, enabling deterministic 429-to-success retry tests.
- **Request ledger**: `mock.request_ledger` records ordered request evidence with simulated outcomes (`success`, `rate_limit`, ...) for assertions and debugging.
- **Anthropic stream disconnect simulation**: `simulate_stream_disconnect` raises a real `APIConnectionError` mid-stream after N events.
- **SDK contract acceptance tests** for OpenAI and Anthropic proving fixture-scoped zero network egress against a counting localhost transport.

### Hardened

- Provider mocks now patch SDK resource methods (`openai.resources.*`, Anthropic equivalents) so every supported call path is intercepted.
- Unhandled provider SDK requests raise immediately instead of silently reaching the network, including clients created before the fixture activates.

### Fixed

- Bounded the `anthropic` extra to `<1.0`: the Anthropic SDK 1.0 release migrated to `httpx2` and rejects `httpx` clients, which broke fresh installs of the previous unbounded range. Anthropic 1.x support is tracked as follow-up work.
- mypy configuration updated to Python 3.10 so type-checking runs again (current mypy no longer accepts 3.9 targets).

## [0.2.3] - 2026-07-10

### Fixed

- Added the missing `PyYAML` runtime dependency so a base installation can load the pytest plugin.
- Fixed LangChain structured output for sync, async, Pydantic, and `include_raw=True` callers.
- Made unfinished recording and replay modes fail closed instead of silently allowing live API traffic.
- Made `--llm-strict` and the `llm_strict` pytest setting apply to every provider fixture.
- Added a clean-wheel pytest smoke test to CI to catch missing runtime dependencies.

### Documentation

- Replaced the absolute network-safety claim with the actual fixture-scoped guarantee.
- Corrected provider installation examples and marked recording/replay as unavailable pending a safe implementation.

## [0.2.2] - 2025-12-22

### Added

- 🔒 **Enterprise Redaction** - Added PII patterns for Azure OpenAI and Google Cloud (GCP) API keys.
- 🛡️ **Thread-Safe Analytics** - Implemented locking for global statistics to support parallel testing with `pytest-xdist`.

### Fixed

- ⚡ **Non-blocking Async Latency** - Fixed a critical issue where `time.sleep` in jitter simulation would block the async event loop; now uses `asyncio.sleep` for async tests.
- 🔗 **LangChain Parity** - Updated LangChain integration to correctly handle async delays and error simulation.

## [0.2.0] - 2025-12-22

### Added

- 🚀 **True Async Support** - Replaced fake async with real coroutines and async iterators for OpenAI, Anthropic, Gemini, and LangChain.
- 🎯 **Accurate Tokenizers** - Integrated `tiktoken` for OpenAI and improved Claude heuristics for high-fidelity token counting.
- 📊 **Cost Analytics Dashboard** - Professional terminal summary showing USD saved per test run.
- ⚡ **Chaos Engineering** - New `simulate_jitter` and `simulate_random_errors` tools to test application resilience.
- 🔒 **Secure Recording** - Automatic PII redaction (API keys, Bearer tokens) in cassettes using the new `PIIRedactor`.
- 🐍 **Python 3.14 Support** - Full compatibility and CI verification for the latest Python version.

### Fixed

- Resolved `TypeError` when calling async methods on mock clients.
- Improved MyPy type fidelity for provider-specific response objects.
- Fixed intermittent CI failures on Windows and MacOS runners.

## [0.1.0] - 2024-12-22

### Added

- 🎉 Initial release of pytest-mockllm
- ✨ Zero-config pytest plugin with automatic discovery
- 🤖 **OpenAI mock** - Full support for Chat Completions, Embeddings, and Images API
  - Streaming responses with proper SSE format
  - Function calling / tool use support
  - Token usage simulation
- 🧠 **Anthropic mock** - Claude Messages API support
  - Streaming responses
  - Tool use support
- 💎 **Google Gemini mock** - GenerativeAI API support  
  - Chat and content generation
  - Streaming support
- 🦜 **LangChain integration** - Native support for LangChain's ChatModel interface
- 📼 **Response recording** - VCR-like recording and replay for golden tests
- 💰 **Cost estimation** - Mock and assert on token usage and API costs
- ⚡ **Chaos testing** - Simulate rate limits, timeouts, and API errors
- 📝 Comprehensive documentation and examples

### Security

- No external network calls in mock mode (completely isolated testing)

[Unreleased]: https://github.com/godhiraj-code/pytest-mockllm/compare/v0.2.3...HEAD
[0.2.3]: https://github.com/godhiraj-code/pytest-mockllm/releases/tag/v0.2.3
[0.1.0]: https://github.com/godhiraj-code/pytest-mockllm/releases/tag/v0.1.0
