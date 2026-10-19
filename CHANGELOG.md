# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - Unreleased

### Breaking changes / upgrading from 0.3.x

- Recording is now implemented: the recording flags and markers no longer always
  fail closed as unavailable. `--llm-record=once` can make real, billable API calls
  when a cassette does not exist; `new_episodes` records unmatched calls and `all`
  replaces recordings. Use `--llm-record=none` (the default) for offline replay;
  missing or mismatched interactions still fail closed. Review the selected mode
  before running tests with real credentials, and regenerate old experimental
  cassettes with the new recorder.
- Replace direct `LLMRecorder(mode="auto"|"record"|"replay")` usage with
  `record_mode="once"|"all"|"none"`, respectively. The old spellings warn but still
  work. Legacy `mode="none"` means live pass-through; new `record_mode="none"`
  means offline replay. Do not substitute one for the other without checking intent.
- Anthropic fixtures keep `mock_anthropic`, `add_response`, `simulate_error`, and
  their existing arguments. Helpers that assumed a bare generator or a configurable
  `MagicMock` from `messages.stream()` must use the SDK's stream context manager:
  `with client.messages.stream(...) as stream:` (or `async with` for async clients),
  then consume `stream.text_stream` or call `stream.get_final_message()` (await it
  for async clients). `messages.create(stream=True)` yields official typed events;
  access text with `event.delta.text` after checking its event/delta type. Configure
  fixture output through `add_response(..., stream_chunks=..., tool_calls=...)`
  rather than mutating private response/event-building helpers. The Anthropic SDK
  minimum is now 0.75.0; public fixture helpers do not require renaming.
- `[google]` now uses `google-genai`; import `from google import genai` and construct
  `genai.Client(...)`. Install `[google-legacy]` to retain the deprecated
  `google.generativeai` adapter temporarily. Enable `--llm-stats` if you relied on
  the previously automatic cost summary.

### Added

- Working OpenAI and Anthropic recording/replay using readable YAML cassettes,
  with `--llm-record=none|once|new_episodes|all`, the `llm_recorder` fixture,
  automatic recording/replay markers, and offline replay by default.
- SDK-parsed sync/async JSON and SSE playback, including fragmented tool calls,
  OpenAI Responses structured parsing and embeddings, Anthropic stream helpers,
  and HTTP error/retry sequences. Recording buffers complete streams; arrival
  timing and partial transport failures are not replayed.
- Request matching includes provider, method, URL/query, JSON body, and selected
  semantic headers. Identical requests consume interactions in order. Misses in
  offline modes and unsupported endpoints fail before network access.
- Credential/header redaction for requests, response JSON and SSE, atomic cassette
  writes, and collision-resistant per-test paths. Redaction is not general PII removal.
- Native `google-genai` typed generation, chat sessions, sync/async streaming,
  function calls, structured output, and provider error classes.
- Typed OpenAI Responses streams and function-call outputs, SDK stream context
  managers, and final structured output through `responses.stream`.
- `--llm-stats`, `--no-llm-stats`, and `llm_stats` configuration; statistics are now
  off by default.
- A tested README quickstart, provider/stream/tool/error/recording examples, and
  an explicit comparison with mock, transport, recording, and evaluation tools.
- An actual asciinema demo (`docs/demo.cast`) and a portable regeneration script.
- `py.typed` package marker and version consistency checks.

### Changed

- `[google]` and `[all]` now install `google-genai`. `[google-legacy]` retains the
  old `google-generativeai` adapter with a deprecation warning. Existing fixture
  names, response configuration arguments, and the legacy module accessor remain.
- New direct recorder usage selects `record_mode`. Legacy `mode=auto|record|replay`
  maps to once/all/none with a deprecation warning. Legacy `mode='none'` retains
  explicit live pass-through; `record_mode='none'` is offline replay.
- Provider dependency floors now match tested SDK versions: OpenAI 1.109.1,
  Anthropic 0.75.0, google-genai 1.47.0, langchain-core 0.3.80, and
  langchain-openai 0.3.35. Removed the Anthropic `<1.0` cap after testing 1.15.0.
  HTTPX and HTTPX2 response types follow the installed provider SDK.

### Fixed

- Redaction preserves JSON/tool-schema fields named `key`, `token`, `auth`, and
  similar names. Credential headers/query parameters and known credential values
  remain redacted; `sk-ant-...` keys carry the Anthropic label.
- Tiny nonzero estimated costs display as `<$0.0001` instead of `$0.0000`.
- Simulated timeout/server errors now use the SDK's correct constructors.
- Async OpenAI Responses latency yields the event loop.
- Responses tool-arguments completion events include the function name required
  by some SDK versions; the Google dependency floor preserves Python 3.9 installs.
- Chat tool streams no longer require explicit argument fragments and support
  multiple tools. Anthropic streams preserve both text and tool-use blocks.
- Legacy Gemini streaming no longer passes duplicate `stream` arguments, and
  positional prompts/model names are included in the request ledger.
- Global strict mode also applies to LangChain.
- Implicit token estimates no longer load/download tiktoken encoders. Already-loaded
  encoders remain usable; otherwise counting falls back to a character estimate.
- Package `__version__` now agrees with project metadata at 0.4.0.

### Validation and remaining limits

- Full suite exercised on Python 3.10–3.13; minimum/current SDK contract tests,
  actual local HTTP record/replay, executable README examples, and package checks.
- No paid/live-provider validation or PyPI publication. Gemini recording, LiteLLM,
  new assertion helpers, and a live-test environment marker are not included.
- See [SDK compatibility](docs/sdk-compatibility.md) for versions, test evidence,
  private-SDK-hook risk, buffered streaming, and unsupported interfaces.

## [0.3.1] - 2026-08-30

### Fixed

- The `langchain` extra now installs `langchain-openai`, so the documented `ChatOpenAI` example
  works after installing `pytest-mockllm[langchain]`.
- Gemini now patches the real SDK module when it was imported before fixture activation.
- LangChain provider classes imported during test collection now have their supported call methods
  intercepted by the active fixture instead of reaching the provider.

### Documentation

- Replaced broad provider-fidelity and network-safety language with the supported fixture paths
  and fixture-scoped guarantees.
- Clarified that recording and replay options are reserved and currently fail closed.
- Removed decorative emoji and obsolete comparison marketing.

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

- **Enterprise Redaction** - Added PII patterns for Azure OpenAI and Google Cloud (GCP) API keys.
- **Thread-Safe Analytics** - Implemented locking for global statistics to support parallel testing with `pytest-xdist`.

### Fixed

- **Non-blocking Async Latency** - Fixed a critical issue where `time.sleep` in jitter simulation would block the async event loop; now uses `asyncio.sleep` for async tests.
- **LangChain Parity** - Updated LangChain integration to correctly handle async delays and error simulation.

## [0.2.0] - 2025-12-22

### Added

- **True Async Support** - Replaced fake async with real coroutines and async iterators for OpenAI, Anthropic, Gemini, and LangChain.
- **Accurate Tokenizers** - Integrated `tiktoken` for OpenAI and improved Claude heuristics for high-fidelity token counting.
- **Cost Analytics Dashboard** - Terminal summary showing estimated USD saved per test run.
- **Chaos Engineering** - New `simulate_jitter` and `simulate_random_errors` tools to test application resilience.
- **Secure Recording** - Automatic PII redaction (API keys, Bearer tokens) in cassettes using the new `PIIRedactor`.
- **Python 3.14 Support** - Compatibility and CI coverage for Python 3.14.

### Fixed

- Resolved `TypeError` when calling async methods on mock clients.
- Improved MyPy type fidelity for provider-specific response objects.
- Fixed intermittent CI failures on Windows and MacOS runners.

## [0.1.0] - 2024-12-22

### Added

- Initial release of pytest-mockllm
- Pytest plugin with automatic discovery
- **OpenAI mock** - Chat Completions and Embeddings support
  - Streaming responses with proper SSE format
  - Function calling / tool use support
  - Token usage simulation
- **Anthropic mock** - Claude Messages API support
  - Streaming responses
  - Tool use support
- **Google Gemini mock** - GenerativeAI API support
  - Chat and content generation
  - Streaming support
- **LangChain integration** - Support for LangChain's ChatModel interface
- **Response recording interface** - Initial VCR-like recording and replay interface
- **Cost estimation** - Mock and assert on token usage and API costs
- **Chaos testing** - Simulate rate limits, timeouts, and API errors
- Documentation and examples

### Security

- No external network calls in mock mode (completely isolated testing)

[Unreleased]: https://github.com/godhiraj-code/pytest-mockllm/compare/v0.3.1...HEAD
[0.4.0]: https://github.com/godhiraj-code/pytest-mockllm/compare/v0.3.1...HEAD
[0.3.1]: https://github.com/godhiraj-code/pytest-mockllm/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/godhiraj-code/pytest-mockllm/releases/tag/v0.3.0
[0.2.3]: https://github.com/godhiraj-code/pytest-mockllm/releases/tag/v0.2.3
[0.1.0]: https://github.com/godhiraj-code/pytest-mockllm/releases/tag/v0.1.0
