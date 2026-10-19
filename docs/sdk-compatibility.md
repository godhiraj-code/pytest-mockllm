# SDK compatibility and validation

Checked on 2026-10-19 using PyPI metadata and the installed SDK source.

| Package | Current version tested | Minimum version tested |
| --- | --- | --- |
| [openai](https://pypi.org/project/openai/3.35.1/) | 3.35.1 | 1.109.1 |
| [anthropic](https://pypi.org/project/anthropic/1.15.0/) | 1.15.0 | 0.75.0 |
| [google-genai](https://pypi.org/project/google-genai/2.31.0/) | 2.31.0 | 1.52.0 |
| [langchain-core](https://pypi.org/project/langchain-core/1.6.12/) | 1.6.12 | 0.3.80 |
| [langchain-openai](https://pypi.org/project/langchain-openai/1.7.2/) | 1.7.2 | 0.3.35 |

Provider dependency floors reflect the older SDK set actually exercised here. This
is not a claim that every version between the endpoints has been tested. OpenAI and
Anthropic extras require Pydantic 2 for typed response/stream validation. All provider
SDKs remain optional. The deprecated google-generativeai adapter is tested separately
with 0.8.6 and issues a deprecation warning on model construction.

## Reproducible checks

```bash
pip install -e ".[all,dev]"
pytest
ruff check src/
mypy src/pytest_mockllm --ignore-missing-imports --python-version 3.11 --follow-imports=silent
```

The full suite was run locally on Windows with Python 3.10, 3.11, 3.12, and 3.13.
The optional legacy Gemini module skips when google-generativeai is absent. The
older SDK set and legacy adapter were also tested on Python 3.12. The existing
cross-platform CI matrix covers Linux, macOS, and Windows; CI results, not a local
Windows run, establish those other platform results.

The CI minimum-SDK job installs:

```bash
pip install -e ".[all,dev,google-legacy]" \
  "openai==1.109.1" "anthropic==0.75.0" "google-genai==1.52.0" \
  "langchain-core==0.3.80" "langchain-openai==0.3.35"
pytest
```

## Evidence and limits

- [Recording tests](../tests/test_recording.py) exercise actual SDK parsing with
  controlled HTTP responses: sync/async JSON, SSE text/tool fragments, SDK retries,
  error types, credential redaction, all four modes, matching, exhaustion, file
  replacement, cleanup, and incomplete reads. Local HTTP-server tests record over
  a real socket and replay after the server stops with socket connections blocked.
- [SDK regressions](../tests/acceptance/test_latest_sdk.py) exercise Responses API
  text/usage/function calls, typed events, stream managers, parsed final output,
  nonblocking async delay, provider errors, and an OpenAI-compatible base URL.
- [Gemini tests](../tests/test_gemini.py) exercise google-genai typed responses,
  chat state, sync/async streams, tools, structured output, strict mode, and errors.
- [Plugin tests](../tests/test_plugin.py) cover CLI modes, automatic markers,
  cassette names, mock/recorder conflicts, strict mode, and opt-in statistics.
- [README tests](../tests/test_readme.py) execute the offline Python examples.
  [Demo tests](../examples/test_demo.py) are part of the normal suite.

No paid/live provider calls were used for this validation. Passing these tests does
not establish that a model exists, your account can access it, or a live API accepts
every parameter combination. Model names in mocked examples are illustrative.

Record/replay uses private SDK request-building hooks plus HTTPX/HTTPX2 send hooks.
The compatibility suites are intended to catch upstream changes to those boundaries.
It buffers full UTF-8 JSON/SSE bodies, so it does not reproduce chunk arrival timing
or partial HTTP disconnects. It rejects unsupported SDK endpoints, non-JSON requests,
and custom HTTP clients overriding send. It is not a network sandbox for WebSockets,
other providers, or unrelated HTTP calls. Concurrent writes to the same cassette
path are unsupported; per-test paths include the full pytest ID's hash.

Handwritten mocks cover the documented interfaces, not entire SDKs. In particular,
Gemini automatic tool execution and LangChain transport/callback behavior are not
emulated. Token/cost totals are estimates based on configured responses and a static
price table. Assertion helpers, LiteLLM-specific integration, and an environment-gated
live-test marker remain future work.
