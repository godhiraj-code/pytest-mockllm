# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/godhiraj-code/pytest-mockllm/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/godhiraj-code/pytest-mockllm/releases/tag/v0.1.0
