"""Fixture-scoped google-genai content generation, streaming and chat mocking."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from dataclasses import replace
from typing import Any
from unittest.mock import patch

from pytest_mockllm.core import MockError, MockResponse
from pytest_mockllm.providers._gemini_legacy import _LEGACY_ACTIVE, LegacyGeminiMock


class GeminiMock(LegacyGeminiMock):
    """Mock google.genai.Client and its async models and chat interfaces.

    The optional deprecated google.generativeai adapter is retained for one
    migration cycle. New responses use real google.genai.types models.
    """

    def _raise_provider_error(self, error: MockError) -> None:
        if _LEGACY_ACTIVE.get():
            return super()._raise_provider_error(error)
        from google.genai import errors

        code, status = {
            "rate_limit": (429, "RESOURCE_EXHAUSTED"),
            "auth": (401, "UNAUTHENTICATED"),
            "timeout": (504, "DEADLINE_EXCEEDED"),
            "server": (500, "INTERNAL"),
            "invalid_request": (400, "INVALID_ARGUMENT"),
        }.get(error.error_type, (500, "INTERNAL"))
        cls = errors.ClientError if code < 500 else errors.ServerError
        raise cls(code, {"error": {"code": code, "status": status, "message": error.message}})

    def _build_genai_response(self, response: MockResponse, model: str, config: Any = None) -> Any:
        from google.genai import types

        parts = [types.Part(text=response.content)] if response.content else []
        for tool in response.tool_calls or []:
            function = tool["function"]
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            parts.append(
                types.Part(function_call=types.FunctionCall(name=function["name"], args=arguments))
            )
        if not parts:
            parts = [types.Part(text="")]
        usage = response.token_usage
        result = types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=types.Content(role="model", parts=parts),
                    finish_reason=types.FinishReason.STOP,
                    index=0,
                )
            ],
            model_version=model,
            usage_metadata=types.GenerateContentResponseUsageMetadata(
                prompt_token_count=usage.prompt_tokens if usage else 0,
                candidates_token_count=usage.completion_tokens if usage else 0,
                total_token_count=usage.total_tokens if usage else 0,
            ),
        )
        schema = (
            config.get("response_schema")
            if isinstance(config, dict)
            else getattr(config, "response_schema", None)
        )
        if schema is not None:
            result.parsed = (
                schema.model_validate_json(response.content)
                if hasattr(schema, "model_validate_json")
                else json.loads(response.content)
            )
        return result

    def _genai_response(
        self, model: str, contents: Any, config: Any, stream: bool = False
    ) -> MockResponse:
        self._record_call(
            type="generate_content_stream" if stream else "generate_content",
            model=model,
            contents=contents,
            config=config,
            stream=stream,
        )
        return self._get_next_response(model=model)

    def _genai_chunks(self, response: MockResponse, model: str) -> Iterator[Any]:
        chunks = response.stream_chunks
        if chunks is None:
            words = response.content.split()
            chunks = [word + " " for word in words[:-1]] + [words[-1]] if words else [""]
        for index, text in enumerate(chunks):
            last = index == len(chunks) - 1
            result = self._build_genai_response(
                replace(response, content=text, tool_calls=response.tool_calls if last else None),
                model,
            )
            if not last:
                result.candidates[0].finish_reason = None
                result.usage_metadata = None
            yield result

    def __enter__(self) -> GeminiMock:
        try:
            super().__enter__()

            def generate(_instance: Any, *, model: str, contents: Any, config: Any = None) -> Any:
                response = self._genai_response(model, contents, config)
                self._simulate_delay(self._get_delay_ms(response))
                return self._build_genai_response(response, model, config)

            async def async_generate(
                _instance: Any, *, model: str, contents: Any, config: Any = None
            ) -> Any:
                response = self._genai_response(model, contents, config)
                await self._simulate_delay_async(self._get_delay_ms(response))
                return self._build_genai_response(response, model, config)

            def stream(
                _instance: Any, *, model: str, contents: Any, config: Any = None
            ) -> Iterator[Any]:
                response = self._genai_response(model, contents, config, stream=True)
                self._simulate_delay(self._get_delay_ms(response))
                yield from self._genai_chunks(response, model)

            async def async_stream(
                _instance: Any, *, model: str, contents: Any, config: Any = None
            ) -> AsyncIterator[Any]:
                response = self._genai_response(model, contents, config, stream=True)
                await self._simulate_delay_async(self._get_delay_ms(response))

                async def chunks() -> AsyncIterator[Any]:
                    for chunk in self._genai_chunks(response, model):
                        yield chunk

                return chunks()

            def blocked(*args: Any, **kwargs: Any) -> Any:
                raise RuntimeError("Gemini network blocked by active mock_gemini fixture")

            async def async_blocked(*args: Any, **kwargs: Any) -> Any:
                raise RuntimeError("Gemini network blocked by active mock_gemini fixture")

            for target, replacement in [
                ("google.genai.models.Models.generate_content", generate),
                ("google.genai.models.Models.generate_content_stream", stream),
                ("google.genai.models.AsyncModels.generate_content", async_generate),
                ("google.genai.models.AsyncModels.generate_content_stream", async_stream),
                ("google.genai._api_client.BaseApiClient.request", blocked),
                ("google.genai._api_client.BaseApiClient.request_streamed", blocked),
                ("google.genai._api_client.BaseApiClient.async_request", async_blocked),
                ("google.genai._api_client.BaseApiClient.async_request_streamed", async_blocked),
            ]:
                patcher = patch(target, new=replacement)
                patcher.start()
                self._patches.append(patcher)
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise
