"""SDK-scoped HTTP interception. Provider SDKs still own parsing and SSE decoding."""

from __future__ import annotations

import importlib
from contextvars import ContextVar
from functools import wraps
from typing import Any
from unittest.mock import patch

from pytest_mockllm.recording import RecordingError

_provider: ContextVar[str | None] = ContextVar("mockllm_provider", default=None)


def _unwrap(error: BaseException) -> None:
    """SDKs may wrap transport failures; preserve actionable cassette errors."""
    cause: BaseException | None = error
    while cause is not None:
        if isinstance(cause, RecordingError):
            raise cause from None
        cause = cause.__cause__


def install(recorder: Any, stack: Any) -> None:
    """Patch only installed SDKs; unrelated HTTP traffic passes through unchanged."""

    def validate_client(client: Any) -> None:
        if not getattr(client._client.send, "_mockllm_intercepted", False):
            raise RecordingError(
                "Custom HTTP clients overriding send are unsupported; use a custom transport instead"
            )

    for name in ("openai", "anthropic"):
        try:
            sdk = importlib.import_module(f"{name}._base_client")
        except ImportError:
            continue
        for asynchronous, cls in ((False, sdk.SyncAPIClient), (True, sdk.AsyncAPIClient)):
            original = cls.request

            def scoped(original: Any, name: str, asynchronous: bool) -> Any:
                @wraps(original)
                def sync_call(*args: Any, **kwargs: Any) -> Any:
                    validate_client(args[0])
                    recorder.remember_credentials(args[0])
                    token = _provider.set(name)
                    try:
                        return original(*args, **kwargs)
                    except Exception as error:
                        _unwrap(error)
                        raise
                    finally:
                        _provider.reset(token)

                @wraps(original)
                async def async_call(*args: Any, **kwargs: Any) -> Any:
                    validate_client(args[0])
                    recorder.remember_credentials(args[0])
                    token = _provider.set(name)
                    try:
                        return await original(*args, **kwargs)
                    except Exception as error:
                        _unwrap(error)
                        raise
                    finally:
                        _provider.reset(token)

                return async_call if asynchronous else sync_call

            stack.enter_context(patch.object(cls, "request", scoped(original, name, asynchronous)))

            # Validate before the SDK's transport exception handler/retry loop.
            # This also catches unsupported endpoints before any network access.
            build = cls._build_request

            def checked(build: Any) -> Any:
                @wraps(build)
                def build_request(*args: Any, **kwargs: Any) -> Any:
                    request = build(*args, **kwargs)
                    provider = _provider.get()
                    if provider is not None:
                        recorder.prepare(provider, request, consume=False)
                    return request

                return build_request

            stack.enter_context(patch.object(cls, "_build_request", checked(build)))

    for module_name in ("httpx", "httpx2"):
        try:
            http = importlib.import_module(module_name)
        except ImportError:
            continue
        for asynchronous, cls in ((False, http.Client), (True, http.AsyncClient)):
            original = cls.send

            def intercept(original: Any, http: Any, asynchronous: bool) -> Any:
                @wraps(original)
                def send(client: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
                    provider = _provider.get()
                    if provider is None:
                        return original(client, request, *args, **kwargs)
                    prepared, replay = recorder.prepare(provider, request)
                    if replay is not None:
                        return recorder.http_response(http, request, replay)
                    response = original(client, request, *args, **kwargs)
                    try:
                        response.read()
                        recorder.capture(provider, prepared, response)
                    except BaseException:
                        response.close()
                        raise
                    return response

                @wraps(original)
                async def async_send(client: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
                    provider = _provider.get()
                    if provider is None:
                        return await original(client, request, *args, **kwargs)
                    prepared, replay = recorder.prepare(provider, request)
                    if replay is not None:
                        return recorder.http_response(http, request, replay)
                    response = await original(client, request, *args, **kwargs)
                    try:
                        await response.aread()
                        recorder.capture(provider, prepared, response)
                    except BaseException:
                        await response.aclose()
                        raise
                    return response

                replacement: Any = async_send if asynchronous else send
                replacement._mockllm_intercepted = True
                return replacement

            stack.enter_context(patch.object(cls, "send", intercept(original, http, asynchronous)))
