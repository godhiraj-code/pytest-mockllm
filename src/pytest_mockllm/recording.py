"""
VCR-style recording and replay for LLM API calls.

Allows tests to record real API responses and replay them later,
enabling "golden test" patterns without repeated API costs.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
import time
import warnings
from contextlib import ExitStack
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import yaml


@dataclass
class RecordedInteraction:
    """A single recorded LLM API interaction."""

    request: dict[str, Any]
    response: dict[str, Any]
    provider: str
    model: str
    timestamp: float = field(default_factory=time.time)
    latency_ms: float = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": self.request,
            "response": self.response,
            "provider": self.provider,
            "model": self.model,
            "timestamp": self.timestamp,
            "latency_ms": self.latency_ms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RecordedInteraction:
        return cls(
            request=data["request"],
            response=data["response"],
            provider=data["provider"],
            model=data["model"],
            timestamp=data.get("timestamp", 0),
            latency_ms=data.get("latency_ms", 0),
        )


@dataclass
class Cassette:
    """A collection of recorded interactions for a test."""

    name: str
    interactions: list[RecordedInteraction] = field(default_factory=list)
    version: str = "2.0"
    created: float = field(default_factory=time.time)

    def to_dict(self, redact: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {
            "name": self.name,
            "version": self.version,
            "created": self.created,
            "interactions": [i.to_dict() for i in self.interactions],
        }
        if redact:
            for interaction in data["interactions"]:
                for side in ("request", "response"):
                    interaction[side] = PIIRedactor.redact_http(interaction[side])
            return PIIRedactor.redact_dict(data)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Cassette:
        return cls(
            name=data["name"],
            version=data.get("version", "1.0"),
            created=data.get("created", 0),
            interactions=[RecordedInteraction.from_dict(i) for i in data.get("interactions", [])],
        )

    def save(self, path: Path) -> None:
        """Save cassette to YAML file with automatic PII redaction."""
        path.parent.mkdir(parents=True, exist_ok=True)
        # Always redact on save for security
        data = self.to_dict(redact=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
            ) as f:
                temporary = Path(f.name)
                yaml.safe_dump(
                    data, f, default_flow_style=False, sort_keys=False, allow_unicode=True
                )
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: Path) -> Cassette:
        """Load cassette from YAML file."""
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)


class PIIRedactor:
    """Class to redact PII (API keys, etc.) from recorded interactions."""

    # Patterns for common API keys and sensitive tokens
    PATTERNS = [
        # Match Anthropic before the broader OpenAI prefix.
        (
            re.compile(r"(?:sk-ant-|ant-api-)[a-zA-Z0-9_-]{20,}", re.IGNORECASE),
            "[REDACTED_ANTHROPIC_KEY]",
        ),
        (re.compile(r"sk-(?!ant-)[a-zA-Z0-9_-]{20,}", re.IGNORECASE), "[REDACTED_OPENAI_KEY]"),
        # Azure OpenAI
        (re.compile(r"api-key: [a-zA-Z0-9]{32}", re.IGNORECASE), "api-key: [REDACTED]"),
        # Google/GCP keys: AIza...
        (re.compile(r"AIza[0-9A-Za-z_-]{35}", re.IGNORECASE), "[REDACTED_GCP_KEY]"),
        # Generic Bearer tokens
        (re.compile(r"Bearer [a-zA-Z0-9._~+/-]+=*", re.IGNORECASE), "Bearer [REDACTED]"),
    ]

    SENSITIVE_HEADERS = {
        "api_key",
        "api-key",
        "authorization",
        "x-api-key",
        "cookie",
        "set-cookie",
        "proxy-authorization",
        "x-goog-api-key",
    }
    SENSITIVE_QUERY_PARAMS = {
        "key", "api_key", "api-key", "apikey", "access_token", "auth_token",
        "token", "client_secret", "signature", "sig",
    }
    # Retained for callers that inspect the old attribute; never applied to JSON keys.
    SENSITIVE_KEYS = SENSITIVE_HEADERS | SENSITIVE_QUERY_PARAMS

    @classmethod
    def redact_headers(cls, headers: dict[str, Any]) -> dict[str, Any]:
        """Redact authentication headers without interpreting JSON field names."""
        return {
            key: "[REDACTED]" if key.lower() in cls.SENSITIVE_HEADERS else cls.redact_dict(value)
            for key, value in headers.items()
        }

    @classmethod
    def redact_url(cls, value: str) -> str:
        """Remove URL credentials and sensitive query parameters."""
        url = urlsplit(value)
        query = [
            (key, "[REDACTED]" if key.lower() in cls.SENSITIVE_QUERY_PARAMS else cls.redact_text(val))
            for key, val in parse_qsl(url.query, keep_blank_values=True)
        ]
        return urlunsplit(
            (url.scheme, url.netloc.rsplit("@", 1)[-1], url.path, urlencode(sorted(query)), "")
        )

    @classmethod
    def redact_http(cls, message: dict[str, Any]) -> dict[str, Any]:
        """Redact an HTTP envelope; nested body objects retain their field names."""
        result = cls.redact_dict(message)
        if isinstance(message.get("headers"), dict):
            result["headers"] = cls.redact_headers(message["headers"])
        if isinstance(message.get("url"), str):
            result["url"] = cls.redact_url(message["url"])
        return result

    @classmethod
    def redact_dict(cls, data: Any) -> Any:
        """Redact recognizable secret values, preserving arbitrary JSON schemas."""
        if isinstance(data, dict):
            return {k: cls.redact_dict(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [cls.redact_dict(item) for item in data]
        elif isinstance(data, str):
            return cls.redact_text(data)
        else:
            return data

    @classmethod
    def redact_text(cls, text: str) -> str:
        """Redact sensitive patterns within a string."""
        for pattern, replacement in cls.PATTERNS:
            text = pattern.sub(replacement, text)
        return text


class RecordingError(RuntimeError):
    """A cassette cannot safely satisfy the requested SDK call."""


class CassetteMissError(RecordingError):
    """No unused interaction matches, and the mode forbids recording."""


_active: ContextVar[bool] = ContextVar("mockllm_recorder_active", default=False)
_recorder_lock = threading.Lock()


class LLMRecorder:
    """Record/replay JSON and SSE HTTP responses from OpenAI and Anthropic.

    ``record_mode`` is ``none`` (offline, the default), ``once`` (record only
    when the file does not exist), ``new_episodes`` (append misses), or ``all``
    (replace). Each recorded request can be replayed once per context entry.
    Recording buffers complete responses, including SSE, before returning them.
    SDK parsers and stream managers run normally during both record and replay.

    The legacy ``mode`` argument remains supported with a deprecation warning;
    notably legacy ``mode='none'`` retains its explicit pass-through behavior.
    """

    def __init__(
        self,
        cassette_path: str | Path,
        mode: str | None = None,
        *,
        record_mode: str | None = None,
    ) -> None:
        if mode is not None and record_mode is not None:
            raise ValueError("Use record_mode or legacy mode, not both")
        if mode is not None:
            aliases = {"auto": "once", "record": "all", "replay": "none", "none": "passthrough"}
            if mode not in aliases:
                raise ValueError(
                    "Legacy mode must be auto, record, replay, or none; use record_mode"
                )
            warnings.warn(
                "LLMRecorder(mode=...) is deprecated; use record_mode='none', 'once', "
                "'new_episodes', or 'all'. Legacy mode='none' still passes through live calls.",
                DeprecationWarning,
                stacklevel=2,
            )
            record_mode = aliases[mode]
        elif record_mode is not None and record_mode not in {"none", "once", "new_episodes", "all"}:
            raise ValueError(f"Unknown record_mode: {record_mode!r}")
        self.cassette_path = Path(cassette_path)
        self.mode = mode if mode is not None else (record_mode or "none")
        self.record_mode = record_mode or "none"
        self._cassette: Cassette | None = None
        self._interaction_index = 0
        self._used: set[int] = set()
        self._recording = False
        self._dirty = False
        self._stack: ExitStack | None = None
        self._lock = threading.RLock()
        self._secrets: set[str] = set()

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def cassette(self) -> Cassette | None:
        return self._cassette

    def __enter__(self) -> LLMRecorder:
        if self._stack is not None or _active.get():
            raise RecordingError("Nested or overlapping LLMRecorder contexts are not supported")
        if not _recorder_lock.acquire(blocking=False):
            raise RecordingError("Nested or overlapping LLMRecorder contexts are not supported")
        self._stack = ExitStack()
        self._stack.callback(_recorder_lock.release)
        token = _active.set(True)
        self._stack.callback(_active.reset, token)
        try:
            if self.record_mode == "passthrough":
                return self
            exists = self.cassette_path.exists()
            self._recording = self.record_mode in {"all", "new_episodes"} or (
                self.record_mode == "once" and not exists
            )
            self._cassette = (
                Cassette.load(self.cassette_path)
                if exists and self.record_mode != "all"
                else Cassette(name=self.cassette_path.stem)
            )
            if self._cassette.version != "2.0":
                raise RecordingError(
                    "Unsupported cassette version; record a new cassette with --llm-record=all"
                )
            self._used.clear()
            self._secrets.clear()
            self._interaction_index = 0
            self._dirty = False
            from pytest_mockllm._recording_http import install

            install(self, self._stack)
            return self
        except BaseException:
            self._stack.close()
            self._stack = None
            raise

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._stack is not None:
            self._stack.close()
            self._stack = None
        if self._dirty and self._cassette is not None:
            # Scrub again so secrets learned from later requests cannot leak from
            # earlier echoed responses. Atomic replacement preserves old files.
            self._cassette = Cassette.from_dict(self._redact(self._cassette.to_dict()))
            self._cassette.save(self.cassette_path)

    def _redact(self, value: Any) -> Any:
        value = PIIRedactor.redact_dict(value)
        if isinstance(value, dict):
            return {k: self._redact(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._redact(v) for v in value]
        if isinstance(value, str):
            for secret in sorted(self._secrets, key=len, reverse=True):
                value = value.replace(secret, "[REDACTED]")
        return value

    def remember_credentials(self, client: Any) -> None:
        for name in ("api_key", "auth_token"):
            value = getattr(client, name, None)
            if isinstance(value, str) and value:
                self._secrets.add(value)

    def _request(self, provider: str, request: Any) -> dict[str, Any]:
        url = urlsplit(str(request.url))
        endpoints = {
            "openai": ("/chat/completions", "/responses", "/embeddings"),
            "anthropic": ("/messages",),
        }
        if request.method != "POST" or not url.path.rstrip("/").endswith(endpoints[provider]):
            raise RecordingError(f"Unsupported {provider} endpoint for recording/replay")
        try:
            body = json.loads(request.content)
        except (ValueError, UnicodeError) as error:
            raise RecordingError("Recording supports JSON request bodies only") from error
        headers = dict(request.headers)
        query = parse_qsl(url.query, keep_blank_values=True)
        for values, sensitive in (
            (headers.items(), PIIRedactor.SENSITIVE_HEADERS),
            (query, PIIRedactor.SENSITIVE_QUERY_PARAMS),
        ):
            for key, value in values:
                if key.lower() in sensitive and value:
                    self._secrets.add(value)
                    if value.lower().startswith("bearer "):
                        self._secrets.add(value[7:])
        clean_url = PIIRedactor.redact_url(str(request.url))
        return self._redact(
            {
                "method": request.method,
                "url": clean_url,
                "headers": PIIRedactor.redact_headers(headers),
                "body": body,
            }
        )

    def _hash_request(self, request: dict[str, Any]) -> str:
        semantic_headers = {
            "content-type",
            "anthropic-version",
            "anthropic-beta",
            "openai-beta",
            "openai-organization",
            "openai-project",
        }
        normalized = {
            **request,
            "headers": {
                k: v for k, v in request.get("headers", {}).items() if k.lower() in semantic_headers
            },
        }
        return hashlib.sha256(json.dumps(normalized, sort_keys=True).encode()).hexdigest()

    def prepare(
        self, provider: str, request: Any, *, consume: bool = True
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        with self._lock:
            prepared = self._request(provider, request)
            assert self._cassette is not None
            if self.record_mode != "all":
                wanted = self._hash_request(prepared)
                for index, interaction in enumerate(self._cassette.interactions):
                    if (
                        index not in self._used
                        and interaction.provider == provider
                        and self._hash_request(interaction.request) == wanted
                    ):
                        if consume:
                            self._used.add(index)
                        return prepared, interaction.response
            if not self._recording:
                raise CassetteMissError(
                    f"No unused {provider} interaction matches in {self.cassette_path}. "
                    "Network access refused. Record explicitly with --llm-record=once "
                    "(new file), new_episodes (append), or all (replace)."
                )
            return prepared, None

    def capture(self, provider: str, request: dict[str, Any], response: Any) -> None:
        """Store a completed HTTP response; incomplete reads never reach this method."""
        content_type = response.headers.get("content-type", "")
        text = response.content.decode("utf-8")
        if "json" in content_type:
            body = self._redact(json.loads(text))
            encoding = "json"
        else:
            # SSE data may contain JSON credentials unrelated to the actual key.
            lines = []
            for line in text.splitlines(keepends=True):
                if line.startswith("data:"):
                    try:
                        data = self._redact(json.loads(line[5:].strip()))
                        line = "data: " + json.dumps(data, ensure_ascii=False) + "\n"
                    except ValueError:
                        pass
                lines.append(line)
            body = self._redact("".join(lines))
            encoding = "text"
        headers = {
            k: v
            for k, v in response.headers.items()
            if k.lower() not in {"content-length", "content-encoding", "transfer-encoding"}
        }
        self.record_interaction(
            request,
            {
                "status_code": response.status_code,
                "headers": PIIRedactor.redact_headers(headers),
                "encoding": encoding,
                "body": body,
            },
            provider,
            str(request["body"].get("model", "")),
        )

    @staticmethod
    def http_response(http: Any, request: Any, replay: dict[str, Any]) -> Any:
        body = replay["body"]
        content = json.dumps(body, ensure_ascii=False) if replay["encoding"] == "json" else body
        return http.Response(
            replay["status_code"],
            headers=replay["headers"],
            content=content.encode("utf-8"),
            request=request,
        )

    def record_interaction(
        self,
        request: dict[str, Any],
        response: dict[str, Any],
        provider: str,
        model: str,
        latency_ms: float = 0,
    ) -> None:
        with self._lock:
            if self._cassette is None:
                self._cassette = Cassette(name=self.cassette_path.stem)
            self._used.add(len(self._cassette.interactions))
            self._cassette.interactions.append(
                RecordedInteraction(
                    self._redact(request),
                    self._redact(response),
                    provider,
                    model,
                    latency_ms=latency_ms,
                )
            )
            self._dirty = True

    def get_next_replay(self) -> RecordedInteraction | None:
        if self._cassette is None or self._interaction_index >= len(self._cassette.interactions):
            return None
        interaction = self._cassette.interactions[self._interaction_index]
        self._interaction_index += 1
        return interaction

    def clear(self) -> None:
        self._cassette = Cassette(name=self.cassette_path.stem)
        self._interaction_index = 0
        self._used.clear()
        self._dirty = False
        self.cassette_path.unlink(missing_ok=True)
