"""
VCR-style recording and replay for LLM API calls.

Allows tests to record real API responses and replay them later,
enabling "golden test" patterns without repeated API costs.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    version: str = "1.0"
    created: float = field(default_factory=time.time)

    def to_dict(self, redact: bool = False) -> dict[str, Any]:
        data = {
            "name": self.name,
            "version": self.version,
            "created": self.created,
            "interactions": [i.to_dict() for i in self.interactions],
        }
        if redact:
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
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

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
        # OpenAI keys: sk-...
        (re.compile(r"sk-[a-zA-Z0-9]{20,}", re.IGNORECASE), "[REDACTED_OPENAI_KEY]"),
        # Anthropic keys: ant-api-...
        (re.compile(r"ant-api-[a-zA-Z0-9\-_]{20,}", re.IGNORECASE), "[REDACTED_ANTHROPIC_KEY]"),
        # Azure OpenAI
        (re.compile(r"api-key: [a-zA-Z0-9]{32}", re.IGNORECASE), "api-key: [REDACTED]"),
        # Google/GCP keys: AIza...
        (re.compile(r"AIza[0-9A-Za-z-_]{35}", re.IGNORECASE), "[REDACTED_GCP_KEY]"),
        # Generic Bearer tokens
        (re.compile(r"Bearer [a-zA-Z0-9\.\-_]{20,}", re.IGNORECASE), "Bearer [REDACTED]"),
    ]

    SENSITIVE_KEYS = {
        "api_key",
        "api-key",
        "authorization",
        "x-api-key",
        "token",
        "access_token",
        "secret",
        "auth",
    }

    @classmethod
    def redact_dict(cls, data: Any) -> Any:
        """Deeply redact a dictionary or list of any sensitive information."""
        if isinstance(data, dict):
            redacted = {}
            for k, v in data.items():
                if str(k).lower() in cls.SENSITIVE_KEYS:
                    redacted[k] = "[REDACTED]"
                else:
                    redacted[k] = cls.redact_dict(v)
            return redacted
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


class LLMRecorder:
    """
    Record and replay LLM API calls.

    Modes:
        - "auto": Replay if cassette exists, otherwise record
        - "record": Always record (overwrites existing cassettes)
        - "replay": Always replay (fails if cassette missing)
        - "none": Disable recording (pass through to real APIs)

    Example:
        >>> recorder = LLMRecorder(cassette_path=Path("tests/cassettes/my_test.yaml"))
        >>> with recorder:
        ...     # First run: records real API call
        ...     # Later runs: replays recorded response
        ...     response = openai_client.chat.completions.create(...)
    """

    def __init__(
        self,
        cassette_path: str | Path,
        mode: str = "auto",
    ) -> None:
        self.cassette_path = Path(cassette_path)
        self.mode = mode
        self._cassette: Cassette | None = None
        self._interaction_index: int = 0
        self._patches: list[Any] = []
        self._recording: bool = False

    @property
    def is_recording(self) -> bool:
        """Check if currently in recording mode."""
        return self._recording

    @property
    def cassette(self) -> Cassette | None:
        """Access the current cassette."""
        return self._cassette

    def _should_record(self) -> bool:
        """Determine if we should record or replay."""
        if self.mode == "record":
            return True
        elif self.mode == "replay":
            if not self.cassette_path.exists():
                raise FileNotFoundError(
                    f"Cassette not found: {self.cassette_path}. "
                    "Run with --llm-record or @pytest.mark.llm_record to record first."
                )
            return False
        elif self.mode == "auto":
            return not self.cassette_path.exists()
        else:  # mode == "none"
            return False

    def _hash_request(self, request: dict[str, Any]) -> str:
        """Create a hash of the request for matching."""
        # Normalize the request for hashing
        normalized = json.dumps(request, sort_keys=True, default=str)
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def record_interaction(
        self,
        request: dict[str, Any],
        response: dict[str, Any],
        provider: str,
        model: str,
        latency_ms: float = 0,
    ) -> None:
        """Record a new interaction."""
        if self._cassette is None:
            self._cassette = Cassette(name=self.cassette_path.stem)

        interaction = RecordedInteraction(
            request=request,
            response=response,
            provider=provider,
            model=model,
            latency_ms=latency_ms,
        )
        self._cassette.interactions.append(interaction)

    def get_next_replay(self) -> RecordedInteraction | None:
        """Get the next recorded interaction for replay."""
        if self._cassette is None or self._interaction_index >= len(self._cassette.interactions):
            return None

        interaction = self._cassette.interactions[self._interaction_index]
        self._interaction_index += 1
        return interaction

    def __enter__(self) -> LLMRecorder:
        """Start recording/replaying."""
        self._recording = self._should_record()

        if not self._recording and self.cassette_path.exists():
            # Load existing cassette for replay
            self._cassette = Cassette.load(self.cassette_path)
            self._interaction_index = 0
        elif self._recording:
            # Start fresh cassette
            self._cassette = Cassette(name=self.cassette_path.stem)

        if not self._recording:
            # Set up replay patches
            self._setup_replay_patches()

        return self

    def _setup_replay_patches(self) -> None:
        """Set up patches to replay recorded responses."""
        # For now, the replay is handled by the individual provider mocks
        # This can be extended to intercept real API calls
        pass

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Stop recording and save cassette if needed."""
        # Stop all patches
        for p in reversed(self._patches):
            p.stop()
        self._patches.clear()

        # Save cassette if we were recording
        if self._recording and self._cassette and self._cassette.interactions:
            self._cassette.save(self.cassette_path)

    def clear(self) -> None:
        """Clear all recorded interactions."""
        self._cassette = None
        self._interaction_index = 0

        # Delete cassette file if it exists
        if self.cassette_path.exists():
            self.cassette_path.unlink()
