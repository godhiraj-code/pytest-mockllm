"""
Core mock classes and response types for pytest-mockllm.

This module contains the foundational classes that all provider-specific
mocks inherit from, ensuring a consistent API across providers.
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional, Union
from unittest.mock import MagicMock, patch


@dataclass
class TokenUsage:
    """Token usage statistics for a mock response."""
    
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    def __post_init__(self) -> None:
        if self.total_tokens == 0:
            self.total_tokens = self.prompt_tokens + self.completion_tokens


@dataclass
class MockResponse:
    """
    A configurable mock response for LLM APIs.
    
    This provides a provider-agnostic way to define responses that
    will be converted to the appropriate format for each provider.
    
    Examples:
        >>> response = MockResponse(content="Hello, world!")
        >>> response = MockResponse(
        ...     content="Let me help with that.",
        ...     token_usage=TokenUsage(prompt_tokens=10, completion_tokens=20),
        ...     latency_ms=100,
        ... )
    """
    
    content: str
    role: str = "assistant"
    model: str = "mock-model"
    token_usage: Optional[TokenUsage] = None
    finish_reason: str = "stop"
    latency_ms: int = 0
    id: str = field(default_factory=lambda: f"mock-{uuid.uuid4().hex[:8]}")
    
    # For streaming responses
    stream_chunks: Optional[list[str]] = None
    
    # For function/tool calling
    tool_calls: Optional[list[dict[str, Any]]] = None
    function_call: Optional[dict[str, Any]] = None
    
    def __post_init__(self) -> None:
        if self.token_usage is None:
            # Estimate tokens (rough approximation: ~4 chars per token)
            estimated_tokens = len(self.content) // 4 + 1
            self.token_usage = TokenUsage(
                prompt_tokens=10,  # Assume minimal prompt
                completion_tokens=estimated_tokens,
            )


@dataclass 
class MockError:
    """Configuration for simulating API errors."""
    
    error_type: str  # "rate_limit", "timeout", "auth", "server", "invalid_request"
    message: str = ""
    after_calls: int = 0  # Trigger after N successful calls (0 = immediate)
    
    def __post_init__(self) -> None:
        error_messages = {
            "rate_limit": "Rate limit exceeded. Please retry after 60 seconds.",
            "timeout": "Request timed out.",
            "auth": "Invalid API key provided.",
            "server": "Internal server error.",
            "invalid_request": "Invalid request parameters.",
        }
        if not self.message:
            self.message = error_messages.get(self.error_type, "Unknown error")


class MockLLM(ABC):
    """
    Abstract base class for LLM mocks.
    
    All provider-specific mocks (OpenAI, Anthropic, etc.) inherit from this
    class to ensure a consistent API.
    
    Features:
        - Response queue management
        - Call history tracking
        - Token usage accumulation
        - Error simulation
        - Latency simulation
    """
    
    def __init__(self) -> None:
        self._responses: list[MockResponse] = []
        self._response_index: int = 0
        self._calls: list[dict[str, Any]] = []
        self._patches: list[Any] = []
        self._error: Optional[MockError] = None
        self._call_count: int = 0
        self._total_tokens: int = 0
        self._total_prompt_tokens: int = 0
        self._total_completion_tokens: int = 0
        self._default_response: Optional[MockResponse] = None
        self._strict_mode: bool = False
    
    @property
    def calls(self) -> list[dict[str, Any]]:
        """All recorded API calls made to this mock."""
        return self._calls
    
    @property
    def call_count(self) -> int:
        """Number of API calls made to this mock."""
        return self._call_count
    
    @property
    def last_call(self) -> Optional[dict[str, Any]]:
        """The most recent API call, or None if no calls made."""
        return self._calls[-1] if self._calls else None
    
    @property
    def total_tokens(self) -> int:
        """Total tokens used across all calls."""
        return self._total_tokens
    
    @property
    def total_prompt_tokens(self) -> int:
        """Total prompt tokens used across all calls."""
        return self._total_prompt_tokens
    
    @property
    def total_completion_tokens(self) -> int:
        """Total completion tokens used across all calls."""
        return self._total_completion_tokens
    
    def add_response(
        self,
        content: str,
        *,
        model: str = "mock-model",
        token_usage: Optional[TokenUsage] = None,
        latency_ms: int = 0,
        tool_calls: Optional[list[dict[str, Any]]] = None,
        stream_chunks: Optional[list[str]] = None,
    ) -> "MockLLM":
        """
        Add a response to the queue.
        
        Responses are returned in order. When exhausted, returns default response
        or raises an error in strict mode.
        
        Args:
            content: The response content/message
            model: Model name to include in response
            token_usage: Custom token usage stats
            latency_ms: Simulated latency in milliseconds
            tool_calls: Tool/function calls to include
            stream_chunks: For streaming, custom chunk splits
            
        Returns:
            self for method chaining
            
        Example:
            >>> mock.add_response("First response").add_response("Second response")
        """
        response = MockResponse(
            content=content,
            model=model,
            token_usage=token_usage,
            latency_ms=latency_ms,
            tool_calls=tool_calls,
            stream_chunks=stream_chunks,
        )
        self._responses.append(response)
        return self
    
    def add_responses(self, *contents: str) -> "MockLLM":
        """
        Add multiple simple text responses at once.
        
        Example:
            >>> mock.add_responses("Hello!", "How can I help?", "Goodbye!")
        """
        for content in contents:
            self.add_response(content)
        return self
    
    def set_default_response(self, content: str, **kwargs: Any) -> "MockLLM":
        """
        Set a default response when the queue is exhausted.
        
        This response will be returned for all calls after the queue is empty,
        unless strict mode is enabled.
        """
        self._default_response = MockResponse(content=content, **kwargs)
        return self
    
    def simulate_error(
        self,
        error_type: str,
        *,
        message: str = "",
        after_calls: int = 0,
    ) -> "MockLLM":
        """
        Configure the mock to simulate an API error.
        
        Args:
            error_type: One of "rate_limit", "timeout", "auth", "server", "invalid_request"
            message: Custom error message (uses default if not provided)
            after_calls: Number of successful calls before error (0 = immediate)
            
        Example:
            >>> mock.simulate_error("rate_limit", after_calls=5)
        """
        self._error = MockError(
            error_type=error_type,
            message=message,
            after_calls=after_calls,
        )
        return self
    
    def set_strict_mode(self, enabled: bool = True) -> "MockLLM":
        """
        Enable strict mode - raises error if no response configured.
        
        In strict mode, tests fail immediately if an unconfigured LLM call is made,
        helping catch missing mock configurations.
        """
        self._strict_mode = enabled
        return self
    
    def _get_next_response(self) -> MockResponse:
        """Get the next response from the queue."""
        # Check for error simulation
        if self._error and self._call_count >= self._error.after_calls:
            self._raise_provider_error(self._error)
        
        # Try to get from queue
        if self._response_index < len(self._responses):
            response = self._responses[self._response_index]
            self._response_index += 1
        elif self._default_response:
            response = self._default_response
        elif self._strict_mode:
            raise RuntimeError(
                f"No mock response configured for call #{self._call_count + 1}. "
                "Use mock.add_response() or mock.set_default_response() to configure responses, "
                "or disable strict mode with mock.set_strict_mode(False)."
            )
        else:
            # Fallback default
            response = MockResponse(content="Mock response from pytest-mockllm")
        
        # Simulate latency
        if response.latency_ms > 0:
            time.sleep(response.latency_ms / 1000.0)
        
        # Track usage
        if response.token_usage:
            self._total_tokens += response.token_usage.total_tokens
            self._total_prompt_tokens += response.token_usage.prompt_tokens
            self._total_completion_tokens += response.token_usage.completion_tokens
        
        self._call_count += 1
        return response
    
    def _record_call(self, **kwargs: Any) -> None:
        """Record an API call for later inspection."""
        self._calls.append({
            "call_number": self._call_count + 1,
            "timestamp": time.time(),
            **kwargs,
        })
    
    @abstractmethod
    def _raise_provider_error(self, error: MockError) -> None:
        """Raise a provider-specific error. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def __enter__(self) -> "MockLLM":
        """Start mocking. Must be implemented by subclasses."""
        pass
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Stop mocking and restore original behavior."""
        for p in reversed(self._patches):
            p.stop()
        self._patches.clear()
    
    def reset(self) -> None:
        """Reset the mock state (calls, responses, counters)."""
        self._responses.clear()
        self._response_index = 0
        self._calls.clear()
        self._call_count = 0
        self._total_tokens = 0
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._error = None


# Cost estimation data (USD per 1K tokens, as of Dec 2024)
MODEL_COSTS: dict[str, dict[str, float]] = {
    # OpenAI
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    # Anthropic
    "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
    "claude-3-5-haiku-20241022": {"input": 0.001, "output": 0.005},
    "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
    # Google
    "gemini-1.5-pro": {"input": 0.00125, "output": 0.005},
    "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},
    # Default fallback
    "default": {"input": 0.001, "output": 0.002},
}


def estimate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> float:
    """
    Estimate the cost in USD for a given model and token counts.
    
    Args:
        model: Model name (e.g., "gpt-4o", "claude-3-5-sonnet-20241022")
        prompt_tokens: Number of input/prompt tokens
        completion_tokens: Number of output/completion tokens
        
    Returns:
        Estimated cost in USD
    """
    costs = MODEL_COSTS.get(model, MODEL_COSTS["default"])
    input_cost = (prompt_tokens / 1000) * costs["input"]
    output_cost = (completion_tokens / 1000) * costs["output"]
    return input_cost + output_cost
