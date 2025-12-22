"""Tests for the core MockLLM functionality."""

import pytest
from pytest_mockllm.core import MockLLM, MockResponse, TokenUsage, estimate_cost


class TestMockResponse:
    """Tests for MockResponse dataclass."""
    
    def test_basic_response(self):
        """A basic response should have sensible defaults."""
        response = MockResponse(content="Hello, world!")
        
        assert response.content == "Hello, world!"
        assert response.role == "assistant"
        assert response.finish_reason == "stop"
        assert response.token_usage is not None
        assert response.token_usage.total_tokens > 0
    
    def test_response_with_custom_tokens(self):
        """Custom token usage should be preserved."""
        usage = TokenUsage(prompt_tokens=50, completion_tokens=100)
        response = MockResponse(content="Test", token_usage=usage)
        
        assert response.token_usage.prompt_tokens == 50
        assert response.token_usage.completion_tokens == 100
        assert response.token_usage.total_tokens == 150
    
    def test_response_with_tool_calls(self):
        """Tool calls should be stored correctly."""
        tool_calls = [
            {
                "id": "call_123",
                "function": {
                    "name": "get_weather",
                    "arguments": {"location": "NYC"},
                },
            }
        ]
        response = MockResponse(content="", tool_calls=tool_calls)
        
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["function"]["name"] == "get_weather"


class TestTokenUsage:
    """Tests for TokenUsage dataclass."""
    
    def test_auto_calculates_total(self):
        """Total tokens should be auto-calculated if not provided."""
        usage = TokenUsage(prompt_tokens=10, completion_tokens=20)
        assert usage.total_tokens == 30
    
    def test_explicit_total(self):
        """Explicit total should be preserved."""
        usage = TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=50)
        assert usage.total_tokens == 50


class TestEstimateCost:
    """Tests for cost estimation."""
    
    def test_gpt4o_cost(self):
        """GPT-4o cost should be calculated correctly."""
        cost = estimate_cost("gpt-4o", prompt_tokens=1000, completion_tokens=1000)
        
        # $0.0025/1K input + $0.01/1K output = $0.0125
        assert 0.012 < cost < 0.013
    
    def test_claude_cost(self):
        """Claude cost should be calculated correctly."""
        cost = estimate_cost(
            "claude-3-5-sonnet-20241022",
            prompt_tokens=1000,
            completion_tokens=1000,
        )
        
        # $0.003/1K input + $0.015/1K output = $0.018
        assert 0.017 < cost < 0.019
    
    def test_unknown_model_uses_default(self):
        """Unknown models should use default pricing."""
        cost = estimate_cost("some-unknown-model", prompt_tokens=1000, completion_tokens=1000)
        
        assert cost > 0  # Should still return a value
