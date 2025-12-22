"""Tests for the OpenAI mock functionality."""

import pytest


class TestOpenAIMock:
    """Tests for OpenAIMock fixture and functionality."""
    
    def test_basic_response(self, mock_openai):
        """Basic response should work out of the box."""
        mock_openai.add_response("Hello! I'm here to help.")
        
        # The mock should return our configured response
        response = mock_openai._get_next_response()
        assert response.content == "Hello! I'm here to help."
    
    def test_multiple_responses(self, mock_openai):
        """Multiple responses should be returned in order."""
        mock_openai.add_responses(
            "First response",
            "Second response",
            "Third response",
        )
        
        assert mock_openai._get_next_response().content == "First response"
        assert mock_openai._get_next_response().content == "Second response"
        assert mock_openai._get_next_response().content == "Third response"
    
    def test_default_response(self, mock_openai):
        """Default response should be used when queue is empty."""
        mock_openai.set_default_response("I'm the default!")
        
        # First call - no responses in queue, uses default
        response = mock_openai._get_next_response()
        assert response.content == "I'm the default!"
        
        # Second call - still uses default
        response = mock_openai._get_next_response()
        assert response.content == "I'm the default!"
    
    def test_strict_mode(self, mock_openai):
        """Strict mode should raise error when no response configured."""
        mock_openai.set_strict_mode(True)
        
        with pytest.raises(RuntimeError, match="No mock response configured"):
            mock_openai._get_next_response()
    
    def test_call_tracking(self, mock_openai):
        """Calls should be tracked."""
        mock_openai.add_response("Response 1")
        mock_openai.add_response("Response 2")
        
        mock_openai._record_call(type="test", message="call 1")
        mock_openai._get_next_response()
        
        mock_openai._record_call(type="test", message="call 2")
        mock_openai._get_next_response()
        
        assert mock_openai.call_count == 2
        assert len(mock_openai.calls) == 2
        assert mock_openai.last_call["message"] == "call 2"
    
    def test_token_accumulation(self, mock_openai):
        """Tokens should accumulate across calls."""
        from pytest_mockllm.core import TokenUsage
        
        mock_openai.add_response("Short", token_usage=TokenUsage(10, 20))
        mock_openai.add_response("Longer", token_usage=TokenUsage(15, 25))
        
        mock_openai._get_next_response()
        mock_openai._get_next_response()
        
        assert mock_openai.total_prompt_tokens == 25
        assert mock_openai.total_completion_tokens == 45
        assert mock_openai.total_tokens == 70
    
    def test_method_chaining(self, mock_openai):
        """Methods should support chaining."""
        result = (
            mock_openai
            .add_response("First")
            .add_response("Second")
            .set_default_response("Default")
            .set_strict_mode(False)
        )
        
        assert result is mock_openai
    
    def test_reset(self, mock_openai):
        """Reset should clear all state."""
        mock_openai.add_response("Test")
        mock_openai._record_call(type="test")
        mock_openai._get_next_response()
        
        mock_openai.reset()
        
        assert mock_openai.call_count == 0
        assert mock_openai.total_tokens == 0
        assert len(mock_openai.calls) == 0


class TestOpenAIErrorSimulation:
    """Tests for error simulation functionality."""
    
    def test_immediate_error(self, mock_openai):
        """Simulating an immediate error should raise on first call."""
        mock_openai.simulate_error("rate_limit")
        
        with pytest.raises((RuntimeError, Exception)) as exc_info:
            mock_openai._get_next_response()
        
        assert "rate_limit" in str(exc_info.value).lower() or "Rate limit" in str(exc_info.value)
    
    def test_delayed_error(self, mock_openai):
        """Error after N calls should work correctly."""
        mock_openai.add_responses("OK 1", "OK 2", "OK 3")
        mock_openai.simulate_error("rate_limit", after_calls=2)
        
        # First two calls succeed
        assert mock_openai._get_next_response().content == "OK 1"
        assert mock_openai._get_next_response().content == "OK 2"
        
        # Third call fails
        with pytest.raises((RuntimeError, Exception)):
            mock_openai._get_next_response()


class TestOpenAIStreaming:
    """Tests for streaming functionality."""
    
    def test_streaming_chunks(self, mock_openai):
        """Streaming should split content into chunks."""
        mock_openai.add_response("Hello world from streaming!")
        
        # Create streaming response
        stream = mock_openai._create_streaming_completion(model="gpt-4o")
        
        # Collect all chunks
        chunks = list(stream)
        
        # Should have multiple chunks
        assert len(chunks) > 1
        
        # Concatenated content should match
        full_content = ""
        for chunk in chunks:
            if chunk.choices[0].delta.content:
                full_content += chunk.choices[0].delta.content
        
        assert "Hello" in full_content
        assert "streaming" in full_content
