"""
Google Gemini API mock for pytest-mockllm.

Provides comprehensive mocking for the Google GenerativeAI SDK including:
- Content generation (sync, async, streaming)
- Chat sessions
- Embeddings
"""

from __future__ import annotations

import time
from typing import Any, Dict, Iterator, List, Optional
from unittest.mock import MagicMock, patch

from pytest_mockllm.core import MockLLM, MockResponse, MockError, TokenUsage


class GeminiMock(MockLLM):
    """
    Mock for Google Generative AI (Gemini) SDK.
    
    Supports:
        - model.generate_content()
        - model.generate_content() with streaming
        - Chat sessions
        - Embeddings
    
    Example:
        >>> def test_gemini_bot(mock_gemini):
        ...     mock_gemini.add_response("Here's what I found...")
        ...     
        ...     import google.generativeai as genai
        ...     model = genai.GenerativeModel("gemini-1.5-pro")
        ...     response = model.generate_content("Tell me about AI")
        ...     assert "found" in response.text.lower()
    """
    
    def __init__(self) -> None:
        super().__init__()
        self._default_model: str = "gemini-1.5-pro"
    
    def _raise_provider_error(self, error: MockError) -> None:
        """Raise a Gemini-style API error."""
        try:
            from google.api_core import exceptions
            
            error_classes = {
                "rate_limit": exceptions.ResourceExhausted,
                "auth": exceptions.Unauthenticated,
                "timeout": exceptions.DeadlineExceeded,
                "server": exceptions.InternalServerError,
                "invalid_request": exceptions.InvalidArgument,
            }
            
            error_class = error_classes.get(error.error_type, exceptions.GoogleAPIError)
            raise error_class(error.message)
        except ImportError:
            raise RuntimeError(f"Gemini API Error ({error.error_type}): {error.message}")
    
    def _create_generation(self, **kwargs: Any) -> Any:
        """Create a mock content generation response."""
        self._record_call(type="generate_content", **kwargs)
        response = self._get_next_response()
        
        return self._build_generation_response(response)
    
    def _build_generation_response(self, response: MockResponse) -> Any:
        """Build a Gemini-style GenerateContentResponse."""
        try:
            from google.generativeai.types import GenerateContentResponse
            from google.ai.generativelanguage_v1beta import (
                GenerateContentResponse as ProtoResponse,
                Candidate,
                Content,
                Part,
                UsageMetadata,
            )
            
            # Build proto-style response
            proto = ProtoResponse(
                candidates=[
                    Candidate(
                        content=Content(
                            parts=[Part(text=response.content)],
                            role="model",
                        ),
                        finish_reason=1,  # STOP
                        index=0,
                    )
                ],
                usage_metadata=UsageMetadata(
                    prompt_token_count=response.token_usage.prompt_tokens if response.token_usage else 10,
                    candidates_token_count=response.token_usage.completion_tokens if response.token_usage else 50,
                    total_token_count=response.token_usage.total_tokens if response.token_usage else 60,
                ),
            )
            
            return GenerateContentResponse.from_response(proto)
        except ImportError:
            return self._build_mock_generation(response)
    
    def _build_mock_generation(self, response: MockResponse) -> MagicMock:
        """Build a MagicMock generation when google-generativeai not installed."""
        mock = MagicMock()
        mock.text = response.content
        
        # Candidates
        candidate = MagicMock()
        candidate.content.parts = [MagicMock(text=response.content)]
        candidate.content.role = "model"
        candidate.finish_reason = "STOP"
        mock.candidates = [candidate]
        
        # Usage metadata
        mock.usage_metadata = MagicMock()
        mock.usage_metadata.prompt_token_count = response.token_usage.prompt_tokens if response.token_usage else 10
        mock.usage_metadata.candidates_token_count = response.token_usage.completion_tokens if response.token_usage else 50
        mock.usage_metadata.total_token_count = response.token_usage.total_tokens if response.token_usage else 60
        
        return mock
    
    def _create_streaming_generation(self, **kwargs: Any) -> Iterator[Any]:
        """Create a streaming content generation response."""
        self._record_call(type="generate_content", stream=True, **kwargs)
        response = self._get_next_response()
        
        # Split content into chunks
        if response.stream_chunks:
            chunks = response.stream_chunks
        else:
            words = response.content.split()
            chunks = [word + " " for word in words[:-1]] + [words[-1]] if words else [""]
        
        for i, chunk_content in enumerate(chunks):
            is_last = (i == len(chunks) - 1)
            yield self._build_stream_chunk(chunk_content, is_last, response)
    
    def _build_stream_chunk(
        self,
        content: str,
        is_last: bool,
        response: MockResponse,
    ) -> Any:
        """Build a streaming chunk."""
        mock = MagicMock()
        mock.text = content
        
        candidate = MagicMock()
        candidate.content.parts = [MagicMock(text=content)]
        candidate.content.role = "model"
        candidate.finish_reason = "STOP" if is_last else None
        mock.candidates = [candidate]
        
        return mock
    
    def _create_chat_session(self) -> MagicMock:
        """Create a mock chat session."""
        session = MagicMock()
        
        def send_message(content: str, **kwargs: Any) -> Any:
            self._record_call(type="chat.send_message", content=content, **kwargs)
            response = self._get_next_response()
            return self._build_mock_generation(response)
        
        session.send_message = send_message
        session.history = []
        
        return session
    
    def __enter__(self) -> "GeminiMock":
        """Start mocking Gemini API calls."""
        mock_module = MagicMock()
        
        # Create mock GenerativeModel class
        def mock_model_class(model_name: str = "gemini-1.5-pro", **kwargs: Any) -> MagicMock:
            model = MagicMock()
            model.model_name = model_name
            
            def generate_content(*args: Any, **kw: Any) -> Any:
                if kw.get("stream", False):
                    return self._create_streaming_generation(**kw)
                return self._create_generation(**kw)
            
            model.generate_content = generate_content
            model.start_chat = lambda **kw: self._create_chat_session()
            model.count_tokens = lambda content: MagicMock(total_tokens=len(str(content).split()) * 1.3)
            
            return model
        
        mock_module.GenerativeModel = mock_model_class
        mock_module.configure = MagicMock()
        
        try:
            patcher = patch.dict("sys.modules", {"google.generativeai": mock_module})
            self._patches.append(patcher)
            patcher.start()
            
            # Also patch if already imported
            import_patcher = patch("google.generativeai.GenerativeModel", mock_model_class)
            self._patches.append(import_patcher)
            import_patcher.start()
        except Exception:
            pass
        
        self._mock_module = mock_module
        return self
    
    @property
    def module(self) -> MagicMock:
        """Access the mock module directly for advanced usage."""
        return self._mock_module
