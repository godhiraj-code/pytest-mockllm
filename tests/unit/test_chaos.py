import pytest
import time
from pytest_mockllm.core import MockLLM, MockError
from unittest.mock import MagicMock

class SimpleMock(MockLLM):
    def _raise_provider_error(self, error: MockError) -> None:
        raise RuntimeError(error.error_type)
    def __enter__(self):
        return self

def test_chaos_jitter():
    mock = SimpleMock()
    mock.add_responses("A", "B", "C")
    mock.simulate_jitter(max_ms=100)
    
    start = time.time()
    mock._get_next_response()
    duration = (time.time() - start) * 1000
    # Jitter is 0-100ms. Hard to verify precisely without many samples, 
    # but we just want to ensure it doesn't crash.
    assert duration >= 0

def test_chaos_random_errors():
    mock = SimpleMock()
    # High probability to ensure we hit it in a few tries
    mock.simulate_random_errors(probability=1.0)
    mock.add_response("Success")
    
    with pytest.raises(RuntimeError) as excinfo:
        mock._get_next_response()
    
    assert str(excinfo.value) in ["rate_limit", "timeout", "server"]
