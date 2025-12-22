"""
Global statistics tracker for pytest-mockllm.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from pytest_mockllm.core import estimate_cost


@dataclass
class UsageStats:
    total_calls: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cost_saved: float = 0.0
    model_counts: dict[str, int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record_call(self, model: str, prompt: int, completion: int) -> None:
        with self._lock:
            self.total_calls += 1
            self.total_prompt_tokens += prompt
            self.total_completion_tokens += completion
            self.total_cost_saved += estimate_cost(model, prompt, completion)

            self.model_counts[model] = self.model_counts.get(model, 0) + 1

    def reset(self) -> None:
        with self._lock:
            self.total_calls = 0
            self.total_prompt_tokens = 0
            self.total_completion_tokens = 0
            self.total_cost_saved = 0.0
            self.model_counts.clear()


# Global singleton
GLOBAL_STATS = UsageStats()
