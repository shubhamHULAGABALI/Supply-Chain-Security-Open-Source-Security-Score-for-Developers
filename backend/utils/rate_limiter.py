"""
backend/utils/rate_limiter.py — Sliding-window in-process rate limiter.

Used by main.py as a FastAPI middleware.
No external dependencies — pure Python collections.

Default: 120 requests/minute per IP (configurable via RATE_LIMIT_PER_MINUTE).
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from threading import Lock
from typing import Dict, Deque


class SlidingWindowRateLimiter:
    """
    Per-IP sliding-window rate limiter.

    Thread-safe.  O(1) per check (amortised) via deque expiry.
    """

    def __init__(self, limit: int, window_s: int = 60):
        self._limit  = limit
        self._window = window_s
        self._data   : Dict[str, Deque[float]] = defaultdict(deque)
        self._lock   = Lock()

    def is_allowed(self, key: str) -> bool:
        """Return True if this request is within the rate limit."""
        now     = time.monotonic()
        cutoff  = now - self._window
        with self._lock:
            dq = self._data[key]
            # Drop timestamps outside the window
            while dq and dq[0] < cutoff:
                dq.popleft()
            if len(dq) >= self._limit:
                return False
            dq.append(now)
        return True

    def remaining(self, key: str) -> int:
        """Return how many requests are left in the current window."""
        now    = time.monotonic()
        cutoff = now - self._window
        with self._lock:
            dq = self._data[key]
            while dq and dq[0] < cutoff:
                dq.popleft()
            return max(0, self._limit - len(dq))

    def reset(self, key: str) -> None:
        """Clear rate limit data for a given key."""
        with self._lock:
            self._data.pop(key, None)

    def clear(self) -> None:
        """Clear all rate limit data (e.g. for testing)."""
        with self._lock:
            self._data.clear()
