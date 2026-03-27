"""
backend/services/cache.py — Dual-layer cache with Redis + in-process TTLCache fallback.

Speed goal: <5ms cache hit at all times.

Layer 1 — Redis (shared across workers, survives restarts):
    Key:   deeprisk:pred:<package_name>
    Value: JSON-encoded PredictionResult dict
    TTL:   CACHE_TTL_SECONDS (default 24h)
    Install: pip install redis

Layer 2 — TTLCache (in-process, zero network cost, sub-millisecond):
    Used when Redis is unavailable or not configured.
    Also acts as an L1 in front of Redis: if Redis responds slowly,
    the TTLCache still returns results in <0.1ms.
    Max: LOCAL_CACHE_MAX entries (default 5000)

When Redis IS available, we use both:
  - Write: write to Redis AND TTLCache
  - Read:  check TTLCache first (0.1ms), then Redis (~1ms), then compute (~200ms)
"""
from __future__ import annotations

import json
import time
from collections import OrderedDict
from threading import Lock
from typing import Any, Optional

from backend.config.settings import settings
from backend.utils.logger import get_logger

log = get_logger("deeprisk.cache")


# ─────────────────────────────────────────────────────────────────────────────
# In-process TTLCache (always available — no dependencies)
# ─────────────────────────────────────────────────────────────────────────────

class TTLCache:
    """
    Thread-safe LRU cache with per-entry time-to-live.

    Eviction: LRU order + TTL expiry.
    All operations O(1).
    """

    def __init__(self, max_size: int, ttl_s: int):
        self._max  = max_size
        self._ttl  = ttl_s
        self._data : OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._lock = Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._data:
                return None
            val, exp = self._data[key]
            if time.monotonic() > exp:
                del self._data[key]
                return None
            self._data.move_to_end(key)   # LRU refresh
            return val

    def set(self, key: str, val: Any) -> None:
        with self._lock:
            exp = time.monotonic() + self._ttl
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = (val, exp)
            while len(self._data) > self._max:
                self._data.popitem(last=False)

    def delete(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)

    def clear(self) -> int:
        with self._lock:
            n = len(self._data)
            self._data.clear()
            return n

    def size(self) -> int:
        with self._lock:
            return len(self._data)


# ─────────────────────────────────────────────────────────────────────────────
# Cache manager
# ─────────────────────────────────────────────────────────────────────────────

_REDIS_PREFIX = "deeprisk:pred:"

class CacheManager:
    """
    Unified cache interface: Redis (optional) + TTLCache (always).

    Usage:
        cache = CacheManager()
        cache.set("lodash", result_dict)
        data = cache.get("lodash")   # returns dict or None
    """

    def __init__(self):
        self._local = TTLCache(
            max_size = settings.LOCAL_CACHE_MAX,
            ttl_s    = settings.CACHE_TTL_SECONDS,
        )
        self._redis = None
        self._redis_ok = False
        self._connect_redis()

    def _connect_redis(self) -> None:
        if not settings.REDIS_URL:
            log.info("Redis not configured — using in-process cache only")
            return
        try:
            import redis as redis_lib
            r = redis_lib.from_url(settings.REDIS_URL, socket_timeout=1.0,
                                   decode_responses=True)
            r.ping()
            self._redis    = r
            self._redis_ok = True
            log.info(f"Redis connected: {settings.REDIS_URL}")
        except Exception as e:
            log.warning(f"Redis unavailable ({e}) — falling back to in-process cache")
            self._redis    = None
            self._redis_ok = False

    # ── Public API ────────────────────────────────────────────────────────────

    def get(self, package: str) -> Optional[dict]:
        """Check L1 (TTLCache) then L2 (Redis). Returns None on miss."""
        key = package.lower().strip()

        # L1: in-process (sub-millisecond)
        val = self._local.get(key)
        if val is not None:
            return val

        # L2: Redis
        if self._redis_ok:
            try:
                raw = self._redis.get(_REDIS_PREFIX + key)
                if raw:
                    data = json.loads(raw)
                    self._local.set(key, data)   # populate L1
                    return data
            except Exception as e:
                log.debug(f"Redis get error ({e}); falling back to local")
                self._redis_ok = False           # stop trying Redis until next restart

        return None

    def set(self, package: str, data: dict) -> None:
        """Write to both L1 and L2."""
        key = package.lower().strip()
        self._local.set(key, data)

        if self._redis_ok:
            try:
                self._redis.setex(
                    _REDIS_PREFIX + key,
                    settings.CACHE_TTL_SECONDS,
                    json.dumps(data),
                )
            except Exception as e:
                log.debug(f"Redis set error ({e})")
                self._redis_ok = False

    def delete(self, package: str) -> None:
        key = package.lower().strip()
        self._local.delete(key)
        if self._redis_ok:
            try:
                self._redis.delete(_REDIS_PREFIX + key)
            except Exception:
                pass

    def clear(self) -> dict:
        n_local = self._local.clear()
        n_redis = 0
        if self._redis_ok:
            try:
                keys = self._redis.keys(_REDIS_PREFIX + "*")
                if keys:
                    n_redis = len(keys)
                    self._redis.delete(*keys)
            except Exception:
                pass
        return {"local_cleared": n_local, "redis_cleared": n_redis}

    def stats(self) -> dict:
        return {
            "local_entries"  : self._local.size(),
            "local_max"      : settings.LOCAL_CACHE_MAX,
            "redis_connected": self._redis_ok,
            "redis_url"      : settings.REDIS_URL or "not configured",
            "ttl_hours"      : settings.CACHE_TTL_SECONDS / 3600,
        }


# Singleton — shared across the entire process
cache = CacheManager()
