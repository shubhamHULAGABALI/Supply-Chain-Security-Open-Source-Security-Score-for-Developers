"""
backend/config/settings.py

Single source of truth for all configuration.
All values can be overridden via environment variables or a .env file.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


class Settings:
    """
    Reads from environment variables with sensible defaults.
    Priority: env var > .env file > default.
    """

    # ── Paths ────────────────────────────────────────────────────────────────
    DATA_DIR: Path = Path(os.getenv("DEEPRISK_DATA_DIR", "backend/data"))
    CKPT_NAME: str = os.getenv("DEEPRISK_CKPT", "best_model.pt")

    # ── Server ────────────────────────────────────────────────────────────────
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    WORKERS: int = int(os.getenv("WORKERS", "1"))   # >1 needs gunicorn
    RELOAD: bool = os.getenv("RELOAD", "0") == "1"

    # ── Device ────────────────────────────────────────────────────────────────
    DEVICE: str = os.getenv("DEEPRISK_DEVICE", "cpu")

    # ── Redis cache ───────────────────────────────────────────────────────────
    # Redis is optional. If REDIS_URL is not set (or redis-py not installed),
    # the server falls back to an in-process TTLCache automatically.
    REDIS_URL: Optional[str] = os.getenv("REDIS_URL", None)   # e.g. redis://localhost:6379/0
    CACHE_TTL_SECONDS: int   = int(os.getenv("CACHE_TTL_SECONDS", str(24 * 3600)))  # 24h
    LOCAL_CACHE_MAX: int     = int(os.getenv("LOCAL_CACHE_MAX", "5000"))

    # ── Rate limiting ─────────────────────────────────────────────────────────
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "120"))

    # ── OSV.dev fallback ──────────────────────────────────────────────────────
    # If True, packages not in the graph are cross-checked against OSV.dev
    OSV_FALLBACK: bool   = os.getenv("OSV_FALLBACK", "1") == "1"
    OSV_TIMEOUT_S: float = float(os.getenv("OSV_TIMEOUT_S", "2.0"))

    # ── Security ──────────────────────────────────────────────────────────────
    API_KEY: Optional[str] = os.getenv("DEEPRISK_API_KEY", None)   # optional auth

    # ── Logging ───────────────────────────────────────────────────────────────
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    @classmethod
    def load_dotenv(cls) -> None:
        """Load .env file if present (no external dependency needed)."""
        env_file = Path(".env")
        if not env_file.exists():
            return
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val
        # Re-read after loading
        cls.DATA_DIR             = Path(os.getenv("DEEPRISK_DATA_DIR", str(cls.DATA_DIR)))
        cls.REDIS_URL            = os.getenv("REDIS_URL", cls.REDIS_URL)
        cls.DEVICE               = os.getenv("DEEPRISK_DEVICE", cls.DEVICE)
        cls.CACHE_TTL_SECONDS    = int(os.getenv("CACHE_TTL_SECONDS", str(cls.CACHE_TTL_SECONDS)))
        cls.RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", str(cls.RATE_LIMIT_PER_MINUTE)))
        cls.OSV_FALLBACK         = os.getenv("OSV_FALLBACK", "1" if cls.OSV_FALLBACK else "0") == "1"
        cls.LOG_LEVEL            = os.getenv("LOG_LEVEL", cls.LOG_LEVEL)


settings = Settings()
Settings.load_dotenv()
