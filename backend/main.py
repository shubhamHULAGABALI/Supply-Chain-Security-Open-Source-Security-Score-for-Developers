"""
backend/main.py — FastAPI application entry point.

Start with:
    python backend/main.py
    or
    uvicorn backend.main:app --host 0.0.0.0 --port 8000

Architecture:
  Startup → load model + data → run full-graph inference (warms logit+attention cache)
  /predict → L1 TTLCache (0.1ms) → return
  /predict/batch → single graph inference (already warm) + N O(1) lookups
  /predict/{name} → GET shortcut for curl / browser demos
"""
from __future__ import annotations

import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

_here = Path(__file__).parent
if str(_here.parent) not in sys.path:
    sys.path.insert(0, str(_here.parent))

from fastapi import FastAPI, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader

from backend.config.settings import settings
from backend.model.inference import RiskPredictor
from backend.schemas import (
    PredictRequest, BatchPredictRequest,
    PredictResponse, BatchPredictResponse, StatsResponse, HealthResponse,
    NeighbourInfo, OSVInfo,
)
from backend.services.cache import cache
from backend.utils.logger import get_logger
from backend.utils.rate_limiter import SlidingWindowRateLimiter

log = get_logger("deeprisk-oss.main")

_predictor      : Optional[RiskPredictor] = None
_startup_error  : Optional[str]           = None
_startup_ms     : float                   = 0.0
_rate_limiter   = SlidingWindowRateLimiter(
    limit    = settings.RATE_LIMIT_PER_MINUTE,
    window_s = 60,
)


# ─── Lifespan (startup / shutdown) ────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predictor, _startup_error, _startup_ms
    t0 = time.perf_counter()
    log.info(
        f"Starting DeepRisk OSS API  "
        f"data={settings.DATA_DIR}  device={settings.DEVICE}"
    )
    try:
        _predictor = RiskPredictor.from_directory()
        # Warm the full-graph inference so the first user request is <5ms
        # This also caches attention weights (FIX-1)
        _ = _predictor._run_full_graph()
        _startup_ms = (time.perf_counter() - t0) * 1000
        log.info(
            f"Ready in {_startup_ms:.0f}ms  "
            f"attn_cached={_predictor._attn_weights is not None}"
        )
    except Exception as e:
        _startup_error = str(e)
        log.error(f"Startup FAILED: {e}")
    yield
    log.info("Shutting down …")


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "DeepRisk OSS — Supply Chain Security API",
    description = (
        "GAT+LSTM supply-chain risk predictor for npm packages.\n\n"
        "**<500ms cold, <5ms warm** — powered by graph attention + temporal LSTM.\n\n"
        "Returns calibrated risk probability, CVE data, top graph neighbours "
        "and safer alternatives."
    ),
    version  = "1.0.0",
    lifespan = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


# ─── Middleware ────────────────────────────────────────────────────────────────

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    # Only rate-limit prediction endpoints
    if request.url.path.startswith("/predict"):
        ip = request.client.host if request.client else "unknown"
        if not _rate_limiter.is_allowed(ip):
            remaining = _rate_limiter.remaining(ip)
            return JSONResponse(
                status_code = 429,
                content     = {
                    "detail"   : "Rate limit exceeded",
                    "limit"    : settings.RATE_LIMIT_PER_MINUTE,
                    "window_s" : 60,
                    "remaining": remaining,
                },
                headers = {"Retry-After": "60"},
            )
    return await call_next(request)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _check_auth(api_key: Optional[str] = None) -> None:
    if not settings.API_KEY:
        return
    if api_key != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def _require_predictor() -> RiskPredictor:
    if _predictor is None:
        detail = f"Model not ready: {_startup_error}" if _startup_error else "Model loading"
        raise HTTPException(status_code=503, detail=detail)
    return _predictor


def _to_response(r) -> PredictResponse:
    return PredictResponse(
        package       = r.package,
        risk_score    = r.risk_score,
        risk_prob     = r.risk_prob,
        risk_label    = r.risk_label,
        risk_tier     = r.risk_tier,
        threshold     = r.threshold,
        temperature   = r.temperature,
        in_dataset    = r.in_dataset,
        val_auc       = r.val_auc,
        inference_ms  = r.inference_ms,
        top_neighbors = [NeighbourInfo(**n) for n in (r.top_neighbors or [])],
        osv           = OSVInfo(**r.osv) if r.osv else None,
        explanation   = r.explanation,
        warnings      = r.warnings or [],
        cached        = r.cached,
        alternatives  = r.alternatives or [],
    )


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health():
    """Liveness + readiness check."""
    return HealthResponse(
        status        = "ok",
        model_ready   = _predictor is not None,
        startup_ms    = round(_startup_ms, 1),
        startup_error = _startup_error,
    )


@app.get("/stats", response_model=StatsResponse, tags=["meta"])
async def stats(api_key: Optional[str] = Security(_api_key_header)):
    _check_auth(api_key)
    p = _require_predictor()
    return StatsResponse(**p.stats())


@app.get("/cache/clear", tags=["meta"])
async def clear_cache(api_key: Optional[str] = Security(_api_key_header)):
    _check_auth(api_key)
    result = cache.clear()
    if _predictor:
        _predictor._logit_cache     = None
        _predictor._attn_edge_index = None
        _predictor._attn_weights    = None
    log.info(f"Cache cleared: {result}")
    return {"cleared": result, "status": "ok"}


@app.post("/predict", response_model=PredictResponse, tags=["prediction"])
async def predict(
    req: PredictRequest,
    api_key: Optional[str] = Security(_api_key_header),
):
    """
    Predict supply-chain risk for a single npm package.

    - **<5ms** for cached packages (L1 TTLCache)
    - **<500ms** cold (first ever request triggers full-graph inference ~250ms)
    - Returns calibrated probability, tier, graph neighbours, OSV CVEs, alternatives
    """
    _check_auth(api_key)
    p = _require_predictor()
    r = p.predict(req.package, with_neighbors=req.with_neighbors)
    return _to_response(r)


@app.get("/predict/{package_name}", response_model=PredictResponse, tags=["prediction"])
async def predict_get(
    package_name: str,
    with_neighbors: bool = False,
    api_key: Optional[str] = Security(_api_key_header),
):
    """GET shortcut — useful for curl / browser / quick demos."""
    _check_auth(api_key)
    pkg = package_name.split("@")[0].strip().lower()
    p   = _require_predictor()
    r   = p.predict(pkg, with_neighbors=with_neighbors)
    return _to_response(r)


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["prediction"])
async def predict_batch(
    req: BatchPredictRequest,
    api_key: Optional[str] = Security(_api_key_header),
):
    """
    Batch scan up to 50 packages.
    Full-graph inference runs once; all per-package results are O(1) cache lookups.
    """
    _check_auth(api_key)
    t0      = time.perf_counter()
    p       = _require_predictor()
    results = p.predict_batch(req.packages, with_neighbors=req.with_neighbors)
    total_ms = (time.perf_counter() - t0) * 1000
    return BatchPredictResponse(
        results              = [_to_response(r) for r in results],
        total_ms             = round(total_ms, 2),
        high_risk_count      = sum(1 for r in results if r.risk_label == "HIGH"),
        packages_in_dataset  = sum(1 for r in results if r.in_dataset),
    )


@app.get("/verify", tags=["meta"])
async def verify(api_key: Optional[str] = Security(_api_key_header)):
    """Sanity-check model against known vulnerable / safe packages."""
    _check_auth(api_key)
    p = _require_predictor()
    known = [
        ("minimist", "high"),
        ("lodash",   "high"),
        ("chalk",    "low"),
        ("bluebird", "low"),
        ("axios",    "high"),
    ]
    logits = p._run_full_graph()
    out = []
    for pkg, expected in known:
        idx = p.name_to_idx.get(pkg)
        if idx is None:
            out.append({"package": pkg, "status": "NOT_IN_DATASET"})
            continue
        raw_prob  = p._prob(float(logits[idx]))
        raw_label = "HIGH" if raw_prob >= p.threshold else "LOW"

        # FIX: pass osv_data=None — /verify is a quick sanity-check, not a
        # live scan, so we skip the OSV network call here.
        # Signature: (pkg, prob, label, tier, warnings, osv_data)
        prob, label, _, _ = p._apply_known_safe_override(
            pkg, raw_prob, raw_label, "", [], osv_data=None
        )

        actual = label.lower()
        out.append({
            "package" : pkg,
            "expected": expected,
            "actual"  : actual,
            "prob"    : round(prob, 4),
            "raw_prob": round(raw_prob, 4),
            "match"   : actual == expected,
        })
    passed = sum(1 for r in out if r.get("match"))
    total  = sum(1 for r in out if "match" in r)
    return {
        "passed"  : passed,
        "total"   : total,
        "accuracy": round(passed / max(total, 1), 4),
        "details" : out,
    }


# ─── Run directly ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    import logging

    logging.basicConfig(
        level  = getattr(logging, settings.LOG_LEVEL, logging.INFO),
        format = "%(asctime)s  %(levelname)-8s  %(message)s",
    )
    uvicorn.run(
        "backend.main:app",
        host      = settings.HOST,
        port      = settings.PORT,
        reload    = settings.RELOAD,
        log_level = settings.LOG_LEVEL.lower(),
    )