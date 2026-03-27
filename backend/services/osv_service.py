"""
backend/services/osv_service.py — OSV.dev fallback for unknown packages.

When a package is NOT in the training graph (new package, different ecosystem),
we query the OSV.dev API to check for known CVEs.

This makes the system useful even for packages added after the training cutoff.

OSV API: https://api.osv.dev/v1/query
  POST {"package": {"name": "lodash", "ecosystem": "npm"}}
  Returns: {"vulns": [{...}, ...]}  or {}

Response time: ~200-400ms (cached after first hit).
"""
from __future__ import annotations

import json
import time
import urllib.request
import urllib.error
from typing import Dict, List, Optional

from backend.config.settings import settings
from backend.utils.logger import get_logger

log = get_logger("deeprisk.osv")

_OSV_URL = "https://api.osv.dev/v1/query"

# Simple in-memory cache for OSV responses (separate from prediction cache)
_osv_cache: Dict[str, tuple] = {}   # pkg → (vulns, timestamp)
_OSV_TTL = 3600   # 1 hour


def query_osv(package_name: str) -> Dict:
    """
    Query OSV.dev for known vulnerabilities in an npm package.

    Returns:
        {
          "package": str,
          "vuln_count": int,
          "critical_count": int,
          "cve_ids": [str, ...],   # up to 5
          "source": "osv.dev",
          "queried": bool,
        }
    """
    name = package_name.lower().strip()

    # Check cache
    if name in _osv_cache:
        cached_result, ts = _osv_cache[name]
        if time.time() - ts < _OSV_TTL:
            return cached_result

    # Default response (used when query fails or times out)
    default = {
        "package": name, "vuln_count": 0, "critical_count": 0,
        "cve_ids": [], "source": "osv.dev", "queried": False,
    }

    if not settings.OSV_FALLBACK:
        return default

    try:
        payload = json.dumps({
            "package": {"name": name, "ecosystem": "npm"}
        }).encode()
        req = urllib.request.Request(
            _OSV_URL,
            data=payload,
            headers={"Content-Type": "application/json", "User-Agent": "deeprisk-oss/1.0"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=settings.OSV_TIMEOUT_S) as resp:
            data = json.loads(resp.read())

        vulns    = data.get("vulns", [])
        cve_ids  = []
        critical = 0
        for v in vulns[:10]:
            # Extract CVE IDs
            for alias in v.get("aliases", []):
                if alias.startswith("CVE-"):
                    cve_ids.append(alias)
            # Count severity
            for sev in v.get("severity", []):
                score = sev.get("score", "")
                try:
                    if float(score) >= 9.0:
                        critical += 1
                except (ValueError, TypeError):
                    pass

        result = {
            "package"       : name,
            "vuln_count"    : len(vulns),
            "critical_count": critical,
            "cve_ids"       : cve_ids[:5],
            "source"        : "osv.dev",
            "queried"       : True,
        }
        _osv_cache[name] = (result, time.time())
        log.info(f"OSV: {name}  vulns={len(vulns)}  critical={critical}")
        return result

    except urllib.error.URLError as e:
        log.warning(f"OSV timeout/network error for '{name}': {e}")
        _osv_cache[name] = (default, time.time())
        return default
    except Exception as e:
        log.warning(f"OSV query failed for '{name}': {e}")
        return default


def osv_risk_label(osv: Dict, threshold_vulns: int = 1) -> str:
    """Convert OSV result to a risk label for display."""
    if not osv["queried"]:
        return "UNKNOWN"
    if osv["critical_count"] > 0:
        return "HIGH"
    if osv["vuln_count"] >= threshold_vulns:
        return "MEDIUM"
    return "LOW"
