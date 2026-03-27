"""
backend/model/inference.py — RiskPredictor: <500ms cold, <5ms warm.

Speed architecture
──────────────────────────
Cold call (first ever request):
  1. Full-graph forward pass WITH attention: ~150–300ms (CPU, 19k nodes)
  2. All 19,727 logits + attention weights stored in RAM
  3. Redis write: ~2ms
  Total: ~160–310ms  ✓ well under 500ms

Warm call (any subsequent request):
  1. L1 TTLCache lookup: ~0.1ms
  Total: ~0.1ms  ✓

Neighbor call (with_neighbors=True, first time):
  1. Logit cache: 0.1ms
  2. Attention cache lookup: 0.1ms
  ZERO re-inference.

After server restart (Redis populated):
  1. L1 miss → Redis get: ~1ms
  Total: ~2ms  ✓

"""
from __future__ import annotations

import json
import pickle
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.utils import add_self_loops

from backend.model.architecture import GATLSTM, load_from_checkpoint
from backend.config.settings import settings
from backend.services.cache import cache
from backend.services.osv_service import query_osv, osv_risk_label
from backend.utils.logger import get_logger

log = get_logger("deeprisk.inference")

# ── No _ALTERNATIVE_HINTS / _KNOWN_SAFE_PKGS — see module docstring ───────────


# ─────────────────────────────────────────────────────────────────────────────
# PredictionResult
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PredictionResult:
    package      : str
    risk_score   : int
    risk_prob    : float
    risk_label   : str
    risk_tier    : str
    threshold    : float
    temperature  : float
    in_dataset   : bool
    val_auc      : float
    inference_ms : float
    top_neighbors: List[Dict]     = field(default_factory=list)
    osv          : Optional[Dict] = None
    explanation  : str            = ""
    warnings     : List[str]      = field(default_factory=list)
    cached       : bool           = False
    alternatives : List[str]      = field(default_factory=list)

    # FIX-6 ────────────────────────────────────────────────────────────────────
    @property
    def osv_summary(self) -> Optional[str]:
        """
        Single formatted string for CLI/API display, e.g.:
          "8 known vuln(s)  CVE-2020-28500, CVE-2021-23337, CVE-2018-16487"
        Returns None when OSV was not queried or returned no results.
        """
        if not self.osv or not self.osv.get("queried"):
            return None
        n = self.osv.get("vuln_count", 0)
        if n == 0:
            return "0 known vulnerabilities (OSV.dev)"
        cves = self.osv.get("cve_ids", [])
        cve_str = ", ".join(cves[:4])
        if len(cves) > 4:
            cve_str += f" (+{len(cves) - 4} more)"
        return f"{n} known vuln(s)  {cve_str}".strip()

    def to_dict(self) -> Dict:
        d = asdict(self)
        # include the computed property so cached hits also carry it
        d["osv_summary"] = self.osv_summary
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _tier(prob: float, thr: float) -> Tuple[str, str]:
    """Map calibrated probability to a human-readable risk tier."""
    critical_thr = min(thr + 0.15, 0.93)
    if prob >= critical_thr:
        return "HIGH", "CRITICAL — known vulnerability signals, avoid or patch immediately"
    if prob >= thr:
        return "HIGH", "HIGH — investigate before use; check CVEs and maintenance status"
    if prob >= max(thr - 0.20, 0.05):
        return "MEDIUM", "MEDIUM — monitor for security updates"
    return "LOW", "LOW — generally safe based on graph analysis"


def _explanation(pkg: str, prob: float, thr: float,
                 neighbors: List[Dict], osv: Optional[Dict]) -> str:
    parts = []
    if osv and osv.get("queried") and osv.get("vuln_count", 0) > 0:
        cves = ", ".join(osv.get("cve_ids", [])[:2]) or "unknown CVEs"
        parts.append(
            f"{osv['vuln_count']} known vulnerability(ies) in OSV database ({cves})"
        )
    if prob >= thr:
        risky_nb = [n for n in neighbors if n["risk_prob"] >= thr]
        if risky_nb:
            names = ", ".join(n["package"] for n in risky_nb[:2])
            parts.append(
                f"strong graph influence from {len(risky_nb)} high-risk "
                f"neighbour(s): {names}"
            )
        if not parts:
            parts.append(
                "feature profile and commit-activity patterns match known-vulnerable packages"
            )
        return f"Flagged HIGH: {'; '.join(parts)}."
    if parts:
        # MEDIUM but has OSV hits
        return f"Flagged MEDIUM: {'; '.join(parts)}. Monitor and patch promptly."
    return (
        "Predicted LOW/MEDIUM: feature similarity and temporal commit history "
        "consistent with well-maintained safe packages."
    )


# ─────────────────────────────────────────────────────────────────────────────
# RiskPredictor
# ─────────────────────────────────────────────────────────────────────────────

class RiskPredictor:
    """
    Load once, serve thousands of requests at <5ms warm latency.
    Thread-safe: a lock guards the one-time full-graph inference.
    """

    def __init__(
        self,
        model          : GATLSTM,
        x              : torch.Tensor,
        edge_index     : torch.Tensor,
        temporal       : torch.Tensor,
        coverage_mask  : torch.Tensor,
        name_to_idx    : Dict[str, int],
        idx_to_name    : Dict[int, str],
        temperature    : float,
        threshold      : float,
        epoch          : int,
        val_auc        : float,
        device         : str,
        node_labels    : Optional[torch.Tensor] = None,
    ):
        self.model          = model
        self.x              = x
        self.edge_index     = edge_index
        self.temporal       = temporal
        self.coverage_mask  = coverage_mask
        self.name_to_idx    = name_to_idx
        self.idx_to_name    = idx_to_name
        self.temperature    = temperature
        self.threshold      = threshold
        self.epoch          = epoch
        self.val_auc        = val_auc
        self.device         = device
        self.node_labels    = node_labels

        # ── Inference caches (populated by _run_full_graph) ────────────────
        self._logit_cache     : Optional[np.ndarray] = None
        self._attn_edge_index : Optional[np.ndarray] = None  # (2, E)
        self._attn_weights    : Optional[np.ndarray] = None  # (E,) mean across heads
        self._lock            = threading.Lock()

        log.info(
            f"RiskPredictor ready | nodes={x.shape[0]:,} "
            f"edges={edge_index.shape[1]:,} "
            f"T={temperature:.3f} thr={threshold:.3f} val_auc={val_auc:.4f}"
        )

    # ── Factory ────────────────────────────────────────────────────────────────

    @classmethod
    def from_directory(
        cls,
        data_dir : str | Path = None,
        device   : str        = None,
    ) -> "RiskPredictor":
        d      = Path(data_dir or settings.DATA_DIR)
        device = device or settings.DEVICE
        log.info(f"Loading from {d.resolve()} on {device}")

        # Checkpoint
        ckpt_path = d / settings.CKPT_NAME
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        model, ckpt = load_from_checkpoint(str(ckpt_path), device)
        cfg = ckpt.get("cfg", {})

        # Calibration — prefer hyperparams.json
        hp_path = d / "hyperparams.json"
        if hp_path.exists():
            hp          = json.loads(hp_path.read_text())
            temperature = float(hp.get("temperature",       cfg.get("temperature",       1.0)))
            threshold   = float(hp.get("optimal_threshold", cfg.get("optimal_threshold", 0.5)))
        else:
            temperature = float(cfg.get("temperature",       1.0))
            threshold   = float(cfg.get("optimal_threshold", 0.5))
        epoch   = int(ckpt.get("epoch",   cfg.get("epoch",   0)))
        val_auc = float(ckpt.get("val_auc", cfg.get("val_auc", 0.0)))

        # Node features + scaler
        nf = np.load(d / "node_features.npy").astype(np.float32)
        sp = d / "feature_scaler.pkl"
        if sp.exists():
            with open(sp, "rb") as f:
                sc = pickle.load(f)
            nf = np.clip(sc.transform(nf).astype(np.float32), -5.0, 5.0)
            log.info("Feature scaler applied ✓")
        else:
            log.warning("feature_scaler.pkl missing — raw features used")

        # Edges
        edges = np.load(d / "edges.npy")
        ei    = torch.tensor(edges, dtype=torch.long).t().contiguous()
        ei, _ = add_self_loops(ei, num_nodes=nf.shape[0])

        # Temporal (per-node z-score)
        temp = np.load(d / "temporal_features.npy").astype(np.float32)
        eps  = 1e-8
        temp = (
            (temp - temp.mean(1, keepdims=True)) /
            (temp.std(1, keepdims=True) + eps)
        ).astype(np.float32)

        # Coverage mask
        cov_p = d / "coverage_mask.npy"
        cov   = np.load(cov_p).astype(bool) if cov_p.exists() else np.ones(nf.shape[0], bool)

        # Ground-truth labels (optional — enables FP/FN warnings)
        lbl_p  = d / "node_labels.npy"
        labels = (
            torch.tensor(np.load(lbl_p), dtype=torch.long).to(device)
            if lbl_p.exists() else None
        )

        # Name maps
        with open(d / "name_to_idx.pkl", "rb") as f:
            n2i = pickle.load(f)
        i2n = {v: k for k, v in n2i.items()}

        # Move tensors to device
        x_t   = torch.tensor(nf,   dtype=torch.float32).to(device)
        ei_t  = ei.to(device)
        t_t   = torch.tensor(temp, dtype=torch.float16).to(device)
        cov_t = torch.tensor(cov,  dtype=torch.bool).to(device)

        return cls(
            model=model, x=x_t, edge_index=ei_t, temporal=t_t,
            coverage_mask=cov_t, name_to_idx=n2i, idx_to_name=i2n,
            temperature=temperature, threshold=threshold, epoch=epoch,
            val_auc=val_auc, device=device, node_labels=labels,
        )

    # ── Core inference ─────────────────────────────────────────────────────────

    def _run_full_graph(self) -> np.ndarray:
        """
        One-time full-graph inference — cached in RAM.  Thread-safe.
        Runs with return_attention=True so attention is available for
        _top_neighbors() with zero extra forward passes (FIX-1).
        """
        with self._lock:
            if self._logit_cache is not None:
                return self._logit_cache

            log.info("Running full-graph inference …")
            t0 = time.perf_counter()
            self.model.eval()

            with torch.no_grad():
                result = self.model(
                    self.x, self.edge_index, self.temporal, self.coverage_mask,
                    return_attention=True,
                )

            if isinstance(result, tuple):
                logits, attentions = result
            else:
                logits, attentions = result, []

            dt = (time.perf_counter() - t0) * 1000
            self._logit_cache = logits.cpu().float().numpy()

            if attentions:
                try:
                    ei_out, attn_w = attentions[-1]
                    self._attn_edge_index = ei_out.cpu().numpy()
                    self._attn_weights    = attn_w.mean(1).cpu().numpy()
                    log.info(
                        f"Attention cached: {self._attn_edge_index.shape[1]:,} edges, "
                        f"{attn_w.shape[1]} heads"
                    )
                except Exception as e:
                    log.warning(f"Could not cache attention weights: {e}")

            log.info(
                f"Full-graph done in {dt:.0f}ms  "
                f"range=[{self._logit_cache.min():.2f},{self._logit_cache.max():.2f}]"
            )
            return self._logit_cache

    def _prob(self, logit: float) -> float:
        """Calibrated probability via temperature scaling."""
        return float(1.0 / (1.0 + np.exp(-logit / self.temperature)))

    def _top_neighbors(self, node_idx: int, logits: np.ndarray,
                       n_top: int = 5) -> List[Dict]:
        """
        Return top-N neighbours by attention weight.
        Uses pre-cached attention — O(mask + sort) instead of O(full graph).
        """
        if self._attn_edge_index is not None and self._attn_weights is not None:
            try:
                ei_np     = self._attn_edge_index
                aw_np     = self._attn_weights
                dest_mask = ei_np[1] == node_idx
                src       = ei_np[0, dest_mask]
                aw        = aw_np[dest_mask]
                order     = np.argsort(aw)[::-1][:n_top]
                return [
                    {
                        "package"         : self.idx_to_name.get(si, f"node_{si}"),
                        "risk_prob"       : round(self._prob(float(logits[si])), 4),
                        "attention_weight": round(float(ai), 4),
                        "is_vulnerable"   : (
                            bool(self.node_labels[si].item())
                            if self.node_labels is not None else None
                        ),
                    }
                    for si, ai in zip(src[order].tolist(), aw[order].tolist())
                ]
            except Exception as e:
                log.debug(f"Attention lookup failed for node {node_idx}: {e}")

        # Fallback: degree-based neighbours
        try:
            ei_np  = self.edge_index.cpu().numpy()
            mask   = ei_np[0] == node_idx
            nb_idx = ei_np[1, mask][:n_top]
            return [
                {
                    "package"         : self.idx_to_name.get(int(ni), f"node_{ni}"),
                    "risk_prob"       : round(self._prob(float(logits[ni])), 4),
                    "attention_weight": 0.0,
                    "is_vulnerable"   : (
                        bool(self.node_labels[ni].item())
                        if self.node_labels is not None else None
                    ),
                }
                for ni in nb_idx
            ]
        except Exception as e:
            log.debug(f"Fallback neighbour lookup failed: {e}")
            return []

    # ── FIX-7: npm registry search for dynamic alternative discovery ───────────

    def _npm_search_candidates(self, pkg: str, n: int = 10) -> List[str]:
        """
        Query the npm registry full-text search for packages semantically
        related to `pkg`.  Results are *not* OSV-validated here — callers
        must validate.  Returns up to `n` package names (pkg itself excluded).

        This is the zero-hardcode Stage-2 source.  The search endpoint uses
        npm's own relevance scoring which encodes keyword, description, and
        dependent-package similarity — the same signals that make "rambda"
        and "just" appear when you search for "ramda".
        """
        try:
            encoded = urllib.parse.quote(pkg, safe="")
            url = (
                f"https://registry.npmjs.org/-/v1/search"
                f"?text={encoded}&size={n + 1}"   # +1 in case pkg itself appears
            )
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent"  : "deeprisk-scanner/1.0",
                    "Accept"      : "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=4) as resp:
                data = json.loads(resp.read())
            return [
                obj["package"]["name"]
                for obj in data.get("objects", [])
                if obj["package"]["name"] != pkg
            ][:n]
        except Exception as e:
            log.debug(f"npm search failed for '{pkg}': {e}")
            return []

    # ── FIX-4 + FIX-5: fully dynamic alternatives for HIGH and MEDIUM ─────────

    def _get_alternatives(
        self,
        pkg       : str,
        logits    : np.ndarray,
        risk_label: str,
    ) -> List[str]:
        """
        Return up to 3 safe alternatives, ranked ascending by risk score.
        Operates on both HIGH and MEDIUM packages (was HIGH-only — FIX-4).

        How it works — zero hardcoded names:
        ─────────────────────────────────────
        The npm dependency graph already encodes semantic proximity: chalk and
        colors share edges because they appear together in thousands of
        package.json files.  When chalk scores HIGH we walk its neighbours,
        score each, and return those below threshold.  No hints dict is needed.

        Stage 1 — graph neighbours below threshold (pure model signal)
          All neighbours of `pkg` are scored via calibrated logits (already in
          RAM — zero extra inference).  Sorted ascending so safest appear first.
          A neighbour is accepted when:
            • it has a name in idx_to_name
            • its calibrated probability < self.threshold
            • it is not the queried package itself

        Stage 2 — npm search + OSV validation
          Triggered when Stage 1 yields < 3 results AND settings.OSV_FALLBACK
          is enabled.  npm registry is queried for semantically similar packages.
          Each candidate is checked:
            a) If it's in the graph → use the model score.
            b) If it's not in the graph → query OSV; accept if vuln_count == 0.
          Packages already collected in Stage 1 are de-duplicated.

        Why this is correct for MEDIUM:
          A MEDIUM package may be only borderline safe.  Its neighbours that
          score LOW are genuine safer alternatives discovered by the model.
          Blocking alternatives for MEDIUM was the regression that caused
          the blank "💡 Safer alternatives" block.
        """
        if risk_label not in ("HIGH", "MEDIUM"):
            return []

        idx = self.name_to_idx.get(pkg)
        candidates: List[str] = []

        # ── Stage 1: graph neighbours below threshold ─────────────────────────
        if idx is not None:
            ei_np = self.edge_index.cpu().numpy()
            mask  = ei_np[0] == idx
            nb_idxs = ei_np[1, mask]

            # Sort all neighbours ascending by calibrated probability so the
            # safest alternatives bubble up first.
            scored: List[Tuple[int, float]] = sorted(
                [
                    (int(ni), self._prob(float(logits[ni])))
                    for ni in nb_idxs
                    if int(ni) != idx          # exclude self-loop
                ],
                key=lambda t: t[1],
            )

            for ni, pr in scored:
                if len(candidates) >= 3:
                    break
                name = self.idx_to_name.get(ni)
                if not name:
                    continue   # unnamed node — handled in Stage 2
                if name == pkg:
                    continue   # never recommend the queried package itself
                if pr < self.threshold and name not in candidates:
                    candidates.append(name)

        # ── Stage 2: npm search → model or OSV validation ────────────────────
        if len(candidates) < 3 and settings.OSV_FALLBACK:
            npm_hits = self._npm_search_candidates(pkg, n=12)
            for name in npm_hits:
                if len(candidates) >= 3:
                    break
                if name == pkg or name in candidates:
                    continue

                ni = self.name_to_idx.get(name)
                if ni is not None:
                    # Package is in the graph — trust the model score
                    pr = self._prob(float(logits[ni]))
                    if pr < self.threshold:
                        candidates.append(name)
                        log.debug(
                            f"Stage-2 graph hit: {name}  prob={pr:.3f} "
                            f"(threshold={self.threshold:.3f})"
                        )
                else:
                    # Package not in graph — validate via OSV
                    osv_data = query_osv(name)
                    if osv_data and osv_data.get("queried") and osv_data.get("vuln_count", 1) == 0:
                        candidates.append(name)
                        log.debug(f"Stage-2 OSV hit: {name}  vuln_count=0")

        return candidates[:3]

    # ── FIX-2: dynamic OSV-based safe override ────────────────────────────────

    def _apply_known_safe_override(
        self,
        pkg      : str,
        prob     : float,
        label    : str,
        tier     : str,
        warnings : List[str],
        osv_data : Optional[Dict],
    ) -> Tuple[float, str, str, List[str]]:
        """
        Downgrade HIGH → MEDIUM when live OSV data reports zero vulnerabilities.
        Relies on the OSV response already fetched during predict() — no extra
        network call, no hardcoded package names.
        """
        if label != "HIGH":
            return prob, label, tier, warnings

        if (
            osv_data is not None
            and osv_data.get("queried", False)
            and osv_data.get("vuln_count", -1) == 0
        ):
            adj_prob  = self.threshold - 0.05
            adj_label = "MEDIUM"
            adj_tier  = (
                f"MEDIUM — model predicted HIGH but OSV.dev reports 0 known "
                f"vulnerabilities for {pkg}. Verify independently."
            )
            new_warns = list(warnings) + [
                f"Model over-fired (prob={prob:.3f}). "
                f"OSV.dev reports 0 vulns for '{pkg}'. "
                f"Downgraded HIGH → MEDIUM. Always verify."
            ]
            log.info(
                f"OSV-based safe override: {pkg}  {prob:.3f} HIGH → MEDIUM "
                f"(OSV vuln_count=0)"
            )
            return adj_prob, adj_label, adj_tier, new_warns

        return prob, label, tier, warnings

    # ── Public predict API ─────────────────────────────────────────────────────

    def predict(
        self,
        package       : str,
        with_neighbors: bool = True,
        check_cache   : bool = True,
    ) -> PredictionResult:
        t0  = time.perf_counter()
        pkg = package.lower().strip()

        # ── L1/L2 cache hit ───────────────────────────────────────────────────
        if check_cache:
            hit = cache.get(pkg)
            if hit:
                r              = PredictionResult(**{
                    k: v for k, v in hit.items()
                    if k != "osv_summary"   # computed property, not a field
                })
                r.cached       = True
                r.inference_ms = round((time.perf_counter() - t0) * 1000, 2)
                return r

        # ── Package not in graph — OSV fallback ───────────────────────────────
        idx = self.name_to_idx.get(pkg)
        if idx is None:
            osv_data = query_osv(pkg) if settings.OSV_FALLBACK else None
            osv_lbl  = osv_risk_label(osv_data) if osv_data else "UNKNOWN"
            osv_prob = (
                0.85 if osv_lbl == "HIGH"   else
                0.45 if osv_lbl == "MEDIUM" else
                0.15
            )
            label, tier = _tier(
                osv_prob if osv_lbl != "UNKNOWN" else 0.5, self.threshold
            )
            if osv_lbl == "UNKNOWN":
                label = "UNKNOWN"
                tier  = "Not in dataset — manual review needed"

            # Still try to find alternatives via npm+OSV even for unknown pkgs
            alts = self._get_alternatives(pkg, np.array([]), label) \
                if label in ("HIGH", "MEDIUM") else []

            dt     = (time.perf_counter() - t0) * 1000
            result = PredictionResult(
                package=pkg, risk_score=int(osv_prob * 100), risk_prob=osv_prob,
                risk_label=label, risk_tier=tier,
                threshold=self.threshold, temperature=self.temperature,
                in_dataset=False, val_auc=self.val_auc, inference_ms=round(dt, 2),
                osv=osv_data, cached=False, alternatives=alts,
                warnings=["Package not in training graph — OSV.dev result shown"],
                explanation=f"Not in dataset. OSV result: {osv_lbl}.",
            )
            cache.set(pkg, result.to_dict())
            return result

        # ── Graph-based prediction ─────────────────────────────────────────────
        logits      = self._run_full_graph()
        raw_prob    = self._prob(float(logits[idx]))
        label, tier = _tier(raw_prob, self.threshold)
        neighbors   = self._top_neighbors(idx, logits) if with_neighbors else []

        # FIX-5: query OSV for HIGH *and* MEDIUM (was HIGH-only).
        # MEDIUM packages may have real CVEs that should surface in the report.
        # LOW packages are skipped to keep warm-path latency near zero.
        osv_data: Optional[Dict] = None
        if settings.OSV_FALLBACK and label in ("HIGH", "MEDIUM"):
            osv_data = query_osv(pkg)

        # FIX-4: alternatives for HIGH and MEDIUM, fully dynamic
        alts = self._get_alternatives(pkg, logits, label)

        warns: List[str] = []

        # Ground-truth mismatch warnings (dev/debug)
        if self.node_labels is not None:
            gt = int(self.node_labels[idx].item())
            if label == "HIGH" and gt == 0:
                warns.append(
                    "Model says HIGH but ground-truth label is safe — possible FP"
                )
            elif label in ("LOW", "MEDIUM") and gt == 1:
                warns.append(
                    "Model says LOW/MEDIUM but ground-truth is vulnerable — possible FN"
                )

        # FIX-2: dynamic override using live OSV data
        prob, label, tier, warns = self._apply_known_safe_override(
            pkg, raw_prob, label, tier, warns, osv_data
        )

        # Re-compute alternatives after possible label downgrade so a
        # HIGH→MEDIUM override still surfaces alternatives
        if not alts and label in ("HIGH", "MEDIUM"):
            alts = self._get_alternatives(pkg, logits, label)

        expl = _explanation(pkg, prob, self.threshold, neighbors, osv_data)
        dt   = (time.perf_counter() - t0) * 1000

        result = PredictionResult(
            package=pkg, risk_score=int(round(prob * 100)), risk_prob=round(prob, 4),
            risk_label=label, risk_tier=tier,
            threshold=self.threshold, temperature=self.temperature,
            in_dataset=True, val_auc=self.val_auc, inference_ms=round(dt, 2),
            top_neighbors=neighbors, osv=osv_data, explanation=expl,
            warnings=warns, cached=False, alternatives=alts,
        )
        cache.set(pkg, result.to_dict())
        return result

    def predict_batch(
        self,
        packages      : List[str],
        with_neighbors: bool = False,
    ) -> List[PredictionResult]:
        """Single full-graph inference, then O(1) per package."""
        _ = self._run_full_graph()   # warm once
        return [self.predict(p, with_neighbors=with_neighbors) for p in packages]

    def stats(self) -> Dict:
        logits = self._run_full_graph()
        probs  = 1.0 / (1.0 + np.exp(-logits / self.temperature))
        return {
            "n_nodes"           : int(self.x.shape[0]),
            "n_edges"           : int(self.edge_index.shape[1]),
            "n_indexed_packages": len(self.name_to_idx),
            "temperature"       : round(float(self.temperature), 4),
            "threshold"         : round(float(self.threshold),   4),
            "val_auc"           : round(float(self.val_auc),      4),
            "best_epoch"        : self.epoch,
            "device"            : self.device,
            "prob_mean"         : round(float(probs.mean()), 4),
            "prob_std"          : round(float(probs.std()),  4),
            "pct_high_risk"     : round(float((probs >= self.threshold).mean()), 4),
            "attn_cached"       : self._attn_weights is not None,
            "cache"             : cache.stats(),
        }