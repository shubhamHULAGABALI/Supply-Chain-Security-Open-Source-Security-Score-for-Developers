"""backend/schemas/__init__.py — All Pydantic request/response models."""
from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    package: str = Field(..., min_length=1, max_length=214,
                         description="npm package name, e.g. 'lodash' or 'lodash@4.17.21'",
                         examples=["lodash"])
    with_neighbors: bool = Field(True, description="Include top-5 graph neighbours")

    @field_validator("package")
    @classmethod
    def clean(cls, v: str) -> str:
        # Strip version suffix and normalise
        return v.split("@")[0].strip().lower().replace(" ", "-")


class BatchPredictRequest(BaseModel):
    packages: List[str] = Field(..., min_length=1, max_length=50)
    with_neighbors: bool = False

    @field_validator("packages")
    @classmethod
    def clean_all(cls, names: List[str]) -> List[str]:
        return [n.split("@")[0].strip().lower() for n in names]


class NeighbourInfo(BaseModel):
    package: str
    risk_prob: float
    attention_weight: float
    is_vulnerable: Optional[bool]


class OSVInfo(BaseModel):
    vuln_count: int
    critical_count: int
    cve_ids: List[str]
    source: str
    queried: bool


class PredictResponse(BaseModel):
    package: str
    risk_score: int                    # 0–100
    risk_prob: float                   # 0.0–1.0
    risk_label: str                    # HIGH | MEDIUM | LOW | UNKNOWN
    risk_tier: str                     # human description
    threshold: float
    temperature: float
    in_dataset: bool
    val_auc: float
    inference_ms: float
    top_neighbors: List[NeighbourInfo] = []
    osv: Optional[OSVInfo]             = None
    explanation: str                   = ""
    warnings: List[str]                = []
    cached: bool                       = False
    alternatives: List[str]            = []        # safer alternatives for HIGH risk


class BatchPredictResponse(BaseModel):
    results: List[PredictResponse]
    total_ms: float
    high_risk_count: int
    packages_in_dataset: int


class StatsResponse(BaseModel):
    n_nodes: int
    n_edges: int
    n_indexed_packages: int
    temperature: float
    threshold: float
    val_auc: float
    best_epoch: int
    cache: dict
    device: str


class HealthResponse(BaseModel):
    status: str
    model_ready: bool
    startup_ms: float
    startup_error: Optional[str]
    version: str = "1.0.0"
