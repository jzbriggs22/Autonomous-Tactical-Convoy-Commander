"""Shared fixtures for governance tests."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from ai_governance.alerts import AlertEngine
from ai_governance.config import GovernanceConfig
from ai_governance.dashboard import DashboardBuilder
from ai_governance.drift import DriftDetector
from ai_governance.ingestion import IngestRequest, IngestionLayer
from ai_governance.storage import DecisionRecord, GovernanceDB


@pytest.fixture
def config() -> GovernanceConfig:
    return GovernanceConfig.default_customer_service()


@pytest.fixture
def db() -> GovernanceDB:
    return GovernanceDB(":memory:")


@pytest.fixture
def ingestion(config, db) -> IngestionLayer:
    return IngestionLayer(config, db)


@pytest.fixture
def detector(config, db) -> DriftDetector:
    return DriftDetector(config, db)


@pytest.fixture
def engine(config, db) -> AlertEngine:
    return AlertEngine(config, db)


@pytest.fixture
def dashboard(config, db, detector, engine) -> DashboardBuilder:
    return DashboardBuilder(config, db, detector, engine)


def make_request(
    case_id: str = None,
    category: str = "billing_dispute",
    decision: str = "resolve",
    rt_ms: int = 500,
    metadata: dict = None,
    ground_truth: str = None,
) -> IngestRequest:
    return IngestRequest(
        case_id=case_id or str(uuid.uuid4()),
        case_category=category,
        decision=decision,
        resolution_time_ms=rt_ms,
        metadata=metadata or {},
        ground_truth=ground_truth,
        timestamp=datetime.now(timezone.utc),
    )


def seed_decisions(
    ingestion: IngestionLayer,
    n: int,
    category: str,
    decision: str = "resolve",
    is_baseline: bool = False,
    rt_ms: int = 500,
) -> list:
    """Insert n decisions with sequential timestamps (oldest first)."""
    now = datetime.now(timezone.utc)
    results = []
    for i in range(n):
        ts = now - timedelta(hours=n - i)
        req = IngestRequest(
            case_id=str(uuid.uuid4()),
            case_category=category,
            decision=decision,
            resolution_time_ms=rt_ms,
            timestamp=ts,
        )
        results.append(ingestion.ingest(req))
    return results
