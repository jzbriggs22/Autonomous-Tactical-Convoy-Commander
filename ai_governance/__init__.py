"""AI agent observability and governance MVP.

Detect when AI agents silently drift on high-risk cases —
even when aggregate metrics look healthy.
"""

from .alerts import AlertEngine
from .audit import AuditLog
from .auth import APIKey, AuthMiddleware, configure as configure_auth
from .config import GovernanceConfig
from .correlation import CorrelationMiddleware, get_correlation_id
from .dashboard import DashboardBuilder
from .drift import DriftDetector
from .ingestion import IngestionLayer
from .reports import ComplianceReport, ReportGenerator
from .scheduler import DriftScheduler
from .storage import GovernanceDB
from .structured import DecisionDecoder, DecodeError, GovernanceDecision
from .test_runner import GovernanceTestRunner
from .webhooks import WebhookDispatcher

__all__ = [
    "AlertEngine",
    "APIKey",
    "AuditLog",
    "AuthMiddleware",
    "ComplianceReport",
    "configure_auth",
    "CorrelationMiddleware",
    "DashboardBuilder",
    "DecisionDecoder",
    "DecodeError",
    "DriftDetector",
    "DriftScheduler",
    "get_correlation_id",
    "GovernanceConfig",
    "GovernanceDB",
    "GovernanceDecision",
    "GovernanceTestRunner",
    "IngestionLayer",
    "ReportGenerator",
    "WebhookDispatcher",
]
