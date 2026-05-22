"""AI agent observability and governance MVP.

Detect when AI agents silently drift on high-risk cases —
even when aggregate metrics look healthy.
"""

from .alerts import AlertEngine
from .audit import AuditLog
from .auth import APIKey, AuthMiddleware, configure as configure_auth
from .config import GovernanceConfig
from .dashboard import DashboardBuilder
from .drift import DriftDetector
from .ingestion import IngestionLayer
from .storage import GovernanceDB
from .structured import DecisionDecoder, DecodeError, GovernanceDecision
from .test_runner import GovernanceTestRunner
from .webhooks import WebhookDispatcher

__all__ = [
    "AlertEngine",
    "APIKey",
    "AuditLog",
    "AuthMiddleware",
    "configure_auth",
    "DashboardBuilder",
    "DecisionDecoder",
    "DecodeError",
    "DriftDetector",
    "GovernanceConfig",
    "GovernanceDB",
    "GovernanceDecision",
    "GovernanceTestRunner",
    "IngestionLayer",
    "WebhookDispatcher",
]
