"""Optional dependency guards for weather API packages."""

from __future__ import annotations

from typing import Any


def require_requests() -> Any:
    """Return the requests module or raise with install instructions."""
    try:
        import requests
        return requests
    except ImportError:
        raise ImportError(
            "requests is required for weather API integration. "
            "Install with: pip install 'convoy_commander[weather]'"
        ) from None
