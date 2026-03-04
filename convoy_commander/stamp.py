"""Reproducibility stamp: git hash, python version, platform, config.

Every simulation run records provenance metadata so results can be
reproduced or compared across machines and code versions.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class ReproStamp:
    """Immutable reproducibility metadata for a simulation run."""

    git_commit: str
    git_dirty: bool
    python_version: str
    platform_info: str
    package_version: str
    seed: int
    scenario: str
    config_dict: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def collect_stamp(config: Any) -> ReproStamp:
    """Gather reproducibility metadata. Safe to call in any environment."""
    from convoy_commander import __version__

    git_commit = "unknown"
    git_dirty = False
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        git_dirty = len(status) > 0
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    return ReproStamp(
        git_commit=git_commit,
        git_dirty=git_dirty,
        python_version=platform.python_version(),
        platform_info=platform.platform(),
        package_version=__version__,
        seed=config.seed,
        scenario=config.scenario or "unknown",
        config_dict=config.model_dump(),
    )
