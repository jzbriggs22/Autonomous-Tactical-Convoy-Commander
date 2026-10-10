"""Constrained-decoding layer for structured agent decisions.

Every agent decision flows through GovernanceDecision, a typed Pydantic
model that eliminates free-text parsing across the drift detector.

Backend priority for LLM integration:
  1. outlines  — token-level constraint via outlines.Generator
  2. instructor — post-hoc structured extraction via Anthropic/OpenAI SDK
  3. Pydantic   — direct validation (always available; used in tests)

For decision validation without an LLM (the common path in our system),
outlines_core.json_schema.build_regex_from_schema generates a regex from
the GovernanceDecision JSON schema.  Every incoming payload is matched
against that regex before Pydantic sees it — invalid structure is caught
immediately with a clear error, not a cryptic KeyError deep in the drift
detector.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator

try:
    from outlines_core.json_schema import build_regex_from_schema as _build_regex
    _OUTLINES_CORE_AVAILABLE = True
except ImportError:  # pragma: no cover
    _OUTLINES_CORE_AVAILABLE = False

try:
    import outlines as _outlines
    _OUTLINES_AVAILABLE = True
except ImportError:  # pragma: no cover
    _OUTLINES_AVAILABLE = False

try:
    import instructor as _instructor
    _INSTRUCTOR_AVAILABLE = True
except ImportError:  # pragma: no cover
    _INSTRUCTOR_AVAILABLE = False


# ── typed decision model ──────────────────────────────────────────────────────

class GovernanceDecision(BaseModel):
    """The canonical structured output from any governed agent.

    All decisions ingested into the governance system must conform to this
    schema.  When an LLM backs the agent, outlines or instructor constrains
    token generation so the model *cannot* produce an invalid structure.
    When decisions are synthesised programmatically (tests, rules engines),
    Pydantic validation enforces the same contract.
    """

    case_category: str = Field(
        description="Slug identifying the case type (e.g. 'fraud_claim', 'billing_dispute')"
    )
    risk_level: Literal["low", "medium", "high", "critical"] = Field(
        description="Agent's self-assessed risk level for this decision"
    )
    decision: Literal["resolve", "escalate", "deny", "defer", "partial_resolve"] = Field(
        description="Action taken on the case"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in the decision (0 = uncertain, 1 = certain)"
    )
    flags: list[str] = Field(
        default_factory=list,
        description="Optional signal tags (e.g. 'high_value_customer', 'review_required')"
    )

    @property
    def is_high_risk(self) -> bool:
        return self.risk_level in ("high", "critical")

    @model_validator(mode="after")
    def _validate_category_nonempty(self) -> "GovernanceDecision":
        if not self.case_category or not self.case_category.strip():
            raise ValueError("case_category must be a non-empty string")
        return self


# ── decoder ───────────────────────────────────────────────────────────────────

class DecodeError(ValueError):
    """Raised when a raw payload cannot be parsed into GovernanceDecision."""


class DecisionDecoder:
    """Validates raw agent output into typed GovernanceDecision objects.

    Instantiation pre-compiles the outlines_core regex from the
    GovernanceDecision JSON schema so subsequent decode() calls pay
    only a regex-match cost, not a schema-compilation cost.
    """

    def __init__(self) -> None:
        self._schema_json = json.dumps(GovernanceDecision.model_json_schema())
        if _OUTLINES_CORE_AVAILABLE:
            self._regex: Optional[re.Pattern] = re.compile(
                _build_regex(self._schema_json)
            )
        else:
            self._regex = None

    # ── validation (no LLM required) ─────────────────────────────────────────

    def decode(self, raw: str | dict | Any) -> GovernanceDecision:
        """Parse and validate a raw payload into GovernanceDecision.

        Accepts:
          - dict  — validated directly by Pydantic
          - str   — expected to be JSON; regex-checked first if outlines_core
                    is available, then Pydantic-validated
          - anything else with .model_dump() (e.g. another BaseModel)
        """
        if isinstance(raw, GovernanceDecision):
            return raw

        if hasattr(raw, "model_dump"):
            raw = raw.model_dump()

        if isinstance(raw, dict):
            try:
                return GovernanceDecision.model_validate(raw)
            except Exception as exc:
                raise DecodeError(f"Invalid GovernanceDecision payload: {exc}") from exc

        if isinstance(raw, str):
            raw = raw.strip()
            if self._regex is not None and not self._regex.fullmatch(raw):
                raise DecodeError(
                    f"Payload does not match GovernanceDecision schema.\n"
                    f"Expected JSON matching: {self._schema_json[:120]}...\n"
                    f"Got: {raw[:200]}"
                )
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise DecodeError(f"Payload is not valid JSON: {exc}") from exc
            try:
                return GovernanceDecision.model_validate(data)
            except Exception as exc:
                raise DecodeError(f"Invalid GovernanceDecision fields: {exc}") from exc

        raise DecodeError(
            f"Cannot decode type {type(raw).__name__!r} into GovernanceDecision. "
            "Pass a dict, JSON string, or GovernanceDecision instance."
        )

    def schema_json(self) -> str:
        """Return the JSON schema for GovernanceDecision (useful for LLM prompts)."""
        return self._schema_json

    # ── LLM integration — outlines ────────────────────────────────────────────

    @staticmethod
    def wrap_anthropic_outlines(client: Any) -> Any:
        """Wrap an Anthropic client with outlines to constrain output to GovernanceDecision.

        Usage::

            import anthropic, outlines
            raw_client = anthropic.Anthropic()
            gen = DecisionDecoder.wrap_anthropic_outlines(raw_client)
            decision = gen(
                model="claude-opus-4-5",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
            )
            assert isinstance(decision, GovernanceDecision)
        """
        if not _OUTLINES_AVAILABLE:
            raise ImportError(
                "outlines is required for wrap_anthropic_outlines(). "
                "pip install outlines"
            )
        model = _outlines.from_anthropic(client)
        return _outlines.Generator(model, GovernanceDecision)

    @staticmethod
    def wrap_openai_outlines(client: Any) -> Any:
        """Wrap an OpenAI client with outlines to constrain output to GovernanceDecision."""
        if not _OUTLINES_AVAILABLE:
            raise ImportError(
                "outlines is required for wrap_openai_outlines(). "
                "pip install outlines"
            )
        model = _outlines.from_openai(client)
        return _outlines.Generator(model, GovernanceDecision)

    # ── LLM integration — instructor fallback ────────────────────────────────

    @staticmethod
    def wrap_anthropic_instructor(client: Any, mode: str = "ANTHROPIC_JSON") -> Any:
        """Wrap an Anthropic client with instructor as a fallback to outlines.

        Usage::

            import anthropic, instructor
            raw = anthropic.Anthropic()
            patched = DecisionDecoder.wrap_anthropic_instructor(raw)
            decision = patched.messages.create(
                model="claude-opus-4-5",
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
                response_model=GovernanceDecision,
            )
        """
        if not _INSTRUCTOR_AVAILABLE:
            raise ImportError(
                "instructor is required for wrap_anthropic_instructor(). "
                "pip install instructor"
            )
        mode_obj = _instructor.Mode[mode]
        return _instructor.from_anthropic(client, mode=mode_obj)

    @staticmethod
    def wrap_openai_instructor(client: Any) -> Any:
        """Wrap an OpenAI client with instructor as a fallback to outlines."""
        if not _INSTRUCTOR_AVAILABLE:
            raise ImportError(
                "instructor is required for wrap_openai_instructor(). "
                "pip install instructor"
            )
        return _instructor.from_openai(client)


# ── module-level singleton ────────────────────────────────────────────────────

#: Default decoder instance — import and call .decode() directly.
decoder = DecisionDecoder()
