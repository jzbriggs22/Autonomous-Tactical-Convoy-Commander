"""Tests for webhook HMAC-SHA256 signature verification."""

from __future__ import annotations

import json

import pytest

from ai_governance.webhooks import (
    WebhookDispatcher,
    WebhookTarget,
    compute_signature,
    verify_signature,
)


class TestSignatureComputation:
    def test_compute_signature_format(self):
        body = b'{"test": true}'
        sig = compute_signature(body, "my-secret")
        assert sig.startswith("sha256=")
        assert len(sig) == 7 + 64  # "sha256=" + 64 hex chars

    def test_same_input_same_signature(self):
        body = b'{"event": "alert"}'
        s1 = compute_signature(body, "secret")
        s2 = compute_signature(body, "secret")
        assert s1 == s2

    def test_different_secret_different_signature(self):
        body = b'{"event": "alert"}'
        s1 = compute_signature(body, "secret-a")
        s2 = compute_signature(body, "secret-b")
        assert s1 != s2

    def test_different_body_different_signature(self):
        s1 = compute_signature(b'{"a": 1}', "secret")
        s2 = compute_signature(b'{"a": 2}', "secret")
        assert s1 != s2


class TestSignatureVerification:
    def test_valid_signature(self):
        body = b'{"event": "alert"}'
        secret = "test-secret-key"
        sig = compute_signature(body, secret)
        assert verify_signature(body, secret, sig) is True

    def test_invalid_signature(self):
        body = b'{"event": "alert"}'
        assert verify_signature(body, "secret", "sha256=0000000000000000000000000000000000000000000000000000000000000000") is False

    def test_tampered_body(self):
        body = b'{"event": "alert"}'
        sig = compute_signature(body, "secret")
        tampered = b'{"event": "hacked"}'
        assert verify_signature(tampered, "secret", sig) is False

    def test_wrong_secret(self):
        body = b'{"event": "alert"}'
        sig = compute_signature(body, "correct-secret")
        assert verify_signature(body, "wrong-secret", sig) is False

    def test_empty_body(self):
        body = b""
        secret = "secret"
        sig = compute_signature(body, secret)
        assert verify_signature(body, secret, sig) is True

    def test_unicode_secret(self):
        body = b'{"data": "test"}'
        secret = "sécret-üñicode"
        sig = compute_signature(body, secret)
        assert verify_signature(body, secret, sig) is True


class TestWebhookTargetSigning:
    def test_target_without_secret(self):
        target = WebhookTarget(url="http://example.com/hook")
        assert target.signing_secret is None

    def test_target_with_secret(self):
        target = WebhookTarget(
            url="http://example.com/hook",
            signing_secret="my-secret-key",
        )
        assert target.signing_secret == "my-secret-key"

    def test_dispatcher_with_signed_target(self):
        target = WebhookTarget(
            url="http://127.0.0.1:1/unreachable",
            signing_secret="test-secret",
            max_retries=1,
            timeout_seconds=0.5,
        )
        dispatcher = WebhookDispatcher(targets=[target], worker_count=0)
        results = dispatcher.dispatch({"severity": "warn", "message": "test"})
        assert len(results) == 1
        assert results[0].status == "failed"
        dispatcher.shutdown(timeout=1.0)
