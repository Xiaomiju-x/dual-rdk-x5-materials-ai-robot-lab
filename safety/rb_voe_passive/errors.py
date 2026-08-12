"""Typed errors for passive bundle validation and evidence writing."""

from __future__ import annotations


class PassiveAuditError(ValueError):
    """A structured failure that is safe to expose in an audit finding."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class BundleInvalid(PassiveAuditError):
    """The sealed input cannot be trusted or evaluated."""


class EvidenceError(PassiveAuditError):
    """The isolated evidence destination is not safe or writable."""


class TrustPolicyError(PassiveAuditError):
    """The explicit v2 trust policy is invalid, unsafe, or expired."""


class PathPolicyError(PassiveAuditError):
    """A v2 path falls outside the policy-confined local roots."""
