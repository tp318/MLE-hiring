"""
router.py  —  Rule-based routing: replied vs escalated.

No LLM calls.  All decisions are deterministic given the same inputs,
which satisfies the "identical output on re-run" requirement.

Rules are checked in strict priority order; first match wins.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

# ──────────────────────────────────────────────────────────────────────────────
# Keyword lists  (all lowercase; matched against lowercased conversation)
# ──────────────────────────────────────────────────────────────────────────────

_LEGAL_KEYWORDS: list[str] = [
    "lawsuit", "sue you", "suing", "legal action", "take legal",
    "attorney", "lawyer", "court", "litigation", "file a complaint",
    "consumer protection", "class action", "regulatory", "gdpr complaint",
    "small claims", "arbitration", "demand letter",
]

_FRAUD_KEYWORDS: list[str] = [
    "identity theft", "stolen card", "stolen my card", "card stolen",
    "unauthorized transaction", "unauthorized charge", "fraudulent charge",
    "someone used my card", "someone hacked", "account hacked",
    "hacked my account", "unauthorized access", "account takeover",
    "compromised account", "phishing", "scammed", "scam",
]

_DATA_EXFIL_KEYWORDS: list[str] = [
    "send me all", "dump all", "export all users", "list all tickets",
    "print all data", "extract all records", "show me all entries",
]

# ──────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RouterDecision:
    status:  str                        # "replied" | "escalated"
    reasons: List[str] = field(default_factory=list)

    @property
    def reason_str(self) -> str:
        return "; ".join(self.reasons) if self.reasons else "no reason recorded"


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def route(
    risk_level:         str,
    is_injection:       bool,
    injection_reasons:  List[str],
    pii_detected:       bool,
    true_company:       str,
    request_type:       str,
    has_corpus_docs:    bool,
    conversation_text:  str,
) -> RouterDecision:
    """
    Decide whether to reply or escalate.

    Parameters
    ──────────
    risk_level          classifier output: low|medium|high|critical
    is_injection        safety layer flag
    injection_reasons   list of matched injection pattern names
    pii_detected        safety layer PII flag
    true_company        classifier output: claude|devplatform|visa|unknown
    request_type        classifier output: bug|product_issue|feature_request|invalid
    has_corpus_docs     True if retriever returned ≥1 result
    conversation_text   full lowercased conversation string for keyword scan
    """
    lower = conversation_text.lower()

    # ── Rule 1: Prompt injection ───────────────────────────────────────────────
    if is_injection:
        return RouterDecision(
            status  = "escalated",
            reasons = [
                "prompt injection detected",
                *([f"patterns: {', '.join(injection_reasons[:3])}"]
                  if injection_reasons else []),
            ],
        )

    # ── Rule 2: Critical risk level ───────────────────────────────────────────
    if risk_level == "critical":
        return RouterDecision(
            status  = "escalated",
            reasons = ["risk_level=critical"],
        )

    # ── Rule 3: Legal threat ──────────────────────────────────────────────────
    legal_hits = [kw for kw in _LEGAL_KEYWORDS if kw in lower]
    if legal_hits:
        return RouterDecision(
            status  = "escalated",
            reasons = [f"legal threat: {', '.join(legal_hits[:3])}"],
        )

    # ── Rule 4: Fraud / identity theft ────────────────────────────────────────
    fraud_hits = [kw for kw in _FRAUD_KEYWORDS if kw in lower]
    if fraud_hits:
        return RouterDecision(
            status  = "escalated",
            reasons = [f"fraud/identity theft: {', '.join(fraud_hits[:3])}"],
        )

    # ── Rule 5: Data exfiltration attempt ─────────────────────────────────────
    exfil_hits = [kw for kw in _DATA_EXFIL_KEYWORDS if kw in lower]
    if exfil_hits:
        return RouterDecision(
            status  = "escalated",
            reasons = [f"data exfiltration attempt: {', '.join(exfil_hits[:2])}"],
        )

    # ── Rule 6: High risk + no corpus docs ───────────────────────────────────
    if risk_level == "high" and not has_corpus_docs:
        return RouterDecision(
            status  = "escalated",
            reasons = ["risk_level=high with no supporting corpus documents"],
        )

    # ── Rule 7: Unknown company + dangerous topic ─────────────────────────────
    if true_company == "unknown" and risk_level in ("high", "critical"):
        return RouterDecision(
            status  = "escalated",
            reasons = ["unknown company with high/critical risk — cannot safely answer"],
        )

    # ── Rule 8: Corpus has a clear answer → reply ─────────────────────────────
    if has_corpus_docs:
        return RouterDecision(
            status  = "replied",
            reasons = ["corpus documents found; proceeding with grounded response"],
        )

    # ── Rule 9: Out of scope but harmless → reply with guidance ───────────────
    if request_type == "invalid":
        return RouterDecision(
            status  = "replied",
            reasons = ["request out of scope; replying with out-of-scope message"],
        )

    # ── Rule 10: Low / medium risk without docs → reply with caveat ───────────
    if risk_level in ("low", "medium"):
        return RouterDecision(
            status  = "replied",
            reasons = [
                f"risk_level={risk_level}; no corpus docs but low risk",
                "replying with general guidance and escalation offer",
            ],
        )

    # ── Default: escalate anything ambiguous ──────────────────────────────────
    return RouterDecision(
        status  = "escalated",
        reasons = ["ambiguous risk with no corpus support — defaulting to escalation"],
    )