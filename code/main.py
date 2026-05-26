"""
main.py  —  End-to-end support triage pipeline.

Reads  : support_tickets.csv
Writes : support_tickets/output.csv

Pipeline per row
────────────────
  1. Parse issue JSON  → conversation_text
  2. safety.py         → injection / PII flags
  3. retriever.py      → relevant corpus docs
  4. classifier.py     → language, true_company, product_area, etc.
  5. router.py         → status (replied / escalated)
  6. responder.py      → response, justification, confidence, actions
  7. Assemble 14-column output row

Parallelism: ThreadPoolExecutor(max_workers=6) so the 3-minute
budget is met even on the hidden 150-ticket test set.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import pandas as pd

# ── Add code/ to path so sibling imports work regardless of CWD ──────────────
sys.path.insert(0, str(Path(__file__).parent.absolute()))

from classifier import classify, ClassifierResult
from responder  import respond,  ResponderResult
from retriever  import Retriever, RetrievalResult
from router     import route,    RouterDecision

# safety.py lives in the same directory
try:
    from safety import analyze as safety_analyze, redact_pii, SafetyResult
    _SAFETY_AVAILABLE = True
except ImportError:
    print("[main] ⚠  safety.py not found — safety checks disabled")
    _SAFETY_AVAILABLE = False

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

INPUT_CSV   = Path("support_tickets.csv")
OUTPUT_DIR  = Path("support_tickets")
OUTPUT_CSV  = OUTPUT_DIR / "output.csv"
DATA_ROOT   = Path("data")
MAX_WORKERS = 6      # parallel threads; 6 keeps Gemini rate limits comfortable

# Output column order (must match validate_output.py expectations)
OUTPUT_COLUMNS = [
    "status",
    "product_area",
    "response",
    "justification",
    "request_type",
    "confidence_score",
    "source_documents",
    "risk_level",
    "pii_detected",
    "language",
    "actions_taken",
]

# ──────────────────────────────────────────────────────────────────────────────
# Global retriever (built once at startup, shared across threads)
# ──────────────────────────────────────────────────────────────────────────────

_RETRIEVER: Optional[Retriever] = None


def _get_retriever() -> Retriever:
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = Retriever(data_root=DATA_ROOT)
    return _RETRIEVER


# ──────────────────────────────────────────────────────────────────────────────
# Conversation parser
# ──────────────────────────────────────────────────────────────────────────────

def parse_conversation(issue_raw: str) -> tuple[str, str]:
    """
    Parse the JSON-encoded issue field.

    Returns
    ───────
    (conversation_text, last_user_message)

    conversation_text  : full formatted conversation for LLM prompts
    last_user_message  : most recent user turn (used as retriever query)
    """
    try:
        messages = json.loads(issue_raw)
        if not isinstance(messages, list):
            raise ValueError("issue is not a JSON array")
    except (json.JSONDecodeError, ValueError):
        # Treat raw string as a single user message
        return issue_raw.strip(), issue_raw.strip()

    lines: list[str] = []
    last_user = ""
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role    = str(msg.get("role", "user")).upper()
        content = str(msg.get("content", "")).strip()
        if not content:
            continue
        lines.append(f"[{role}]: {content}")
        if role == "USER":
            last_user = content

    conversation_text = "\n".join(lines)
    return conversation_text, last_user or conversation_text


# ──────────────────────────────────────────────────────────────────────────────
# Per-row processing
# ──────────────────────────────────────────────────────────────────────────────

def _error_row(row_id: int, error: str) -> dict:
    """Safe default output row when processing fails completely."""
    return {
        "status":           "escalated",
        "product_area":     "unknown",
        "response":         "We encountered an issue processing your request. "
                            "A support agent will follow up shortly.",
        "justification":    f"Processing error: {error}",
        "request_type":     "product_issue",
        "confidence_score": 0.1,
        "source_documents": "",
        "risk_level":       "high",
        "pii_detected":     "false",
        "language":         "en",
        "actions_taken":    "[]",
    }


def process_row(row: pd.Series) -> dict:
    """
    Full pipeline for a single CSV row.

    Never raises — all exceptions are caught and converted to a safe
    error row so the CSV write never stalls.
    """
    row_id  = int(row.name) if hasattr(row, "name") else -1
    subject = str(row.get("subject", "") or "").strip()
    company = str(row.get("company", "") or "").strip()
    issue   = str(row.get("issue",   "") or "").strip()

    print(f"\n{'─'*60}")
    print(f"[main] 🎫 Row {row_id} | company={company!r} | subject={subject[:50]!r}")

    try:
        # ── Step 1: Parse conversation ────────────────────────────────────────
        conversation_text, last_user_msg = parse_conversation(issue)
        retriever_query = f"{subject} {last_user_msg}".strip()

        # ── Step 2: Safety check ──────────────────────────────────────────────
        if _SAFETY_AVAILABLE:
            # Use raw (un-redacted) text for detection, then redact for LLM use
            safety: SafetyResult = safety_analyze(
                text    = f"{subject}\n{conversation_text}",
                company = company,
            )
            pii_detected   = safety.pii_detected
            is_injection   = safety.is_injection
            injection_reasons = safety.injection_reasons
            safety_flags   = safety.summary

            # Redact PII from everything that will be sent to LLMs
            safe_conversation = redact_pii(conversation_text)
            safe_subject      = redact_pii(subject)
        else:
            # Minimal fallback if safety.py is missing
            pii_detected      = False
            is_injection      = False
            injection_reasons = []
            safety_flags      = "safety-module-unavailable"
            safe_conversation = conversation_text
            safe_subject      = subject

        print(f"[main] 🛡  safety: {safety_flags}")

        # ── Step 3: Retrieval ─────────────────────────────────────────────────
        retriever = _get_retrieval()
        try:
            retrieval_results: list[RetrievalResult] = retriever.retrieve_for_ticket(
                query   = retriever_query[:512],
                company = company,
                top_k   = 3,
            )
        except Exception as exc:
            print(f"[main] ⚠  Retriever error: {exc}")
            retrieval_results = []

        has_corpus_docs = len(retrieval_results) > 0
        print(f"[main] 📚 Retrieved {len(retrieval_results)} corpus docs")

        # ── Step 4: Classify ──────────────────────────────────────────────────
        classifier_result: ClassifierResult = classify(
            conversation_text = safe_conversation,
            subject           = safe_subject,
            company_hint      = company,
            safety_flags      = safety_flags,
            is_injection      = is_injection,
        )

        print(f"[main] 🏷  classified: company={classifier_result.true_company} "
              f"area={classifier_result.product_area} "
              f"type={classifier_result.request_type} "
              f"risk={classifier_result.risk_level} "
              f"lang={classifier_result.language}")

        # ── Step 5: Route ─────────────────────────────────────────────────────
        router_decision: RouterDecision = route(
            risk_level          = classifier_result.risk_level,
            is_injection        = is_injection,
            injection_reasons   = injection_reasons,
            pii_detected        = pii_detected,
            true_company        = classifier_result.true_company,
            request_type        = classifier_result.request_type,
            has_corpus_docs     = has_corpus_docs,
            conversation_text   = conversation_text,   # unredacted for keyword scan
        )

        print(f"[main] 🚦 routed: status={router_decision.status} "
              f"reasons={router_decision.reason_str[:80]}")

        # ── Step 6: Respond ───────────────────────────────────────────────────
        responder_result: ResponderResult = respond(
            conversation_text   = safe_conversation,
            subject             = safe_subject,
            company             = classifier_result.true_company,
            product_area        = classifier_result.product_area,
            risk_level          = classifier_result.risk_level,
            language            = classifier_result.language,
            status              = router_decision.status,
            escalation_reasons  = router_decision.reasons,
            safety_flags        = safety_flags,
            retrieval_results   = retrieval_results,
            is_injection        = is_injection,
        )

        print(f"[main] ✍️  response confidence={responder_result.confidence_score:.2f} "
              f"sources={responder_result.source_documents[:60]!r}")

        # ── Step 7: Assemble output row ───────────────────────────────────────
        return {
            "status":           router_decision.status,
            "product_area":     classifier_result.product_area,
            "response":         responder_result.response,
            "justification":    (
                f"{classifier_result.reasoning} | "
                f"Route: {router_decision.reason_str} | "
                f"{responder_result.justification}"
            ),
            "request_type":     classifier_result.request_type,
            "confidence_score": round(responder_result.confidence_score, 4),
            "source_documents": responder_result.source_documents,
            "risk_level":       classifier_result.risk_level,
            "pii_detected":     "true" if pii_detected else "false",
            "language":         classifier_result.language,
            "actions_taken":    responder_result.actions_taken,
        }

    except Exception as exc:
        print(f"[main] ❌ Row {row_id} failed: {exc}")
        traceback.print_exc()
        return _error_row(row_id, str(exc))


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()

    # ── Load input CSV ────────────────────────────────────────────────────────
    if not INPUT_CSV.exists():
        print(f"[main] ❌ Input file not found: {INPUT_CSV}")
        sys.exit(1)

    df_in = pd.read_csv(INPUT_CSV, dtype=str).fillna("")
    total = len(df_in)
    print(f"[main] 📥 Loaded {total} tickets from {INPUT_CSV}")

    # ── Pre-warm retriever (builds/loads index before threads start) ──────────
    print("[main] 🔧 Pre-warming retriever …")
    _get_retriever()

    # ── Prepare output storage ────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(exist_ok=True)
    results: dict[int, dict] = {}   # row_index → output_dict

    # ── Parallel processing ───────────────────────────────────────────────────
    print(f"[main] 🚀 Processing {total} tickets with {MAX_WORKERS} workers …\n")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {
            executor.submit(process_row, df_in.iloc[i]): i
            for i in range(total)
        }

        completed = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                print(f"[main] ❌ Unexpected error on row {idx}: {exc}")
                results[idx] = _error_row(idx, str(exc))
            completed += 1
            elapsed = time.time() - t_start
            print(f"[main] ⏱  {completed}/{total} done "
                  f"({elapsed:.1f}s elapsed, "
                  f"~{elapsed/completed*(total-completed):.0f}s remaining)")

    # ── Assemble output DataFrame ─────────────────────────────────────────────
    output_rows = [results[i] for i in range(total)]
    df_out = df_in.copy()
    for col in OUTPUT_COLUMNS:
        df_out[col] = [row[col] for row in output_rows]

    # ── Write output CSV ──────────────────────────────────────────────────────
    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    elapsed_total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"[main] ✅ Done — {total} tickets processed in {elapsed_total:.1f}s")
    print(f"[main] 📤 Output written to {OUTPUT_CSV}")
    print(f"[main] ⏱  Average per ticket: {elapsed_total/total:.2f}s")

    # Warn if we're cutting it close on the 3-minute budget
    if elapsed_total > 150:
        print(f"[main] ⚠  Processing took {elapsed_total:.0f}s "
              f"— close to the 3-minute limit. Consider reducing MAX_WORKERS "
              f"or top_k in retriever.")


if __name__ == "__main__":
    main()