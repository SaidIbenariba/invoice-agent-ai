"""Score a batch against ground truth: decisions, reasons, PO matching, safety, extraction."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from invoice_ai.evaluate import compare

from .models import AgentInvoice, norm_bank, norm_po


def load_truth(data_dir: Path) -> dict[str, dict]:
    d = data_dir / "truth"
    return {p.stem: json.loads(p.read_text()) for p in sorted(d.glob("*.json"))} if d.exists() else {}


def score(cases: list[dict], truth: dict[str, dict]) -> dict:
    rows = []
    for c in cases:
        t = truth.get(c["doc_id"])
        if not t:
            continue
        exp = t["expected"]
        got_reasons = set(c.get("reasons") or [])
        decision_ok = c["decision"] == exp["decision"]
        reasons_ok = decision_ok and (exp["decision"] == "APPROVE" or set(exp["reasons"]) <= got_reasons)
        po_ok = exp["decision"] != "APPROVE" or (c["decision"] == "APPROVE" and c.get("po_number") == exp["po_number"])
        wrong_payment = c["decision"] == "APPROVE" and (exp["decision"] != "APPROVE" or not po_ok)
        pred = AgentInvoice.model_validate(c["invoice"]) if c.get("invoice") else AgentInvoice()
        true_inv = AgentInvoice.model_validate(t["invoice"])
        cmp = compare(pred, true_inv)
        po_read = norm_po(pred.po_number) == norm_po(true_inv.po_number)
        bank_read = norm_bank(pred.bank_account) == norm_bank(true_inv.bank_account)
        rows.append({
            "doc_id": c["doc_id"], "scenario": t["scenario"], "photo": t["photo"], "layout": t["layout"],
            "expected": exp["decision"], "expected_reasons": exp["reasons"], "expected_po": exp["po_number"],
            "got": c["decision"], "got_reasons": sorted(got_reasons), "got_po": c.get("po_number"),
            "decision_ok": decision_ok, "reasons_ok": reasons_ok, "po_ok": po_ok, "wrong_payment": wrong_payment,
            "tool_calls": c.get("tool_calls", 0), "guard_blocks": c.get("guard_blocks", 0),
            "tool_errors": c.get("tool_errors", 0), "seconds": c.get("seconds", 0.0),
            "extraction_fully_correct": cmp.fully_correct and po_read and bank_read,
            "header_fields_correct": sum(cmp.fields.values()), "header_fields": len(cmp.fields),
            "line_items_correct": cmp.lines_correct, "po_read": po_read, "bank_read": bank_read,
            "extract_seconds": c.get("extract_seconds"), "route": c.get("route"),
            "email_drafted": any(s["tool"] == "draft_email_to_supplier" and s["result"].get("saved")
                                 for s in c.get("steps", [])),
        })
    n = len(rows)
    approve = [r for r in rows if r["expected"] == "APPROVE"]
    problems = [r for r in rows if r["expected"] != "APPROVE"]
    no_po = [r for r in rows if r["scenario"] in ("po_missing_unique", "po_typo")]
    by = defaultdict(lambda: {"n": 0, "correct": 0})
    for r in rows:
        by[r["scenario"]]["n"] += 1
        by[r["scenario"]]["correct"] += r["decision_ok"] and r["reasons_ok"] and r["po_ok"]

    def avg(key, sub=rows):
        vals = [r[key] for r in sub if r[key] is not None]
        return round(sum(vals) / len(vals), 1) if vals else None

    return {
        "summary": {
            "invoices": n,
            "decisions_correct": sum(r["decision_ok"] for r in rows),
            "decisions_and_reasons_correct": sum(r["decision_ok"] and r["reasons_ok"] and r["po_ok"] for r in rows),
            "wrong_payments": sum(r["wrong_payment"] for r in rows),
            "clean_invoices": len(approve),
            "clean_invoices_approved": sum(r["got"] == "APPROVE" and r["po_ok"] for r in approve),
            "problem_invoices": len(problems),
            "problem_invoices_stopped": sum(r["got"] != "APPROVE" for r in problems),
            "problem_reason_correct": sum(r["reasons_ok"] for r in problems),
            "po_found_without_valid_reference": f"{sum(r['po_ok'] and r['got'] == 'APPROVE' for r in no_po)}/{len(no_po)}",
            "duplicates_rejected": f"{sum(r['got'] == 'REJECT' for r in rows if r['scenario'] == 'duplicate')}"
                                   f"/{sum(r['scenario'] == 'duplicate' for r in rows)}",
            "supplier_email_drafted_bank_change": f"{sum(r['email_drafted'] for r in rows if r['scenario'] == 'bank_change')}"
                                                  f"/{sum(r['scenario'] == 'bank_change' for r in rows)}",
            "supplier_email_drafted_price_or_qty": f"{sum(r['email_drafted'] for r in rows if r['scenario'] in ('price_over_po', 'qty_over_received'))}"
                                                   f"/{sum(r['scenario'] in ('price_over_po', 'qty_over_received') for r in rows)}",
            "guard_blocks": sum(r["guard_blocks"] for r in rows),
            "tool_errors": sum(r["tool_errors"] for r in rows),
            "avg_tool_calls": avg("tool_calls"),
            "avg_agent_seconds": avg("seconds"),
            "extraction_fully_correct": sum(r["extraction_fully_correct"] for r in rows),
            "header_field_accuracy": round(sum(r["header_fields_correct"] for r in rows)
                                           / max(1, sum(r["header_fields"] for r in rows)), 4),
            "line_items_correct": sum(r["line_items_correct"] for r in rows),
            "po_reference_read": sum(r["po_read"] for r in rows),
            "bank_account_read": sum(r["bank_read"] for r in rows),
            "avg_extract_seconds": {route: avg("extract_seconds", [r for r in rows if r["route"] == route])
                                    for route in ("text", "vision")},
        },
        "by_scenario": dict(sorted(by.items())),
        "cases": rows,
    }
