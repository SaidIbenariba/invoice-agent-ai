"""The agent's tools: JSON schemas for the model, Python functions for the work.

Read tools return facts from the invoice and the ERP. Three tools end the case:
`approve_for_payment`, `hold_for_review` and `reject_invoice`. Approval and rejection are guarded:
code re-checks the policy and refuses when it is not met, whatever the model asked for.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .erp import ERP
from .extract import to_agent_invoice
from .match import approval_blockers, identify_supplier, invoice_checks, three_way_match
from .models import REASONS, AgentInvoice, norm_bank, norm_po

TERMINAL = {"approve_for_payment", "hold_for_review", "reject_invoice"}


@dataclass
class Case:
    doc_id: str
    path: Path
    invoice: AgentInvoice | None = None
    extraction: dict | None = None
    decision: str | None = None  # APPROVE | HOLD | REJECT
    reasons: list[str] = field(default_factory=list)
    po_number: str | None = None
    note: str = ""
    steps: list[dict] = field(default_factory=list)
    guard_blocks: int = 0
    tool_errors: int = 0
    seconds: float = 0.0

    def summary(self) -> dict:
        return {"doc_id": self.doc_id, "file": self.path.name, "decision": self.decision, "reasons": self.reasons,
                "po_number": self.po_number, "note": self.note, "tool_calls": len(self.steps),
                "guard_blocks": self.guard_blocks, "tool_errors": self.tool_errors, "seconds": round(self.seconds, 1)}


def _mask(acct: str | None) -> str | None:
    n = norm_bank(acct)
    return f"****{n[-4:]}" if n else None


def _fn(name: str, desc: str, props: dict | None = None, required: list[str] | None = None) -> dict:
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props or {}, "required": required or []}}}


SPECS = [
    _fn("read_invoice", "Read the current invoice: supplier, tax ID, invoice number, dates, currency, totals, "
        "PO number, bank account and line items. No arguments."),
    _fn("check_invoice", "Check the current invoice itself: required fields, line maths, tax, totals, dates, "
        "supplier tax ID, and whether it was already paid (duplicate). No arguments: it already has the invoice."),
    _fn("find_supplier", "Find the supplier in the supplier master by tax ID and/or name.",
        {"tax_id": {"type": "string"}, "name": {"type": "string"}}),
    _fn("verify_bank_account", "Compare the bank account printed on the invoice with the one on file for "
        "this supplier.", {"supplier_id": {"type": "string"}}, ["supplier_id"]),
    _fn("get_purchase_order", "Get a purchase order by its number.",
        {"po_number": {"type": "string"}}, ["po_number"]),
    _fn("search_purchase_orders", "List the supplier's open purchase orders.",
        {"supplier_id": {"type": "string"}}, ["supplier_id"]),
    _fn("get_goods_receipts", "Quantities ordered, received and already invoiced for each line of a PO.",
        {"po_number": {"type": "string"}}, ["po_number"]),
    _fn("three_way_match", "Match the invoice against a PO and its goods receipts: lines, quantities, prices.",
        {"po_number": {"type": "string"}}, ["po_number"]),
    _fn("draft_email_to_supplier", "Save a draft email to the supplier's address on file. It is NOT sent; "
        "a person reviews it.", {"subject": {"type": "string"}, "body": {"type": "string"}}, ["subject", "body"]),
    _fn("approve_for_payment", "Approve the invoice for payment against a PO. Only when every check passed.",
        {"po_number": {"type": "string"}, "note": {"type": "string"}}, ["po_number", "note"]),
    _fn("hold_for_review", "Put the invoice on hold for a person, with reason codes and a short note.",
        {"reasons": {"type": "array", "items": {"type": "string", "enum": sorted(REASONS)}},
         "note": {"type": "string"}}, ["reasons", "note"]),
    _fn("reject_invoice", "Reject the invoice. Only for a confirmed duplicate of an invoice already paid.",
        {"note": {"type": "string"}}, ["note"]),
]


class Toolset:
    def __init__(self, erp: ERP, case: Case, read: Callable[[Path], dict]):
        self.erp, self.case, self._read = erp, case, read

    # ---------- helpers ----------

    def invoice(self) -> AgentInvoice:
        if self.case.invoice is None:
            self.case.extraction = self._read(self.case.path)
            self.case.invoice = to_agent_invoice(self.case.extraction["raw"])
        return self.case.invoice

    def supplier(self):
        return identify_supplier(self.erp, self.invoice())

    def _po_view(self, po) -> dict:
        owner = self.erp.supplier(po.supplier_id)
        return {"po_number": po.po_number, "supplier_id": po.supplier_id,
                "supplier_name": owner.name if owner else None, "order_date": po.order_date,
                "currency": po.currency, "status": po.status, "total": po.total,
                "lines": [{"line": li.line, "description": li.description, "quantity": li.quantity,
                           "unit_price": li.unit_price} for li in po.lines]}

    # ---------- dispatch ----------

    def call(self, name: str, args: dict) -> tuple[dict, bool]:
        """Run one tool. Returns (result, case_finished).

        Small models often pass extra arguments (e.g. the whole invoice to a tool that takes none). Every
        tool works on the current invoice, so arguments a tool does not take are ignored, not an error."""
        fn = getattr(self, f"t_{name}", None)
        if fn is None:
            self.case.tool_errors += 1
            return {"error": f"unknown tool '{name}'"}, False
        params = inspect.signature(fn).parameters
        args = args if isinstance(args, dict) else {}
        accepted = {k: v for k, v in args.items() if k in params}
        try:
            result, finished = fn(**accepted)
        except TypeError as e:
            self.case.tool_errors += 1
            return {"error": f"bad arguments for {name}: {e}"}, False
        return result, finished

    # ---------- read tools ----------

    # Results that call for a specific next step say so in `next_step`: the policy, restated where a
    # small model needs it (dev run 2: an invoice without a PO number was held without searching).

    def t_read_invoice(self):
        inv = self.invoice()
        d = inv.model_dump(exclude={"bank_account"})
        d["bank_account"] = _mask(inv.bank_account)
        d["missing"] = [k for k in ("supplier_name", "invoice_number", "total", "po_number") if not d.get(k)]
        if not inv.po_number:
            d["next_step"] = ("No PO number on this invoice. After find_supplier, call search_purchase_orders and "
                              "three_way_match on each open PO; approve only if exactly one is clean.")
        return d, False

    def t_check_invoice(self):
        s = self.supplier()
        issues = invoice_checks(self.erp, self.invoice(), s)
        out = {"passed": not issues, "issues": [i.as_dict() for i in issues]}
        if s is None:
            out["note"] = "supplier not on file, so the duplicate check could not run"
        if any(i.code == "duplicate" for i in issues):
            out["next_step"] = "Duplicate: call reject_invoice."
        elif issues:
            out["next_step"] = "Call hold_for_review with reason invoice_check_failed."
        return out, False

    def t_find_supplier(self, tax_id: str | None = None, name: str | None = None):
        inv = self.invoice()
        s, cands = self.erp.find_supplier(name or inv.supplier_name, tax_id or inv.supplier_tax_id)
        if s is None:
            return {"found": False, "close_names": cands,
                    "next_step": "Call hold_for_review with reason unknown_supplier."}, False
        return {"found": True, "supplier_id": s.supplier_id, "name": s.name, "tax_id": s.tax_id,
                "currency": s.currency, "payment_terms_days": s.terms_days, "email_on_file": s.email,
                "phone_on_file": s.phone, "bank_account_on_file": _mask(s.bank_account)}, False

    def t_verify_bank_account(self, supplier_id: str):
        s = self.erp.supplier(supplier_id)
        if s is None:
            return {"error": f"no supplier {supplier_id}"}, False
        inv = self.invoice()
        ok = bool(inv.bank_account) and norm_bank(inv.bank_account) == norm_bank(s.bank_account)
        out = {"match": ok, "on_invoice": _mask(inv.bank_account), "on_file": _mask(s.bank_account)}
        if not ok:
            out["next_step"] = ("Do not approve. Call draft_email_to_supplier asking the supplier to confirm its bank "
                                "details by phone on the number on file, then hold_for_review with bank_mismatch.")
        return out, False

    def t_get_purchase_order(self, po_number: str):
        po = self.erp.purchase_order(po_number)
        if po is None:
            return {"found": False, "po_number": norm_po(po_number) or po_number,
                    "similar_numbers": self.erp.similar_po_numbers(po_number),
                    "next_step": "Call search_purchase_orders for the supplier and three_way_match on each open PO; "
                                 "approve only if exactly one is clean, otherwise hold with po_not_found."}, False
        return {"found": True, **self._po_view(po)}, False

    def t_search_purchase_orders(self, supplier_id: str):
        if self.erp.supplier(supplier_id) is None:
            return {"error": f"no supplier {supplier_id}"}, False
        return {"open_purchase_orders": [self._po_view(p) for p in self.erp.open_pos(supplier_id)]}, False

    def t_get_goods_receipts(self, po_number: str):
        po = self.erp.purchase_order(po_number)
        if po is None:
            return {"found": False}, False
        rec, done = self.erp.received(po.po_number), self.erp.invoiced(po.po_number)
        return {"po_number": po.po_number, "lines": [
            {"line": li.line, "description": li.description, "ordered": li.quantity,
             "received": rec.get(li.line, 0.0), "already_invoiced": done.get(li.line, 0.0)} for li in po.lines]}, False

    def t_three_way_match(self, po_number: str):
        po = self.erp.purchase_order(po_number)
        if po is None:
            return {"found": False, "po_number": po_number}, False
        out = three_way_match(self.erp, self.invoice(), po, self.supplier()).as_dict()
        if not out["clean"]:
            out["next_step"] = ("Do not approve against this PO. If you are checking candidates from "
                                "search_purchase_orders, try the next one; otherwise call hold_for_review with these "
                                "issue codes.")
        return out, False

    def t_draft_email_to_supplier(self, subject: str, body: str):
        s = self.supplier()
        if s is None:
            return {"error": "supplier not on file; no trusted address to write to"}, False
        rec = self.erp.draft_email(self.case.doc_id, s.email, subject, body)
        return {"saved": True, "to": rec["to"], "status": rec["status"]}, False

    # ---------- actions that end the case ----------

    def t_approve_for_payment(self, po_number: str, note: str = ""):
        inv = self.invoice()
        blockers = approval_blockers(self.erp, inv, po_number)
        if blockers:
            self.case.guard_blocks += 1
            return {"approved": False, "blocked_by_policy": [b.as_dict() for b in blockers],
                    "instruction": "Not approved. Call hold_for_review with these reason codes."}, False
        s = self.supplier()
        po = self.erp.purchase_order(po_number)
        rec = self.erp.post(self.case.doc_id, s, inv, po.po_number, note,
                            three_way_match(self.erp, inv, po, s).invoiced)
        self.case.decision, self.case.po_number, self.case.note = "APPROVE", po.po_number, note
        return {"approved": True, "scheduled_payment_date": rec["scheduled_payment_date"]}, True

    def t_hold_for_review(self, reasons: list[str] | str, note: str = ""):
        reasons = [reasons] if isinstance(reasons, str) else list(reasons or [])
        bad = [r for r in reasons if r not in REASONS]
        if not reasons or bad:
            self.case.tool_errors += 1
            return {"error": f"unknown reason codes {bad}" if bad else "give at least one reason code",
                    "allowed": sorted(REASONS)}, False
        self.erp.hold(self.case.doc_id, reasons, note, self.case.invoice)
        self.case.decision, self.case.reasons, self.case.note = "HOLD", reasons, note
        return {"on_hold": True, "reasons": reasons}, True

    def t_reject_invoice(self, note: str = ""):
        s = self.supplier()
        dup = s and self.erp.is_duplicate(s.supplier_id, self.invoice().invoice_number)
        if not dup:
            self.case.guard_blocks += 1
            return {"rejected": False, "instruction": "Only a confirmed duplicate can be rejected. "
                                                      "Use hold_for_review instead."}, False
        self.erp.reject(self.case.doc_id, "duplicate", note, self.case.invoice)
        self.case.decision, self.case.reasons, self.case.note = "REJECT", ["duplicate"], note
        return {"rejected": True}, True


def dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)
