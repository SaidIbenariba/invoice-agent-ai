"""3-way match and payment policy, in plain code.

The agent never decides on money by itself. `approval_blockers` is the single source of truth for
"can this invoice be paid against this PO?", used by the `approve_for_payment` tool (the guard) and by
`policy_decision`, which the generator uses to check that every scenario has the intended answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from invoice_ai.validate import validate
from rapidfuzz import fuzz

from .erp import ERP
from .models import AgentInvoice, PurchaseOrder, Supplier, norm_bank, norm_po

PRICE_TOL = 0.02  # unit price may exceed the PO price by at most 2%
DESC_MIN = 80  # description similarity needed to pair an invoice line with a PO line


@dataclass
class Issue:
    code: str
    message: str

    def as_dict(self) -> dict:
        return {"code": self.code, "message": self.message}


@dataclass
class MatchResult:
    po_number: str
    issues: list[Issue] = field(default_factory=list)
    pairs: list[dict] = field(default_factory=list)  # one row per invoice line

    @property
    def clean(self) -> bool:
        return not self.issues

    @property
    def invoiced(self) -> dict[int, float]:
        out: dict[int, float] = {}
        for p in self.pairs:
            if p["po_line"] is not None and p["billed_qty"] is not None:
                out[p["po_line"]] = out.get(p["po_line"], 0.0) + p["billed_qty"]
        return out

    def as_dict(self) -> dict:
        return {"po_number": self.po_number, "clean": self.clean,
                "issues": [i.as_dict() for i in self.issues], "lines": self.pairs}


def _pair(desc: str, po: PurchaseOrder, used: set[int]):
    best, score = None, 0.0
    for pl in po.lines:
        if pl.line in used:
            continue
        s = fuzz.token_set_ratio(desc.lower(), pl.description.lower())
        if s > score:
            best, score = pl, s
    return (best, score) if score >= DESC_MIN else (None, score)


def three_way_match(erp: ERP, inv: AgentInvoice, po: PurchaseOrder, supplier: Supplier | None = None) -> MatchResult:
    """Invoice vs PO (what was ordered, at what price) vs receipts (what arrived), net of what was
    already invoiced against the same PO."""
    res = MatchResult(po.po_number)
    if supplier and po.supplier_id != supplier.supplier_id:
        owner = erp.supplier(po.supplier_id)
        res.issues.append(Issue("po_mismatch", f"{po.po_number} belongs to {owner.name if owner else po.supplier_id}, "
                                               f"not {supplier.name}"))
    if po.status != "open":
        res.issues.append(Issue("po_mismatch", f"{po.po_number} is {po.status}"))
    if inv.currency and inv.currency != po.currency:
        res.issues.append(Issue("po_mismatch", f"invoice currency {inv.currency}, PO currency {po.currency}"))

    received = erp.received(po.po_number)
    already = erp.invoiced(po.po_number)
    if not received:
        res.issues.append(Issue("not_received", f"no goods receipt recorded for {po.po_number}"))

    used: set[int] = set()
    for i, li in enumerate(inv.line_items, 1):
        pl, score = _pair(li.description or "", po, used)
        row = {"invoice_line": i, "description": li.description, "po_line": None, "po_description": None,
               "billed_qty": li.quantity, "ordered_qty": None, "received_qty": None, "already_invoiced": None,
               "invoice_price": li.unit_price, "po_price": None}
        if pl is None:
            res.issues.append(Issue("unmatched_line", f"line {i} '{li.description}' matches no line on {po.po_number}"))
            res.pairs.append(row)
            continue
        used.add(pl.line)
        rec, done = received.get(pl.line, 0.0), already.get(pl.line, 0.0)
        row.update(po_line=pl.line, po_description=pl.description, ordered_qty=pl.quantity,
                   received_qty=rec, already_invoiced=done, po_price=pl.unit_price)
        res.pairs.append(row)
        if li.quantity is not None and received and li.quantity > rec - done + 1e-9:
            if rec == 0:
                res.issues.append(Issue("not_received", f"line {i}: billed {li.quantity:g}, nothing received yet"))
            else:
                left = rec - done
                res.issues.append(Issue("qty_variance", f"line {i}: billed {li.quantity:g}, received {rec:g}"
                                        + (f", {done:g} already invoiced" if done else "")
                                        + f" -> at most {max(left, 0):g} can be paid"))
        if li.unit_price is not None and li.unit_price > pl.unit_price * (1 + PRICE_TOL) + 1e-9:
            pct = (li.unit_price / pl.unit_price - 1) * 100
            res.issues.append(Issue("price_variance", f"line {i}: unit price {li.unit_price:.2f} vs PO "
                                                      f"{pl.unit_price:.2f} (+{pct:.1f}%)"))
    return res


def invoice_checks(erp: ERP, inv: AgentInvoice, supplier: Supplier | None) -> list[Issue]:
    """Private Invoice AI's checks (required fields, line maths, tax, totals, dates, tax ID) + duplicates."""
    issues = [Issue("invoice_check_failed", i.message) for i in validate(inv).issues]
    if supplier:
        dup = erp.is_duplicate(supplier.supplier_id, inv.invoice_number)
        if dup:
            issues.append(Issue("duplicate", f"invoice {inv.invoice_number} from {supplier.name} already "
                                             f"{dup['where']} (dated {dup['invoice_date']}, total {dup['total']})"))
    return issues


def identify_supplier(erp: ERP, inv: AgentInvoice) -> Supplier | None:
    s, _ = erp.find_supplier(inv.supplier_name, inv.supplier_tax_id)
    return s


def bank_ok(inv: AgentInvoice, s: Supplier) -> bool:
    return bool(inv.bank_account) and norm_bank(inv.bank_account) == norm_bank(s.bank_account)


def clean_candidates(erp: ERP, inv: AgentInvoice, s: Supplier) -> list[str]:
    return [p.po_number for p in erp.open_pos(s.supplier_id) if three_way_match(erp, inv, p, s).clean]


def approval_blockers(erp: ERP, inv: AgentInvoice, po_number: str | None) -> list[Issue]:
    """Everything that stops this invoice from being paid against `po_number`. Empty list = payable."""
    s = identify_supplier(erp, inv)
    if s is None:
        return [Issue("unknown_supplier", f"'{inv.supplier_name}' (tax ID {inv.supplier_tax_id}) is not on file")]
    out: list[Issue] = []
    if not bank_ok(inv, s):
        out.append(Issue("bank_mismatch", "bank account on the invoice differs from the one on file"))
    out += invoice_checks(erp, inv, s)

    printed = norm_po(inv.po_number)
    printed_po = erp.purchase_order(printed)
    if printed_po and printed_po.po_number != norm_po(po_number):
        out.append(Issue("po_mismatch", f"invoice cites {printed}, approval requested against {po_number}"))
    po = erp.purchase_order(po_number)
    if po is None:
        out.append(Issue("po_not_found", f"no purchase order {po_number}"))
        return out
    out += three_way_match(erp, inv, po, s).issues
    if printed_po is None:  # no usable PO on the invoice: only safe if exactly one open PO fits
        cands = clean_candidates(erp, inv, s)
        if len(cands) > 1:
            out.append(Issue("po_ambiguous", f"invoice gives no valid PO and {len(cands)} open POs match: "
                                             f"{', '.join(cands)}"))
    return out


def policy_decision(erp: ERP, inv: AgentInvoice) -> tuple[str, list[str], str | None]:
    """What a careful clerk following the policy would do. Used to validate generated scenarios."""
    s = identify_supplier(erp, inv)
    if s is None:
        return "HOLD", ["unknown_supplier"], None
    checks = invoice_checks(erp, inv, s)
    if any(i.code == "duplicate" for i in checks):
        return "REJECT", ["duplicate"], None
    printed = norm_po(inv.po_number)
    printed_po = erp.purchase_order(printed)
    if printed_po is not None:
        po_number = printed_po.po_number
    else:
        cands = clean_candidates(erp, inv, s)
        if len(cands) > 1:
            return "HOLD", sorted({"po_ambiguous"} | {i.code for i in checks}), None
        if not cands:
            reasons = {i.code for i in checks}
            if not bank_ok(inv, s):
                reasons.add("bank_mismatch")
            return "HOLD", sorted(reasons or {"po_not_found"}), None
        po_number = cands[0]
    blockers = approval_blockers(erp, inv, po_number)
    if blockers:
        return "HOLD", sorted({b.code for b in blockers}), po_number
    return "APPROVE", [], po_number
