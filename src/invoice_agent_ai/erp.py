"""A tiny ERP: master data read from JSON, plus the state the agent writes (postings, review queue, drafts).

Master data (suppliers, purchase orders, goods receipts, paid invoices) lives in `<data>/erp/`.
Everything the agent writes goes to a separate workspace folder, so a batch can be re-run from scratch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

from rapidfuzz import fuzz

from .models import GoodsReceipt, PaidInvoice, PurchaseOrder, Supplier, norm_code, norm_po


def _load(path: Path, model):
    return [model.model_validate(x) for x in json.loads(path.read_text())] if path.exists() else []


@dataclass
class ERP:
    suppliers: list[Supplier]
    purchase_orders: list[PurchaseOrder]
    receipts: list[GoodsReceipt]
    paid_invoices: list[PaidInvoice]
    workspace: Path | None = None
    postings: list[dict] = field(default_factory=list)
    review_queue: list[dict] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)
    drafts: list[dict] = field(default_factory=list)

    @classmethod
    def load(cls, erp_dir: Path, workspace: Path | None = None, fresh: bool = True) -> "ERP":
        erp = cls(
            suppliers=_load(erp_dir / "suppliers.json", Supplier),
            purchase_orders=_load(erp_dir / "purchase_orders.json", PurchaseOrder),
            receipts=_load(erp_dir / "receipts.json", GoodsReceipt),
            paid_invoices=_load(erp_dir / "paid_invoices.json", PaidInvoice),
            workspace=workspace,
        )
        if workspace:
            workspace.mkdir(parents=True, exist_ok=True)
            if not fresh:
                for name in ("postings", "review_queue", "rejected", "drafts"):
                    p = workspace / f"{name}.json"
                    if p.exists():
                        setattr(erp, name, json.loads(p.read_text()))
        return erp

    def save(self) -> None:
        if not self.workspace:
            return
        for name in ("postings", "review_queue", "rejected", "drafts"):
            (self.workspace / f"{name}.json").write_text(json.dumps(getattr(self, name), indent=2, ensure_ascii=False))

    # ---------- lookups ----------

    def supplier(self, supplier_id: str) -> Supplier | None:
        return next((s for s in self.suppliers if s.supplier_id == supplier_id), None)

    def find_supplier(self, name: str | None = None, tax_id: str | None = None) -> tuple[Supplier | None, list[dict]]:
        """Tax ID first (exact after normalisation), then name (fuzzy). Returns (match, close candidates)."""
        if tax_id:
            t = norm_code(tax_id)
            hit = next((s for s in self.suppliers if norm_code(s.tax_id) == t), None)
            if hit:
                return hit, []
        if name:
            scored = sorted(((fuzz.token_set_ratio(name.lower(), s.name.lower()), s) for s in self.suppliers),
                            key=lambda x: -x[0])
            if scored and scored[0][0] >= 92:
                return scored[0][1], []
            return None, [{"supplier_id": s.supplier_id, "name": s.name, "similarity": round(sc)}
                          for sc, s in scored[:3] if sc >= 60]
        return None, []

    def purchase_order(self, po_number: str | None) -> PurchaseOrder | None:
        key = norm_po(po_number)
        return next((p for p in self.purchase_orders if p.po_number == key), None) if key else None

    def similar_po_numbers(self, po_number: str, limit: int = 3) -> list[str]:
        q = norm_code(po_number)
        scored = sorted(((fuzz.ratio(q, norm_code(p.po_number)), p.po_number) for p in self.purchase_orders),
                        reverse=True)
        return [n for sc, n in scored[:limit] if sc >= 85]

    def open_pos(self, supplier_id: str) -> list[PurchaseOrder]:
        return [p for p in self.purchase_orders if p.supplier_id == supplier_id and p.status == "open"]

    def received(self, po_number: str) -> dict[int, float]:
        """Quantity received per PO line, summed over all receipts."""
        out: dict[int, float] = {}
        for r in self.receipts:
            if r.po_number == po_number:
                for li in r.lines:
                    out[li.line] = out.get(li.line, 0.0) + li.quantity_received
        return out

    def invoiced(self, po_number: str) -> dict[int, float]:
        """Quantity already invoiced per PO line (paid history + approvals in this run)."""
        out: dict[int, float] = {}
        rows = [(p.po_number, p.invoiced) for p in self.paid_invoices]
        rows += [(p.get("po_number"), {int(k): v for k, v in p.get("invoiced", {}).items()}) for p in self.postings]
        for po, lines in rows:
            if po == po_number:
                for line, q in lines.items():
                    out[int(line)] = out.get(int(line), 0.0) + q
        return out

    def is_duplicate(self, supplier_id: str, invoice_number: str | None) -> dict | None:
        if not invoice_number:
            return None
        n = norm_code(invoice_number)
        for p in self.paid_invoices:
            if p.supplier_id == supplier_id and norm_code(p.invoice_number) == n:
                return {"where": "paid", "invoice_date": p.invoice_date, "total": p.total}
        for p in self.postings:
            if p["supplier_id"] == supplier_id and norm_code(p["invoice_number"]) == n:
                return {"where": "posted in this run", "invoice_date": p["invoice_date"], "total": p["total"]}
        return None

    # ---------- writes ----------

    def post(self, doc_id: str, supplier: Supplier, invoice, po_number: str, note: str,
             invoiced: dict[int, float] | None = None) -> dict:
        issued = date.fromisoformat(invoice.invoice_date) if invoice.invoice_date else date.today()
        pay_on = invoice.due_date or (issued + timedelta(days=supplier.terms_days)).isoformat()
        rec = {"doc_id": doc_id, "supplier_id": supplier.supplier_id, "supplier_name": supplier.name,
               "invoice_number": invoice.invoice_number, "invoice_date": invoice.invoice_date,
               "po_number": po_number, "currency": invoice.currency, "total": invoice.total,
               "scheduled_payment_date": pay_on, "note": note,
               "invoiced": {str(k): v for k, v in (invoiced or {}).items()}}
        self.postings.append(rec)
        self.save()
        return rec

    def hold(self, doc_id: str, reasons: list[str], note: str, invoice=None) -> dict:
        rec = {"doc_id": doc_id, "reasons": reasons, "note": note,
               "supplier_name": getattr(invoice, "supplier_name", None),
               "invoice_number": getattr(invoice, "invoice_number", None),
               "total": getattr(invoice, "total", None), "currency": getattr(invoice, "currency", None)}
        self.review_queue.append(rec)
        self.save()
        return rec

    def reject(self, doc_id: str, reason: str, note: str, invoice=None) -> dict:
        rec = {"doc_id": doc_id, "reason": reason, "note": note,
               "supplier_name": getattr(invoice, "supplier_name", None),
               "invoice_number": getattr(invoice, "invoice_number", None)}
        self.rejected.append(rec)
        self.save()
        return rec

    def draft_email(self, doc_id: str, to: str, subject: str, body: str) -> dict:
        rec = {"doc_id": doc_id, "to": to, "subject": subject, "body": body, "status": "draft (not sent)"}
        self.drafts.append(rec)
        self.save()
        return rec
