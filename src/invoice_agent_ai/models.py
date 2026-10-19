"""Data shared by the ERP, the generator, the tools and the evaluation."""

from __future__ import annotations

import re

from invoice_ai.schema import Invoice
from pydantic import BaseModel, Field


class AgentInvoice(Invoice):
    """Private Invoice AI's invoice, plus the two fields accounts payable needs for a 3-way match."""

    po_number: str | None = Field(None, description="Purchase order reference printed on the invoice")
    bank_account: str | None = Field(None, description="Supplier bank account printed on the invoice")


class Supplier(BaseModel):
    supplier_id: str
    name: str
    tax_id: str
    bank_account: str
    currency: str
    terms_days: int = 30
    email: str
    phone: str


class POLine(BaseModel):
    line: int
    description: str
    quantity: float
    unit_price: float


class PurchaseOrder(BaseModel):
    po_number: str
    supplier_id: str
    order_date: str
    currency: str
    status: str = "open"  # open | closed
    lines: list[POLine]

    @property
    def total(self) -> float:
        return round(sum(li.quantity * li.unit_price for li in self.lines), 2)


class ReceiptLine(BaseModel):
    line: int
    quantity_received: float


class GoodsReceipt(BaseModel):
    receipt_id: str
    po_number: str
    received_date: str
    lines: list[ReceiptLine]


class PaidInvoice(BaseModel):
    supplier_id: str
    invoice_number: str
    invoice_date: str
    total: float
    currency: str
    po_number: str | None = None
    invoiced: dict[int, float] = Field(default_factory=dict, description="PO line -> quantity invoiced")


# Reason codes the agent can give when it holds or rejects an invoice.
REASONS = {
    "invoice_check_failed": "the invoice fails an arithmetic, tax, date or required-field check",
    "duplicate": "this invoice was already paid or posted",
    "unknown_supplier": "the supplier is not in the supplier master",
    "bank_mismatch": "the bank account on the invoice differs from the one on file",
    "po_not_found": "no purchase order could be found for this invoice",
    "po_ambiguous": "more than one open purchase order matches and the invoice gives no PO number",
    "po_mismatch": "the PO belongs to another supplier, is closed, or uses another currency",
    "price_variance": "a unit price is above the PO price beyond tolerance",
    "qty_variance": "a billed quantity is more than what was received",
    "not_received": "the goods on the invoice have not been received",
    "unmatched_line": "an invoice line does not match any PO line",
    "agent_error": "the agent could not finish; a person must process this invoice",
}


def norm_code(s: str | None) -> str:
    """Upper-case alphanumerics only: compares tax IDs, bank accounts and invoice numbers."""
    return re.sub(r"[^0-9A-Z]", "", (s or "").upper())


def norm_bank(s: str | None) -> str:
    """Keep only the tokens that carry digits: 'IBAN : FR76 3000 ...' -> 'FR763000...',
    'Sort code 30-90-12 Account 12345678' -> '30901212345678'. Labels like IBAN, RIB, Account drop out."""
    toks = re.findall(r"[0-9A-Z\-]*\d[0-9A-Z\-]*", (s or "").upper())
    return "".join(re.sub(r"[^0-9A-Z]", "", t) for t in toks)


PO_RE = re.compile(r"PO[\s\-_]*(\d{4})[\s\-_]*(\d{3,5})", re.I)


def norm_po(s: str | None) -> str | None:
    """'Réf. commande : po 2026-0412' -> 'PO-2026-0412'. None when nothing usable is printed."""
    if not s or not s.strip():
        return None
    m = PO_RE.search(s)
    if m:
        return f"PO-{m[1]}-{m[2]}"
    return s.strip().upper() or None
