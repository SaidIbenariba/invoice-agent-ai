"""Synthetic accounts-payable world with ground truth.

Writes, for one batch:
  <out>/erp/suppliers.json, purchase_orders.json, receipts.json, paid_invoices.json   (master data)
  <out>/inbox/inv_001.pdf|jpg ...                                                      (supplier invoices)
  <out>/truth/inv_001.json ...                                                         (expected decision)

Invoices reuse the four layouts of Private Invoice AI (US, French, UK, Moroccan) with two extra printed
fields: the customer's PO reference and the supplier's bank account. About 30% are degraded into phone
photos. Every scenario is checked against the payment policy (`match.policy_decision`) before writing,
so the expected answer is what the rules give on a perfectly read invoice.
"""

from __future__ import annotations

import io
import json
import random
import shutil
from datetime import date, timedelta
from pathlib import Path

from invoice_ai import synth as base
from invoice_ai.schema import LineItem
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen.canvas import Canvas

from .erp import ERP
from .match import policy_decision, three_way_match
from .models import (AgentInvoice, GoodsReceipt, PaidInvoice, POLine, PurchaseOrder, ReceiptLine, Supplier,
                     norm_po)

W, H = A4

PLAN = (["clean"] * 7 + ["po_missing_unique"] * 2 + ["po_typo"] + ["price_over_po"] * 2
        + ["qty_over_received"] * 2 + ["not_received"] + ["bank_change"] * 2 + ["duplicate"] * 2
        + ["invoice_error"] * 2 + ["unknown_supplier"] + ["po_missing_ambiguous"] + ["wrong_po"])

EXPECTED = {
    "clean": ("APPROVE", []),
    "po_missing_unique": ("APPROVE", []),
    "po_typo": ("APPROVE", []),
    "price_over_po": ("HOLD", ["price_variance"]),
    "qty_over_received": ("HOLD", ["qty_variance"]),
    "not_received": ("HOLD", ["not_received"]),
    "bank_change": ("HOLD", ["bank_mismatch"]),
    "duplicate": ("REJECT", ["duplicate"]),
    "invoice_error": ("HOLD", ["invoice_check_failed"]),
    "unknown_supplier": ("HOLD", ["unknown_supplier"]),
    "po_missing_ambiguous": ("HOLD", ["po_ambiguous"]),
    "wrong_po": ("HOLD", ["po_mismatch"]),
}

INV_PREFIX = {"us": "INV-", "fr": "F2026-", "uk": "TV", "ma": "2026/"}
PHONE = {"us": "+1 214 555 01{:02d}", "uk": "+44 118 496 0{:03d}", "fr": "+33 1 99 00 {:02d} 10",
         "ma": "+212 537 00 {:02d} 10"}


def _digits(rng: random.Random, n: int) -> str:
    return "".join(str(rng.randint(0, 9)) for _ in range(n))


def bank_account(loc: str, rng: random.Random) -> str:
    if loc == "us":
        return f"Routing {_digits(rng, 9)}  Account {_digits(rng, 10)}"
    if loc == "uk":
        sc = _digits(rng, 6)
        return f"Sort code {sc[:2]}-{sc[2:4]}-{sc[4:]}  Account {_digits(rng, 8)}"
    if loc == "fr":
        raw = "FR76" + _digits(rng, 23)
        return " ".join(raw[i:i + 4] for i in range(0, len(raw), 4))
    return f"{_digits(rng, 3)} {_digits(rng, 3)} {_digits(rng, 16)} {_digits(rng, 2)}"


BANK_LABEL = {"us": "Remit to: {}", "uk": "Bank: {}", "fr": "IBAN : {}", "ma": "RIB : {}"}
PO_LABEL = {"us": "PO Number:", "uk": "Your PO: {}", "fr": "Réf. commande : {}", "ma": "Bon de commande N° : {}"}


# ---------- rendering: Private Invoice AI layouts + PO reference + bank account ----------

class _WithoutFixedBankLine:
    """The UK layout prints the same bank line on every invoice. Drop it at the source (covering it would
    leave the old text in the PDF text layer), so each supplier's own account can be printed instead."""

    def __init__(self, canvas: Canvas):
        self._c = canvas

    def __getattr__(self, name):
        return getattr(self._c, name)

    def drawString(self, x, y, text, *args, **kwargs):
        if not text.startswith("Bank: Lloyds"):
            self._c.drawString(x, y, text, *args, **kwargs)


def render_pdf(inv: AgentInvoice, loc_key: str) -> bytes:
    loc = base.LOCALES[loc_key]
    buf = io.BytesIO()
    c = Canvas(buf, pagesize=A4)
    base.DRAW[loc_key](_WithoutFixedBankLine(c) if loc_key == "uk" else c, inv, loc)
    c.setFillColor(HexColor("#222222"))
    c.setFont("Helvetica", 10)
    if inv.po_number:
        if loc_key == "us":
            c.drawRightString(W - 140, H - 150, PO_LABEL["us"]); c.drawRightString(W - 50, H - 150, inv.po_number)
        elif loc_key == "fr":
            c.drawString(60, H - 195, PO_LABEL["fr"].format(inv.po_number))
        elif loc_key == "uk":
            c.drawString(50, H - 190, PO_LABEL["uk"].format(inv.po_number))
        else:
            c.drawString(50, H - 178, PO_LABEL["ma"].format(inv.po_number))
    if inv.bank_account:
        text = BANK_LABEL[loc_key].format(inv.bank_account)
        c.setFont("Helvetica", 9)
        if loc_key == "uk":
            c.drawString(50, 60, f"{text}. Please quote the invoice number.")
        else:
            c.drawString(50, {"us": 76, "fr": 86, "ma": 96}[loc_key], text)
    c.showPage(); c.save()
    return buf.getvalue()


# ---------- the world ----------

class World:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.suppliers: list[Supplier] = []
        self.loc_of: dict[str, str] = {}
        self.pos: list[PurchaseOrder] = []
        self.receipts: list[GoodsReceipt] = []
        self.paid: list[PaidInvoice] = []
        self.po_ids = iter(rng.sample(range(1000, 9999), 400))
        self.inv_ids = iter(rng.sample(range(1000, 9999), 400))
        n = 0
        for lk, loc in base.LOCALES.items():
            for name, _ in loc.suppliers:
                slug = "".join(ch for ch in name.lower() if ch.isalnum())[:18]
                s = Supplier(supplier_id=f"SUP-{101 + n}", name=name, tax_id=base.tax_id(lk, rng),
                             bank_account=bank_account(lk, rng), currency=loc.currency,
                             terms_days=rng.choice([30, 45]), email=f"accounts@{slug}.example",
                             phone=PHONE[lk].format(n + 10))
                self.suppliers.append(s)
                self.loc_of[s.supplier_id] = lk
                n += 1

    def po(self, s: Supplier, items=None, status="open", when: date | None = None) -> PurchaseOrder:
        rng, loc = self.rng, base.LOCALES[self.loc_of[s.supplier_id]]
        items = items or rng.sample(loc.items, rng.randint(2, min(4, len(loc.items))))
        lines = [POLine(line=i, description=d, quantity=float(rng.choice([2, 4, 5, 6, 8, 10, 12, 20, 25, 50])),
                        unit_price=round(rng.uniform(lo, hi), 2)) for i, (d, lo, hi) in enumerate(items, 1)]
        when = when or date(2026, 8, 20) + timedelta(days=rng.randint(0, 25))
        p = PurchaseOrder(po_number=f"PO-2026-{next(self.po_ids)}", supplier_id=s.supplier_id,
                          order_date=when.isoformat(), currency=s.currency, status=status, lines=lines)
        self.pos.append(p)
        return p

    def receive(self, p: PurchaseOrder, partial: dict[int, float] | None = None) -> GoodsReceipt:
        when = date.fromisoformat(p.order_date) + timedelta(days=self.rng.randint(4, 12))
        lines = [ReceiptLine(line=li.line, quantity_received=(partial or {}).get(li.line, li.quantity))
                 for li in p.lines]
        r = GoodsReceipt(receipt_id=f"GR-{p.po_number[-4:]}-{len(self.receipts) + 1}", po_number=p.po_number,
                         received_date=when.isoformat(), lines=[li for li in lines if li.quantity_received > 0])
        self.receipts.append(r)
        return r

    def invoice(self, s: Supplier, p: PurchaseOrder, qty: dict[int, float], print_po: bool = True) -> AgentInvoice:
        rng, lk = self.rng, self.loc_of[s.supplier_id]
        loc = base.LOCALES[lk]
        items = [LineItem(description=li.description, quantity=qty[li.line], unit_price=li.unit_price,
                          amount=round(qty[li.line] * li.unit_price, 2)) for li in p.lines if qty.get(li.line, 0) > 0]
        issued = date.fromisoformat(p.order_date) + timedelta(days=rng.randint(14, 24))
        inv = AgentInvoice(supplier_name=s.name, supplier_tax_id=s.tax_id,
                           invoice_number=f"{INV_PREFIX[lk]}{next(self.inv_ids)}",
                           invoice_date=issued.isoformat(),
                           due_date=(issued + timedelta(days=s.terms_days)).isoformat(),
                           currency=s.currency, tax_rate=rng.choice(loc.rates), line_items=items,
                           po_number=p.po_number if print_po else None, bank_account=s.bank_account)
        return retotal(inv)


def retotal(inv: AgentInvoice) -> AgentInvoice:
    inv.subtotal = round(sum(li.amount for li in inv.line_items), 2)
    inv.tax_amount = round(inv.subtotal * inv.tax_rate / 100, 2)
    inv.total = round(inv.subtotal + inv.tax_amount, 2)
    return inv


def other_items(w: World, s: Supplier, p: PurchaseOrder):
    """Items of the supplier's catalogue not on `p`, so a distractor PO cannot match p's invoice."""
    loc = base.LOCALES[w.loc_of[s.supplier_id]]
    used = {li.description for li in p.lines}
    rest = [it for it in loc.items if it[0] not in used]
    if not rest:
        rest = [it for it in loc.items if it[0] != p.lines[0].description]
    return w.rng.sample(rest, min(len(rest), w.rng.randint(1, 2)))


def build_case(w: World, scenario: str, s: Supplier, unknown: Supplier) -> AgentInvoice:
    rng = w.rng
    if scenario == "unknown_supplier":
        s = unknown
        p = PurchaseOrder(po_number=f"PO-2026-{next(w.po_ids)}", supplier_id=s.supplier_id, order_date="2026-09-02",
                          currency=s.currency, lines=w.po(s).lines)
        w.pos.pop()  # the PO only exists on the supplier's side, not in our ERP
        return w.invoice(s, p, {li.line: li.quantity for li in p.lines})

    if scenario == "po_missing_ambiguous":
        p1 = w.po(s)
        p2 = w.po(s, items=[(li.description, li.unit_price, li.unit_price) for li in p1.lines])
        for a, b in zip(p1.lines, p2.lines):
            b.quantity, b.unit_price = a.quantity, a.unit_price
        w.receive(p1); w.receive(p2)
        return w.invoice(s, p1, {li.line: li.quantity for li in p1.lines}, print_po=False)

    if scenario == "wrong_po":
        others = [x for x in w.suppliers if x.supplier_id not in (s.supplier_id, unknown.supplier_id)]
        foreign = w.po(rng.choice(others))
        w.receive(foreign)
        mine = w.po(s)
        w.pos.remove(mine)  # the supplier's own order was never raised in our ERP
        inv = w.invoice(s, mine, {li.line: li.quantity for li in mine.lines})
        inv.po_number = foreign.po_number
        return inv

    p = w.po(s)
    full = {li.line: li.quantity for li in p.lines}

    if scenario == "clean" and rng.random() < 0.3:  # partial delivery, billed exactly as received
        li = rng.choice(p.lines)
        part = dict(full); part[li.line] = max(1.0, float(int(li.quantity * rng.choice([0.5, 0.6, 0.8]))))
        w.receive(p, part)
        return w.invoice(s, p, part)

    if scenario == "qty_over_received":
        li = rng.choice(p.lines)
        part = dict(full); part[li.line] = max(1.0, float(int(li.quantity * rng.choice([0.5, 0.6, 0.75]))))
        w.receive(p, part)
        return w.invoice(s, p, full)

    if scenario == "not_received":
        return w.invoice(s, p, full)

    w.receive(p)
    inv = w.invoice(s, p, full, print_po=scenario not in ("po_missing_unique",))

    if scenario in ("po_missing_unique", "po_typo"):
        w.po(s, items=other_items(w, s, p))  # a distractor open PO from the same supplier
        if scenario == "po_typo":
            digits = list(p.po_number[-4:])
            i = rng.randrange(3)
            digits[i], digits[i + 1] = digits[i + 1], digits[i]
            typo = p.po_number[:-4] + "".join(digits)
            if typo == p.po_number or any(x.po_number == typo for x in w.pos):
                typo = p.po_number[:-4] + str((int(p.po_number[-4:]) + 7) % 10000).zfill(4)
            inv.po_number = typo
    elif scenario == "price_over_po":
        li = rng.choice(inv.line_items)
        li.unit_price = round(li.unit_price * (1 + rng.uniform(0.06, 0.15)), 2)
        li.amount = round(li.quantity * li.unit_price, 2)
        retotal(inv)
    elif scenario == "bank_change":
        inv.bank_account = bank_account(w.loc_of[s.supplier_id], rng)
    elif scenario == "invoice_error":
        kind = rng.choice(["total_mismatch", "tax_mismatch", "line_amount_mismatch"])
        if kind == "total_mismatch":
            inv.total = round(inv.total + rng.choice([10, 20, 50, 90]), 2)
        elif kind == "tax_mismatch":
            inv.tax_amount = round(inv.subtotal * max(inv.tax_rate - 2, 1) / 100, 2)
            inv.total = round(inv.subtotal + inv.tax_amount, 2)
        else:
            li = rng.choice(inv.line_items)
            li.amount = round(li.amount + rng.choice([10, 25, 40]), 2)
            inv.subtotal = round(sum(x.amount for x in inv.line_items), 2)
            inv.tax_amount = round(inv.subtotal * inv.tax_rate / 100, 2)
            inv.total = round(inv.subtotal + inv.tax_amount, 2)
    elif scenario == "duplicate":  # this exact invoice was paid last month; the PO is now closed
        p.status = "closed"
        w.paid.append(PaidInvoice(supplier_id=s.supplier_id, invoice_number=inv.invoice_number,
                                  invoice_date=inv.invoice_date, total=inv.total, currency=inv.currency,
                                  po_number=p.po_number, invoiced=full))
    return inv


class Inconsistent(Exception):
    pass


def build_world(rng: random.Random) -> tuple[World, list[tuple[str, str, AgentInvoice]]]:
    w = World(rng)
    unknown = rng.choice(w.suppliers)
    known = [s for s in w.suppliers if s.supplier_id != unknown.supplier_id]
    for s in known:  # each supplier has one unrelated open order, as in any real ERP
        p = w.po(s)
        if rng.random() < 0.6:
            w.receive(p)
    for s in rng.sample(known, 5):  # and some paid history
        w.paid.append(PaidInvoice(supplier_id=s.supplier_id, invoice_number=f"{INV_PREFIX[w.loc_of[s.supplier_id]]}"
                                  f"{next(w.inv_ids)}", invoice_date="2026-08-14",
                                  total=round(rng.uniform(300, 4000), 2), currency=s.currency))
    plan = list(PLAN)
    rng.shuffle(plan)
    cases = [(f"inv_{i:03d}", sc, build_case(w, sc, rng.choice(known), unknown)) for i, sc in enumerate(plan, 1)]
    w.suppliers.remove(unknown)
    return w, cases


def check_world(w: World, cases) -> list[tuple]:
    """Process the inbox in order with the policy on perfectly read invoices; every case must give
    its intended answer, or the world is rebuilt."""
    erp = ERP(suppliers=w.suppliers, purchase_orders=w.pos, receipts=w.receipts, paid_invoices=w.paid)
    out = []
    for doc_id, scenario, inv in cases:
        want_decision, want_reasons = EXPECTED[scenario]
        got_decision, got_reasons, po_used = policy_decision(erp, inv)
        if got_decision != want_decision or not set(want_reasons) <= set(got_reasons):
            raise Inconsistent(f"{doc_id} {scenario}: policy gives {got_decision} {got_reasons}")
        if got_decision == "APPROVE":
            s = erp.find_supplier(inv.supplier_name, inv.supplier_tax_id)[0]
            erp.post(doc_id, s, inv, po_used, "", three_way_match(erp, inv, erp.purchase_order(po_used), s).invoiced)
        out.append((doc_id, scenario, inv, got_reasons, po_used))
    return out


def generate(out_dir: Path, seed: int = 11, photo_share: float = 0.3, max_attempts: int = 50) -> list[dict]:
    for attempt in range(max_attempts):
        w, cases = build_world(random.Random(seed * 1000 + attempt))
        try:
            checked = check_world(w, cases)
            break
        except Inconsistent:
            continue
    else:
        raise RuntimeError(f"no consistent world for seed {seed} in {max_attempts} attempts")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    erp_dir = out_dir / "erp"
    erp_dir.mkdir(parents=True)
    for name, rows in [("suppliers", w.suppliers), ("purchase_orders", w.pos),
                       ("receipts", w.receipts), ("paid_invoices", w.paid)]:
        (erp_dir / f"{name}.json").write_text(json.dumps([r.model_dump() for r in rows], indent=2, ensure_ascii=False))

    rng = random.Random(seed)
    inbox, truth_dir = out_dir / "inbox", out_dir / "truth"
    inbox.mkdir(); truth_dir.mkdir()
    truths = []
    for doc_id, scenario, inv, reasons, po_used in checked:
        lk = next(k for k, loc in base.LOCALES.items() if any(n == inv.supplier_name for n, _ in loc.suppliers))
        pdf = render_pdf(inv, lk)
        photo = rng.random() < photo_share
        fname = f"{doc_id}.jpg" if photo else f"{doc_id}.pdf"
        if photo:
            base.degrade_to_photo(pdf, rng).save(inbox / fname, quality=62)
        else:
            (inbox / fname).write_bytes(pdf)
        t = {"doc_id": doc_id, "file": fname, "layout": lk, "photo": photo, "scenario": scenario,
             "expected": {"decision": EXPECTED[scenario][0], "reasons": EXPECTED[scenario][1],
                          "all_policy_reasons": reasons, "po_number": po_used},
             "invoice": inv.model_dump()}
        (truth_dir / f"{doc_id}.json").write_text(json.dumps(t, indent=2, ensure_ascii=False))
        truths.append(t)
    return truths
