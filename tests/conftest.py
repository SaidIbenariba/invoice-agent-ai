from __future__ import annotations

from pathlib import Path

import pytest

from invoice_agent_ai.synth import generate


def raw_from_truth(inv: dict) -> dict:
    """What a perfect reader would copy off the page, in the raw-string form the extractor returns."""
    money = lambda x: "" if x is None else f"{x:.2f}"
    return {
        "supplier_name": inv["supplier_name"] or "", "supplier_tax_id": inv["supplier_tax_id"] or "",
        "invoice_number": inv["invoice_number"] or "", "invoice_date": inv["invoice_date"] or "",
        "due_date": inv["due_date"] or "", "currency": inv["currency"] or "",
        "subtotal": money(inv["subtotal"]), "tax_rate": "" if inv["tax_rate"] is None else f"{inv['tax_rate']}%",
        "tax_amount": money(inv["tax_amount"]), "total": money(inv["total"]),
        "line_items": [{"description": li["description"], "quantity": f"{li['quantity']:g}",
                        "unit_price": money(li["unit_price"]), "amount": money(li["amount"])}
                       for li in inv["line_items"]],
        "po_number": inv.get("po_number") or "", "bank_account": inv.get("bank_account") or "",
    }


@pytest.fixture(scope="session")
def world(tmp_path_factory) -> tuple[Path, dict]:
    d = tmp_path_factory.mktemp("world") / "batch"
    truths = generate(d, seed=11, photo_share=0.0)
    return d, {t["doc_id"]: t for t in truths}


@pytest.fixture(scope="session")
def perfect_reader(world):
    _, truths = world

    def read(path: Path) -> dict:
        return {"file": path.name, "route": "text", "seconds": 0.0,
                "raw": raw_from_truth(truths[path.stem]["invoice"])}
    return read
