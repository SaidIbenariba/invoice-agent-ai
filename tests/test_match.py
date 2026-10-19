from invoice_ai.schema import LineItem

from invoice_agent_ai.erp import ERP
from invoice_agent_ai.match import approval_blockers, policy_decision, three_way_match
from invoice_agent_ai.models import (AgentInvoice, GoodsReceipt, PaidInvoice, POLine, PurchaseOrder, ReceiptLine,
                                     Supplier, norm_bank, norm_po)

SUP = Supplier(supplier_id="SUP-1", name="Keystone Packaging Co.", tax_id="12-3456789",
               bank_account="Routing 021000021  Account 4839201177", currency="USD", email="a@x.example", phone="1")
OTHER = Supplier(supplier_id="SUP-2", name="Brightline Logistics LLC", tax_id="98-7654321",
                 bank_account="Routing 111000025  Account 1234567890", currency="USD", email="b@x.example", phone="2")


def po(num="PO-2026-1001", supplier="SUP-1", status="open", qty=(10, 4), price=(2.5, 30.0)):
    return PurchaseOrder(po_number=num, supplier_id=supplier, order_date="2026-09-01", currency="USD", status=status,
                         lines=[POLine(line=1, description="Corrugated boxes 24x18x12", quantity=qty[0], unit_price=price[0]),
                                POLine(line=2, description="Stretch wrap roll 18in", quantity=qty[1], unit_price=price[1])])


def receipt(num="PO-2026-1001", q=(10, 4)):
    return GoodsReceipt(receipt_id=f"GR-{num}", po_number=num, received_date="2026-09-08",
                        lines=[ReceiptLine(line=1, quantity_received=q[0]), ReceiptLine(line=2, quantity_received=q[1])])


def invoice(qty=(10, 4), price=(2.5, 30.0), po_number="PO-2026-1001", bank=SUP.bank_account, number="INV-1"):
    items = [LineItem(description="Corrugated boxes 24x18x12", quantity=qty[0], unit_price=price[0],
                      amount=round(qty[0] * price[0], 2)),
             LineItem(description="Stretch wrap roll 18in", quantity=qty[1], unit_price=price[1],
                      amount=round(qty[1] * price[1], 2))]
    sub = round(sum(i.amount for i in items), 2)
    return AgentInvoice(supplier_name=SUP.name, supplier_tax_id=SUP.tax_id, invoice_number=number,
                        invoice_date="2026-09-20", due_date="2026-10-20", currency="USD", subtotal=sub, tax_rate=0.0,
                        tax_amount=0.0, total=sub, line_items=items, po_number=po_number, bank_account=bank)


def erp(pos=None, receipts=None, paid=None):
    return ERP(suppliers=[SUP, OTHER], purchase_orders=pos if pos is not None else [po()],
               receipts=receipts if receipts is not None else [receipt()], paid_invoices=paid or [])


def codes(issues):
    return sorted({i.code for i in issues})


def test_clean_match_and_approval():
    e = erp()
    assert three_way_match(e, invoice(), e.purchase_order("PO-2026-1001"), SUP).clean
    assert approval_blockers(e, invoice(), "PO-2026-1001") == []
    assert policy_decision(e, invoice()) == ("APPROVE", [], "PO-2026-1001")


def test_price_above_tolerance_is_flagged_but_within_tolerance_is_not():
    e = erp()
    assert codes(three_way_match(e, invoice(price=(2.55, 30.0)), e.purchase_order("PO-2026-1001"), SUP).issues) == []
    m = three_way_match(e, invoice(price=(2.80, 30.0)), e.purchase_order("PO-2026-1001"), SUP)
    assert codes(m.issues) == ["price_variance"] and "+12.0%" in m.issues[0].message


def test_billed_more_than_received_net_of_already_invoiced():
    e = erp(receipts=[receipt(q=(6, 4))])
    assert codes(three_way_match(e, invoice(), e.purchase_order("PO-2026-1001"), SUP).issues) == ["qty_variance"]
    assert three_way_match(e, invoice(qty=(6, 4)), e.purchase_order("PO-2026-1001"), SUP).clean
    paid = PaidInvoice(supplier_id="SUP-1", invoice_number="INV-0", invoice_date="2026-09-10", total=1, currency="USD",
                       po_number="PO-2026-1001", invoiced={1: 6})
    e2 = erp(receipts=[receipt(q=(6, 4))], paid=[paid])
    m = three_way_match(e2, invoice(qty=(6, 4)), e2.purchase_order("PO-2026-1001"), SUP)
    assert codes(m.issues) == ["qty_variance"] and "6 already invoiced" in m.issues[0].message


def test_nothing_received():
    e = erp(receipts=[])
    assert codes(three_way_match(e, invoice(), e.purchase_order("PO-2026-1001"), SUP).issues) == ["not_received"]


def test_po_of_another_supplier_is_a_mismatch():
    e = erp(pos=[po(supplier="SUP-2")])
    assert "po_mismatch" in codes(approval_blockers(e, invoice(), "PO-2026-1001"))


def test_no_po_reference_needs_exactly_one_clean_candidate():
    e = erp(pos=[po(), po("PO-2026-1002")], receipts=[receipt(), receipt("PO-2026-1002")])
    assert "po_ambiguous" in codes(approval_blockers(e, invoice(po_number=None), "PO-2026-1001"))
    assert policy_decision(e, invoice(po_number=None))[:2] == ("HOLD", ["po_ambiguous"])
    e1 = erp(pos=[po(), po("PO-2026-1002", qty=(1, 1))], receipts=[receipt(), receipt("PO-2026-1002", q=(1, 1))])
    assert policy_decision(e1, invoice(po_number=None)) == ("APPROVE", [], "PO-2026-1001")


def test_misprinted_po_resolves_only_when_the_misprint_is_not_a_real_po():
    e = erp(pos=[po(), po("PO-2026-1010", qty=(1, 1))], receipts=[receipt()])
    assert policy_decision(e, invoice(po_number="PO-2026-1100")) == ("APPROVE", [], "PO-2026-1001")
    assert "po_mismatch" in codes(approval_blockers(e, invoice(po_number="PO-2026-1010"), "PO-2026-1001"))


def test_bank_change_and_duplicate():
    e = erp()
    assert codes(approval_blockers(e, invoice(bank="Routing 021000021  Account 9999999999"),
                                   "PO-2026-1001")) == ["bank_mismatch"]
    paid = PaidInvoice(supplier_id="SUP-1", invoice_number="INV 1", invoice_date="2026-08-10", total=145, currency="USD")
    assert policy_decision(erp(paid=[paid]), invoice())[:2] == ("REJECT", ["duplicate"])


def test_normalisers():
    assert norm_po("Réf. commande : po 2026-0412") == "PO-2026-0412"
    assert norm_po("PO2026 0412") == "PO-2026-0412"
    assert norm_po("") is None
    assert norm_bank("IBAN : FR76 3000 6000 0112") == "FR76300060000112"
    assert norm_bank("Sort code 30-90-12  Account 12345678") == "30901212345678"
    assert norm_bank("Remit to: Routing 021000021 Account 4839201177") == norm_bank("021000021 / 4839201177")
