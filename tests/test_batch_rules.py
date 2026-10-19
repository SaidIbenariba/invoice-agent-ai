"""End to end without a language model: generated world, perfect reading, rules baseline, guards, scoring."""

import pdfplumber

from invoice_agent_ai.erp import ERP
from invoice_agent_ai.evaluate import load_truth, score
from invoice_agent_ai.pipeline import list_docs, process_one


def test_generated_invoices_print_po_and_bank(world):
    d, truths = world
    for doc in list_docs(d / "inbox"):
        t = truths[doc.stem]["invoice"]
        with pdfplumber.open(doc) as pdf:
            text = pdf.pages[0].extract_text().replace(" ", "")
        if t["po_number"]:
            assert t["po_number"] in text
        assert t["bank_account"].replace(" ", "")[-6:] in text


def test_rules_baseline_on_perfect_reading_gets_every_case(world, perfect_reader, tmp_path):
    d, _ = world
    erp = ERP.load(d / "erp", workspace=tmp_path)
    recs = [process_one(erp, doc, perfect_reader, mode="rules") for doc in list_docs(d / "inbox")]
    s = score(recs, load_truth(d))["summary"]
    assert s["invoices"] == 24
    assert s["decisions_and_reasons_correct"] == 24
    assert s["wrong_payments"] == 0
    assert s["guard_blocks"] == 0
    assert len(erp.postings) == 10 and len(erp.rejected) == 2 and len(erp.review_queue) == 12
