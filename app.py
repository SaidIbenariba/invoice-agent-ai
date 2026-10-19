"""Review screen: what the agent decided, why, and every step it took.

    uv run streamlit run app.py -- --set dev --mode agent
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pypdfium2 as pdfium
import streamlit as st
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument("--set", default="dev")
p.add_argument("--mode", default="agent")
args, _ = p.parse_known_args(sys.argv[1:])

DATA, WS = Path("data") / args.set, Path("out") / args.set / args.mode
st.set_page_config(page_title="Invoice Agent AI", layout="wide")

BADGE = {"APPROVE": "🟢 APPROVED", "HOLD": "🟠 ON HOLD", "REJECT": "🔴 REJECTED"}


@st.cache_data
def page_image(path: str) -> Image.Image:
    f = Path(path)
    if f.suffix.lower() == ".pdf":
        pg = pdfium.PdfDocument(str(f))[0]
        return pg.render(scale=1.4).to_pil()
    return Image.open(f)


def load(name: str, default):
    f = WS / name
    return json.loads(f.read_text()) if f.exists() else default


cases = [json.loads(f.read_text()) for f in sorted((WS / "cases").glob("*.json"))]
report = load("report.json", {})
drafts = load("drafts.json", [])

st.title("Invoice Agent AI")
st.caption(f"Accounts payable 3-way match by a local AI agent · batch `{args.set}` · mode `{args.mode}`")

if not cases:
    st.info(f"No processed invoices in {WS}. Run: uv run python -m invoice_agent_ai run --set {args.set}")
    st.stop()

s = report.get("summary")
cols = st.columns(5)
cols[0].metric("Invoices", len(cases))
cols[1].metric("Approved", sum(c["decision"] == "APPROVE" for c in cases))
cols[2].metric("On hold", sum(c["decision"] == "HOLD" for c in cases))
cols[3].metric("Rejected", sum(c["decision"] == "REJECT" for c in cases))
if s:
    cols[4].metric("Wrong payments", s["wrong_payments"], help="Approved although the ground truth says hold or reject")

labels = {f"{BADGE[c['decision']].split()[0]} {c['doc_id']} · {(c.get('invoice') or {}).get('supplier_name') or '?'}": c
          for c in cases}
pick = st.sidebar.radio("Invoices", list(labels), index=0)
c = labels[pick]
inv = c.get("invoice") or {}

st.subheader(f"{BADGE[c['decision']]} · {inv.get('supplier_name')} · {inv.get('invoice_number')} · "
             f"{inv.get('total')} {inv.get('currency')}")
if c["reasons"]:
    st.write("**Reasons:** " + ", ".join(f"`{r}`" for r in c["reasons"]))
if c.get("note"):
    st.write(f"**Agent note:** {c['note']}")
meta = [f"{c['tool_calls']} tool calls", f"{c['seconds']:.0f} s agent time"]
if c.get("po_number"):
    meta.insert(0, f"PO {c['po_number']}")
if c.get("guard_blocks"):
    meta.append(f"⚠️ guard refused {c['guard_blocks']} unsafe action(s)")
st.caption(" · ".join(meta))

left, right = st.columns([1, 1.2])
doc = DATA / "inbox" / c["file"]
if doc.exists():
    left.image(page_image(str(doc)), width="stretch")

match = next((st_["result"] for st_ in reversed(c["steps"])
              if st_["tool"] == "three_way_match" and st_["result"].get("lines")), None)
if match is None and inv:  # e.g. rules mode: compute the match for display, against master data only
    from invoice_agent_ai.erp import ERP
    from invoice_agent_ai.match import identify_supplier, three_way_match
    from invoice_agent_ai.models import AgentInvoice

    erp = ERP.load(DATA / "erp")
    ai = AgentInvoice.model_validate(inv)
    po = erp.purchase_order(c.get("po_number") or ai.po_number)
    if po:
        match = three_way_match(erp, ai, po, identify_supplier(erp, ai)).as_dict()
if match:
    right.markdown(f"**3-way match against {match['po_number']}**")
    right.dataframe([{"Invoice line": r["description"], "Billed": r["billed_qty"], "Ordered": r["ordered_qty"],
                      "Received": r["received_qty"], "Already invoiced": r["already_invoiced"],
                      "Invoice price": r["invoice_price"], "PO price": r["po_price"]} for r in match["lines"]],
                    hide_index=True, width="stretch")
    for i in match["issues"]:
        right.warning(f"`{i['code']}` {i['message']}")

right.markdown("**Agent trail**")
for n, step in enumerate(c["steps"], 1):
    who = step.get("by", "agent")
    res = step["result"]
    with right.expander(f"{n}. {step['tool']}({', '.join(f'{k}={v}' for k, v in step['args'].items() if k != 'note')})"
                        + ("" if who == "agent" else f" · by {who}")):
        st.json(res)

mine = [d for d in drafts if d["doc_id"] == c["doc_id"]]
for d in mine:
    with right.expander(f"✉️ Draft to {d['to']} (not sent): {d['subject']}"):
        st.write(d["body"])
