# Design: Invoice Agent AI

## The job

An accounts payable clerk who receives a supplier invoice does four things before money leaves:

1. read the invoice
2. check it is internally correct (lines, tax, totals, required fields, not a duplicate)
3. match it to what was ordered (purchase order) and what actually arrived (goods receipt): the "3-way match"
4. confirm the supplier and the bank account are the ones on file

Then they pay, put it on hold with a reason, or reject it. This project gives that job to an AI agent running on a local model, with one rule: **the model plans and explains, code enforces the money rules.**

## Pieces

| Piece | What it does |
|---|---|
| `synth.py` | Synthetic ERP (suppliers, purchase orders, goods receipts, paid-invoice history) and supplier invoices rendered as PDFs or phone photos, with a ground-truth decision for each |
| `extract.py` | Reads an invoice with a local vision-language model (Private Invoice AI's extractor, plus PO reference and bank account) |
| `match.py` | Deterministic 3-way match: invoice lines vs PO lines vs received quantities, price tolerance |
| `tools.py` | The agent's tools, as JSON schemas + Python functions. Terminal actions (`approve_for_payment`, `hold_for_review`, `reject_invoice`) are guarded |
| `agent.py` | Tool-calling loop against any OpenAI-compatible endpoint (Ollama by default), with a step-by-step audit trail |
| `evaluate.py` | Scores decisions, reasons and PO matching against ground truth |
| `api.py` | FastAPI service that n8n calls |
| `mcp_server.py` | The same tools exposed over MCP (Claude Desktop, any MCP client) |
| `n8n/` | Workflow: new file in inbox → agent → route by decision |
| `app.py` | Streamlit review screen: decision, reasons, PO vs invoice vs receipt, agent trace |

## Guard

`approve_for_payment` re-runs every check in code before posting anything:
supplier on file, bank account matches, invoice checks pass, not a duplicate, PO exists / belongs to the supplier / is open / same currency, every line matched, billed quantity ≤ received, price within tolerance.
If any fails, the tool refuses, returns the reasons, and the agent has to hold the invoice. Refusals are counted ("guard blocks") so the report shows how often the model tried something unsafe.

`reject_invoice` is only allowed for confirmed duplicates. Emails to suppliers are drafts, never sent.

## Scenarios (24 per batch)

| Scenario | Count | Expected |
|---|---|---|
| clean (incl. partial delivery billed correctly) | 8 | APPROVE |
| no PO reference, exactly one open PO matches | 2 | APPROVE (agent finds the PO) |
| price above PO | 2 | HOLD `price_variance` |
| billed more than received | 2 | HOLD `qty_variance` |
| goods not received | 1 | HOLD `not_received` |
| bank account differs from the one on file | 2 | HOLD `bank_mismatch` |
| duplicate of an invoice already paid | 2 | REJECT `duplicate` |
| arithmetic / tax / required-field error | 2 | HOLD `invoice_check_failed` |
| supplier not on file | 1 | HOLD `unknown_supplier` |
| no PO reference, two identical open POs | 1 | HOLD `po_ambiguous` |
| PO belongs to another supplier | 1 | HOLD `po_mismatch` |

## Metrics

- decision accuracy (APPROVE / HOLD / REJECT)
- **wrong payments** (approved but should not be): target 0
- clean invoices approved (automation rate)
- reason accuracy (expected reason among the agent's reasons)
- PO found when the invoice has no PO reference
- guard blocks, tool errors, steps and seconds per invoice

Same method as the other demos: build on a dev batch, fix with tests, freeze, then generate a held-out batch with a new seed and run it once.
