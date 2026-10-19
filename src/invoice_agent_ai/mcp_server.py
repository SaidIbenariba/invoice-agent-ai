"""MCP server: the accounts-payable desk as tools for Claude Desktop or any MCP client.

    uv run python -m invoice_agent_ai.mcp_server        # stdio

Ask the assistant things like "process the new invoices", "why is inv_014 on hold?",
"what is still open on PO-2026-4098?".
"""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from .service import Desk

mcp = MCPServer("invoice-agent", instructions="Accounts payable desk: process supplier invoices with a local AI agent, then inspect decisions, the review queue, purchase orders and supplier orders.")
_desk: Desk | None = None


def desk() -> Desk:
    global _desk
    if _desk is None:
        _desk = Desk()
    return _desk


@mcp.tool()
def list_inbox() -> list[dict]:
    """Supplier invoices waiting in the inbox that have not been processed yet."""
    return desk().pending()


@mcp.tool()
def process_invoice(file: str) -> dict:
    """Run the accounts-payable agent on one inbox file (e.g. 'inv_007.pdf'): it reads the invoice, checks it,
    matches it to the purchase order and goods receipt, then approves, holds or rejects it."""
    return desk().process(file)


@mcp.tool()
def get_case(doc_id: str) -> dict:
    """Decision, reasons, note and the agent's step-by-step trail for a processed invoice (e.g. 'inv_014')."""
    c = desk().case(doc_id)
    if c is None:
        return {"found": False}
    return {**{k: c[k] for k in ("doc_id", "decision", "reasons", "po_number", "note", "guard_blocks")},
            "steps": [{"tool": s["tool"], "args": s["args"]} for s in c["steps"]]}


@mcp.tool()
def review_queue() -> list[dict]:
    """Invoices on hold for a person, with their reason codes and notes."""
    return desk().erp.review_queue


@mcp.tool()
def approved_for_payment() -> list[dict]:
    """Invoices approved in this session, with PO and scheduled payment date."""
    return desk().erp.postings


@mcp.tool()
def purchase_order(po_number: str) -> dict:
    """A purchase order with quantities ordered, received and already invoiced per line."""
    return desk().purchase_order(po_number) or {"found": False}


@mcp.tool()
def supplier_open_orders(supplier: str) -> dict:
    """Open purchase orders of a supplier, found by name or tax ID."""
    return desk().supplier_open_orders(supplier)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
