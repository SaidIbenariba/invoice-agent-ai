"""The agent: a tool-calling loop over the AP tools, with an audit trail of every step.

Works with Ollama's native API (default, local) or any OpenAI-compatible endpoint (OpenAI, OpenRouter,
vLLM). Whatever the model does, three things are enforced in code: approval and rejection are guarded
(see tools.py), every step is logged, and a case the model does not finish is put on hold.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

import httpx

from .match import policy_decision
from .tools import SPECS, Case, Toolset, dumps

DEFAULT_AGENT_MODEL = "qwen2.5:7b"

SYSTEM = """You are the accounts payable agent of Riverside Group. You process one supplier invoice at a time with the tools. Every fact you use must come from a tool result; never guess numbers, POs or suppliers.

Procedure:
1. read_invoice.
2. check_invoice. If it reports a duplicate, call reject_invoice. If it reports any other problem, call hold_for_review with reason invoice_check_failed.
3. find_supplier. If the supplier is not on file, call hold_for_review with reason unknown_supplier.
4. verify_bank_account. If it does not match, call draft_email_to_supplier asking the supplier to confirm its bank details by phone on the number we have on file, then hold_for_review with reason bank_mismatch.
5. Find the purchase order.
   - If the invoice shows a PO number, call get_purchase_order. If it exists, use it, even if it belongs to another supplier: three_way_match will report that.
   - If the invoice has no PO number, or get_purchase_order does not find it, call search_purchase_orders for the supplier and three_way_match on each open PO. Use a PO only if exactly one of them is clean. If several are clean, hold with po_ambiguous. If none is clean, hold with po_not_found.
6. three_way_match on the PO you use.
7. Decide.
   - approve_for_payment only when check_invoice passed, the bank account matched and three_way_match is clean.
   - Otherwise call hold_for_review with the reason codes the tools reported (price_variance, qty_variance, not_received, po_mismatch, unmatched_line, ...). For a price or quantity difference, first call draft_email_to_supplier asking for a corrected invoice or a credit note.
   - If approve_for_payment is refused, call hold_for_review with the codes it gives.
8. Finish with exactly one of approve_for_payment, hold_for_review or reject_invoice. Notes are one or two plain sentences for the person who reviews the case."""

NUDGE = "Continue with the tools. Finish with approve_for_payment, hold_for_review or reject_invoice."


@dataclass
class Call:
    id: str | None
    name: str
    args: dict


@dataclass
class Reply:
    message: dict
    calls: list[Call]


def _args(a) -> dict:
    if isinstance(a, dict):
        return a
    try:
        return json.loads(a or "{}")
    except json.JSONDecodeError:
        return {"_unparsed": a}


class OllamaChat:
    """Ollama native /api/chat (lets us set the context window)."""

    def __init__(self, model: str = DEFAULT_AGENT_MODEL, url: str = "http://localhost:11434", num_ctx: int = 16384):
        self.model, self.url, self.num_ctx = model, url.rstrip("/"), num_ctx

    def chat(self, messages: list[dict], tools: list[dict]) -> Reply:
        body = {"model": self.model, "messages": messages, "tools": tools, "stream": False,
                "options": {"temperature": 0, "num_ctx": self.num_ctx}}
        r = httpx.post(f"{self.url}/api/chat", json=body, timeout=600)
        r.raise_for_status()
        msg = r.json()["message"]
        calls = [Call(None, c["function"]["name"], _args(c["function"].get("arguments")))
                 for c in msg.get("tool_calls") or []]
        return Reply({"role": "assistant", "content": msg.get("content", ""),
                      **({"tool_calls": msg["tool_calls"]} if msg.get("tool_calls") else {})}, calls)

    def tool_message(self, call: Call, content: str) -> dict:
        return {"role": "tool", "tool_name": call.name, "content": content}


class OpenAIChat:
    """Any OpenAI-compatible /chat/completions endpoint (OpenAI, OpenRouter, vLLM, Ollama /v1)."""

    def __init__(self, model: str, base_url: str, api_key: str | None = None):
        self.model, self.base_url = model, base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")

    def chat(self, messages: list[dict], tools: list[dict]) -> Reply:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        r = httpx.post(f"{self.base_url}/chat/completions", headers=headers, timeout=600,
                       json={"model": self.model, "messages": messages, "tools": tools, "temperature": 0})
        r.raise_for_status()
        msg = r.json()["choices"][0]["message"]
        calls = [Call(c.get("id"), c["function"]["name"], _args(c["function"].get("arguments")))
                 for c in msg.get("tool_calls") or []]
        out = {"role": "assistant", "content": msg.get("content") or ""}
        if msg.get("tool_calls"):
            out["tool_calls"] = msg["tool_calls"]
        return Reply(out, calls)

    def tool_message(self, call: Call, content: str) -> dict:
        return {"role": "tool", "tool_call_id": call.id, "content": content}


def _finish_on_hold(case: Case, tools: Toolset, note: str, turn: int) -> None:
    result, _ = tools.call("hold_for_review", {"reasons": ["agent_error"], "note": note})
    case.steps.append({"turn": turn, "tool": "hold_for_review", "args": {"reasons": ["agent_error"]},
                       "result": result, "seconds": 0.0, "by": "code (fallback)"})


def run_agent(case: Case, tools: Toolset, llm, max_turns: int = 16, max_errors_in_a_row: int = 4) -> Case:
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"New supplier invoice in the inbox: {case.path.name}. Process it."}]
    nudges, t0, turn, errors_in_a_row = 0, time.perf_counter(), 0, 0
    try:
        for turn in range(1, max_turns + 1):
            if errors_in_a_row >= max_errors_in_a_row:
                _finish_on_hold(case, tools, "The agent kept getting tool errors; process this invoice manually.", turn)
                break
            reply = llm.chat(msgs, SPECS)
            msgs.append(reply.message)
            if not reply.calls:
                if nudges >= 2:
                    break
                nudges += 1
                msgs.append({"role": "user", "content": NUDGE})
                continue
            finished = False
            for call in reply.calls:
                t1 = time.perf_counter()
                result, finished = tools.call(call.name, call.args)
                errors_in_a_row = errors_in_a_row + 1 if "error" in result else 0
                case.steps.append({"turn": turn, "tool": call.name, "args": call.args, "result": result,
                                   "seconds": round(time.perf_counter() - t1, 2)})
                msgs.append(llm.tool_message(call, dumps(result)))
                if finished:
                    break
            if finished:
                break
    except (httpx.HTTPError, KeyError) as e:
        _finish_on_hold(case, tools, f"The agent stopped with an error ({type(e).__name__}); process manually.", turn)
    if case.decision is None:
        _finish_on_hold(case, tools, "The agent did not finish this invoice; process it manually.", turn)
    case.seconds = time.perf_counter() - t0
    return case


def run_rules(case: Case, tools: Toolset) -> Case:
    """Baseline without a language model: the policy in code, on the same extracted invoice."""
    t0 = time.perf_counter()
    decision, reasons, po = policy_decision(tools.erp, tools.invoice())
    if decision == "APPROVE":
        call = ("approve_for_payment", {"po_number": po, "note": "All checks passed (rules)."})
    elif decision == "REJECT":
        call = ("reject_invoice", {"note": "Duplicate of an invoice already paid (rules)."})
    else:
        call = ("hold_for_review", {"reasons": reasons, "note": "Held by rules."})
    result, _ = tools.call(*call)
    case.steps.append({"turn": 1, "tool": call[0], "args": call[1], "result": result, "seconds": 0.0, "by": "rules"})
    if case.decision is None:
        _finish_on_hold(case, tools, "Rules could not finish this invoice.", 1)
    case.seconds = time.perf_counter() - t0
    return case
