"""One live accounts-payable desk, shared by the HTTP API (n8n) and the MCP server.

Configured with environment variables:
  AGENT_DATA        folder with erp/ and inbox/              (default data/dev)
  AGENT_WORKSPACE   where postings, review queue, cases go   (default out/live)
  AGENT_MODE        agent | rules                            (default agent)
  AGENT_MODEL       tool-calling model on Ollama             (default qwen2.5:7b)
  AGENT_EXTRACTION_CACHE  folder of cached extractions to reuse (default <workspace>/extractions)
"""

from __future__ import annotations

import json
import os
import shutil
import threading
from pathlib import Path

from .agent import DEFAULT_AGENT_MODEL, OllamaChat
from .erp import ERP
from .extract import extract_cached
from .pipeline import list_docs, process_one


class Desk:
    def __init__(self, data: Path | None = None, workspace: Path | None = None, mode: str | None = None,
                 model: str | None = None, cache: Path | None = None):
        self.data = Path(data or os.environ.get("AGENT_DATA", "data/dev"))
        self.workspace = Path(workspace or os.environ.get("AGENT_WORKSPACE", "out/live"))
        self.mode = mode or os.environ.get("AGENT_MODE", "agent")
        self.model = model or os.environ.get("AGENT_MODEL", DEFAULT_AGENT_MODEL)
        self.cache = Path(cache or os.environ.get("AGENT_EXTRACTION_CACHE", self.workspace / "extractions"))
        self.inbox = self.data / "inbox"
        self.lock = threading.Lock()
        self.erp = ERP.load(self.data / "erp", workspace=self.workspace, fresh=False)

    @property
    def cases_dir(self) -> Path:
        return self.workspace / "cases"

    def _read(self, path: Path) -> dict:
        return extract_cached(path, self.cache)

    def resolve(self, file: str) -> Path:
        """Only files inside the inbox can be processed."""
        p = (self.inbox / Path(file).name).resolve()
        if p.parent != self.inbox.resolve() or not p.exists():
            raise FileNotFoundError(f"{file} is not in the inbox")
        return p

    # ---------- reads ----------

    def pending(self) -> list[dict]:
        done = {p.stem for p in self.cases_dir.glob("*.json")}
        return [{"file": d.name, "doc_id": d.stem} for d in list_docs(self.inbox) if d.stem not in done]

    def case(self, doc_id: str) -> dict | None:
        p = self.cases_dir / f"{Path(doc_id).stem}.json"
        return json.loads(p.read_text()) if p.exists() else None

    def cases(self) -> list[dict]:
        out = []
        for p in sorted(self.cases_dir.glob("*.json")):
            c = json.loads(p.read_text())
            out.append({k: c[k] for k in ("doc_id", "file", "decision", "reasons", "po_number", "note",
                                          "tool_calls", "guard_blocks", "seconds")})
        return out

    def purchase_order(self, po_number: str) -> dict | None:
        po = self.erp.purchase_order(po_number)
        if po is None:
            return None
        return {**po.model_dump(), "received": self.erp.received(po.po_number),
                "already_invoiced": self.erp.invoiced(po.po_number)}

    def supplier_open_orders(self, name_or_tax_id: str) -> dict:
        s, cands = self.erp.find_supplier(name=name_or_tax_id, tax_id=name_or_tax_id)
        if s is None:
            return {"found": False, "close_names": cands}
        return {"found": True, "supplier_id": s.supplier_id, "name": s.name,
                "open_purchase_orders": [p.model_dump() for p in self.erp.open_pos(s.supplier_id)]}

    # ---------- actions ----------

    def process(self, file: str) -> dict:
        path = self.resolve(file)
        with self.lock:
            done = self.case(path.stem)
            if done:
                return {**{k: done[k] for k in ("doc_id", "file", "decision", "reasons", "po_number", "note")},
                        "already_processed": True}
            llm = OllamaChat(self.model) if self.mode == "agent" else None
            rec = process_one(self.erp, path, self._read, self.mode, llm, self.cases_dir)
        drafts = [d for d in self.erp.drafts if d["doc_id"] == rec["doc_id"]]
        posting = next((p for p in self.erp.postings if p["doc_id"] == rec["doc_id"]), None)
        inv = rec.get("invoice") or {}
        return {k: rec[k] for k in ("doc_id", "file", "decision", "reasons", "po_number", "note", "tool_calls",
                                    "guard_blocks", "seconds")} | {
            "supplier_name": inv.get("supplier_name"), "invoice_number": inv.get("invoice_number"),
            "total": inv.get("total"), "currency": inv.get("currency"),
            "scheduled_payment_date": posting["scheduled_payment_date"] if posting else None,
            "email_drafts": drafts}

    def reset(self) -> None:
        with self.lock:
            for name in ("cases", "postings.json", "review_queue.json", "rejected.json", "drafts.json"):
                p = self.workspace / name
                if p.is_dir():
                    shutil.rmtree(p)
                elif p.exists():
                    p.unlink()
            self.erp = ERP.load(self.data / "erp", workspace=self.workspace, fresh=True)
