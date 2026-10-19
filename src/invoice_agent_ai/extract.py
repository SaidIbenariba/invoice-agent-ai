"""Read an invoice with a local vision-language model.

Reuses Private Invoice AI's extractor (routing, prompt, deterministic normalisation) and asks for two
more fields that accounts payable needs: the PO reference and the bank account.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import httpx
from invoice_ai import extract as base
from invoice_ai.normalize import to_invoice

from .models import AgentInvoice, norm_po

DEFAULT_MODEL = base.DEFAULT_MODEL  # qwen2.5vl

RAW_SCHEMA = copy.deepcopy(base.RAW_SCHEMA)
RAW_SCHEMA["properties"]["po_number"] = {"type": "string"}
RAW_SCHEMA["properties"]["bank_account"] = {"type": "string"}
RAW_SCHEMA["required"] += ["po_number", "bank_account"]

PROMPT = base.PROMPT + """
- po_number: the customer's purchase order reference (PO Number, Your PO, Réf. commande, Bon de commande N°), "" if none is printed
- bank_account: the supplier's bank details for payment exactly as printed (IBAN, RIB, sort code and account, routing and account number)"""


def to_agent_invoice(raw: dict) -> AgentInvoice:
    inv = to_invoice(raw)
    return AgentInvoice(**inv.model_dump(), po_number=norm_po(raw.get("po_number")),
                        bank_account=(raw.get("bank_account") or "").strip() or None)


def extract(path: Path, model: str = DEFAULT_MODEL, timeout: float = 900) -> dict:
    img, text = base.load_page(path)
    if len(text) > 80:
        route, msg = "text", {"role": "user", "content": PROMPT + "\n\nInvoice text:\n" + text}
    else:
        route, msg = "vision", {"role": "user", "content": PROMPT, "images": [base._b64(img)]}
    body = {"model": model, "messages": [msg], "format": RAW_SCHEMA, "stream": False,
            "options": {"temperature": 0, "num_ctx": 8192}}
    t0 = time.perf_counter()
    r = httpx.post(base.OLLAMA_URL, json=body, timeout=timeout)
    r.raise_for_status()
    raw = json.loads(r.json()["message"]["content"])
    return {"file": path.name, "route": route, "seconds": round(time.perf_counter() - t0, 1), "raw": raw}


def extract_cached(path: Path, cache_dir: Path, model: str = DEFAULT_MODEL, log=print) -> dict:
    cache = cache_dir / f"{path.stem}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    rec = extract(path, model)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(rec, indent=2, ensure_ascii=False))
    log(f"  {path.name}: {rec['route']} path, {rec['seconds']:.0f}s")
    return rec
