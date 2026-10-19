"""HTTP API for n8n or any other orchestrator.

    uv run uvicorn invoice_agent_ai.api:app --port 8000

POST /invoices/process {"file": "inv_007.pdf"} runs the agent on one inbox file and returns its decision.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .service import Desk

app = FastAPI(title="Invoice Agent AI", description="Local AI agent for accounts payable 3-way matching")
_desk: Desk | None = None


def desk() -> Desk:
    global _desk
    if _desk is None:
        _desk = Desk()
    return _desk


class ProcessRequest(BaseModel):
    file: str


@app.get("/health")
def health() -> dict:
    d = desk()
    return {"ok": True, "mode": d.mode, "model": d.model, "inbox": str(d.inbox)}


@app.get("/inbox")
def inbox() -> dict:
    """Invoices waiting in the inbox that have not been processed yet."""
    return {"files": desk().pending()}


@app.post("/invoices/process")
def process(req: ProcessRequest) -> dict:
    try:
        return desk().process(req.file)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))


@app.get("/cases")
def cases() -> list[dict]:
    return desk().cases()


@app.get("/cases/{doc_id}")
def case(doc_id: str) -> dict:
    c = desk().case(doc_id)
    if c is None:
        raise HTTPException(404, f"no case {doc_id}")
    return c


@app.get("/review-queue")
def review_queue() -> list[dict]:
    return desk().erp.review_queue


@app.get("/postings")
def postings() -> list[dict]:
    return desk().erp.postings


@app.get("/drafts")
def drafts() -> list[dict]:
    return desk().erp.drafts


@app.post("/reset")
def reset() -> dict:
    desk().reset()
    return {"reset": True}
