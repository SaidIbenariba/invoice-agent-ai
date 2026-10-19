"""Run a batch: every invoice in the inbox, in order, through the agent (or the rules baseline)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .agent import OllamaChat, run_agent, run_rules
from .erp import ERP
from .evaluate import load_truth, score
from .extract import DEFAULT_MODEL, extract_cached
from .tools import Case, Toolset

DOC_SUFFIXES = {".pdf", ".jpg", ".jpeg", ".png"}


def list_docs(inbox: Path) -> list[Path]:
    return sorted(p for p in inbox.iterdir() if p.suffix.lower() in DOC_SUFFIXES) if inbox.exists() else []


def case_record(case: Case) -> dict:
    ex = case.extraction or {}
    return {**case.summary(), "invoice": case.invoice.model_dump() if case.invoice else None,
            "route": ex.get("route"), "extract_seconds": ex.get("seconds"), "steps": case.steps}


def process_one(erp: ERP, path: Path, reader, mode: str = "agent", llm=None, cases_dir: Path | None = None) -> dict:
    case = Case(path.stem, path)
    tools = Toolset(erp, case, reader)
    if mode == "rules":
        run_rules(case, tools)
    else:
        run_agent(case, tools, llm or OllamaChat())
    rec = case_record(case)
    if cases_dir:
        cases_dir.mkdir(parents=True, exist_ok=True)
        (cases_dir / f"{case.doc_id}.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False, default=str))
    return rec


def run_batch(data_dir: Path, out_dir: Path, mode: str = "agent", llm=None, extract_model: str = DEFAULT_MODEL,
              log=print, limit: int | None = None) -> dict:
    ws = out_dir / mode
    if ws.exists():
        shutil.rmtree(ws)
    erp = ERP.load(data_dir / "erp", workspace=ws)
    docs = list_docs(data_dir / "inbox")[:limit]
    reader = lambda p: extract_cached(p, out_dir / "extractions", extract_model, log)
    recs = []
    for d in docs:
        rec = process_one(erp, d, reader, mode, llm, ws / "cases")
        recs.append(rec)
        log(f"{rec['doc_id']}: {rec['decision']:<7} {','.join(rec['reasons']) or '-':<28} "
            f"PO {rec['po_number'] or '-':<13} {rec['tool_calls']:>2} calls  {rec['seconds']:>5.1f}s"
            + (f"  guard blocks {rec['guard_blocks']}" if rec["guard_blocks"] else ""))
    truth = load_truth(data_dir)
    report = score(recs, truth) if truth else {"cases": recs}
    (ws / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return report
