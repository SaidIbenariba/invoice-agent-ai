"""CLI.

    python -m invoice_agent_ai generate --set dev --seed 11      # synthetic ERP + invoices + truth
    python -m invoice_agent_ai extract --set dev                 # read every invoice once (cached)
    python -m invoice_agent_ai run --set dev                     # the agent, local model
    python -m invoice_agent_ai run --set dev --mode rules        # baseline: the policy in code, no LLM
    python -m invoice_agent_ai run --set dev --llm openai --base-url https://openrouter.ai/api/v1 --model <id>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agent import DEFAULT_AGENT_MODEL, OllamaChat, OpenAIChat
from .evaluate import load_truth, score
from .extract import DEFAULT_MODEL, extract_cached
from .pipeline import list_docs, run_batch
from .synth import generate


def main() -> None:
    p = argparse.ArgumentParser(prog="invoice_agent_ai")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("generate", "extract", "run", "rescore"):
        s = sub.add_parser(name)
        s.add_argument("--set", default="dev", help="batch name: data/<set>, out/<set>")
        s.add_argument("--data", default="data")
        s.add_argument("--out", default="out")
        if name == "generate":
            s.add_argument("--seed", type=int, default=11)
        if name in ("extract", "run"):
            s.add_argument("--extract-model", default=DEFAULT_MODEL)
        if name in ("run", "rescore"):
            s.add_argument("--mode", choices=["agent", "rules"], default="agent")
        if name == "run":
            s.add_argument("--llm", choices=["ollama", "openai"], default="ollama")
            s.add_argument("--model", default=DEFAULT_AGENT_MODEL)
            s.add_argument("--base-url", default=None)
            s.add_argument("--limit", type=int, default=None)
    a = p.parse_args()
    data, out = Path(a.data) / a.set, Path(a.out) / a.set

    if a.cmd == "generate":
        truths = generate(data, seed=a.seed)
        print(f"wrote {len(truths)} invoices, ERP and ground truth to {data}")
        return
    if a.cmd == "extract":
        for d in list_docs(data / "inbox"):
            extract_cached(d, out / "extractions", a.extract_model)
        print(f"extractions cached in {out / 'extractions'}")
        return

    if a.cmd == "rescore":  # rebuild report.json from saved case files (no model calls)
        recs = [json.loads(f.read_text()) for f in sorted((out / a.mode / "cases").glob("*.json"))]
        report = score(recs, load_truth(data))
        (out / a.mode / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
        print(json.dumps(report["summary"], indent=2))
        return

    if a.llm == "openai":
        llm = OpenAIChat(a.model, a.base_url or "https://api.openai.com/v1")
    else:
        llm = OllamaChat(a.model, a.base_url or "http://localhost:11434")
    report = run_batch(data, out, a.mode, llm, a.extract_model, limit=a.limit)
    if "summary" in report:
        print(json.dumps(report["summary"], indent=2))
    print(f"report: {out / a.mode / 'report.json'}")


if __name__ == "__main__":
    main()
