"""HTTP API and MCP server, end to end, with the rules baseline and pre-cached perfect extractions."""

import asyncio
import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

from conftest import raw_from_truth


@pytest.fixture()
def live(world, tmp_path, monkeypatch):
    d, truths = world
    cache = tmp_path / "cache"
    cache.mkdir()
    for doc_id, t in truths.items():
        (cache / f"{doc_id}.json").write_text(json.dumps(
            {"file": t["file"], "route": "text", "seconds": 0.0, "raw": raw_from_truth(t["invoice"])}))
    env = {"AGENT_DATA": str(d), "AGENT_WORKSPACE": str(tmp_path / "ws"), "AGENT_MODE": "rules",
           "AGENT_EXTRACTION_CACHE": str(cache)}
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return env, truths


def test_http_api_processes_inbox_and_refuses_paths_outside_it(live):
    import invoice_agent_ai.api as api
    api._desk = None
    _, truths = live
    client = TestClient(api.app)
    pending = client.get("/inbox").json()["files"]
    assert len(pending) == 24
    first = pending[0]["file"]
    r = client.post("/invoices/process", json={"file": first}).json()
    assert r["decision"] == truths[pending[0]["doc_id"]]["expected"]["decision"]
    assert len(client.get("/inbox").json()["files"]) == 23
    assert client.post("/invoices/process", json={"file": first}).json()["already_processed"] is True
    assert client.post("/invoices/process", json={"file": "../erp/suppliers.json"}).status_code == 404
    assert client.get(f"/cases/{pending[0]['doc_id']}").json()["steps"]
    client.post("/reset")
    assert len(client.get("/inbox").json()["files"]) == 24


def test_mcp_server_over_stdio(live):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    env, truths = live
    dup = next(t for t in truths.values() if t["scenario"] == "duplicate")

    async def go():
        params = StdioServerParameters(command=sys.executable, args=["-m", "invoice_agent_ai.mcp_server"],
                                       env={**os.environ, **env})
        async with stdio_client(params) as (r, w):
            async with ClientSession(r, w) as s:
                await s.initialize()
                names = {t.name for t in (await s.list_tools()).tools}
                res = await s.call_tool("process_invoice", {"file": dup["file"]})
                case = await s.call_tool("get_case", {"doc_id": dup["doc_id"]})
                return names, json.loads(res.content[0].text), json.loads(case.content[0].text)

    names, res, case = asyncio.run(go())
    assert {"list_inbox", "process_invoice", "get_case", "review_queue", "purchase_order"} <= names
    assert res["decision"] == "REJECT" and case["reasons"] == ["duplicate"]
