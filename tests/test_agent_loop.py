"""The agent loop and the guards, driven by a scripted fake model (no LLM needed)."""

from invoice_agent_ai.agent import Call, Reply, run_agent
from invoice_agent_ai.erp import ERP
from invoice_agent_ai.tools import Case, Toolset


class Scripted:
    """Replays a fixed list of turns; each turn is a list of (tool, args) or a plain-text answer."""

    def __init__(self, turns):
        self.turns, self.i, self.seen = list(turns), 0, []

    def chat(self, messages, tools):
        self.seen.append(messages[-1])
        turn = self.turns[self.i] if self.i < len(self.turns) else "I am done."
        self.i += 1
        if isinstance(turn, str):
            return Reply({"role": "assistant", "content": turn}, [])
        return Reply({"role": "assistant", "content": ""}, [Call(None, n, a) for n, a in turn])

    def tool_message(self, call, content):
        return {"role": "tool", "tool_name": call.name, "content": content}


def setup(world, reader, tmp_path, scenario):
    d, truths = world
    doc_id = next(k for k, t in sorted(truths.items()) if t["scenario"] == scenario)
    path = d / "inbox" / truths[doc_id]["file"]
    erp = ERP.load(d / "erp", workspace=tmp_path)
    case = Case(doc_id, path)
    return erp, case, Toolset(erp, case, reader), truths[doc_id]


def test_happy_path_approves_and_posts(world, perfect_reader, tmp_path):
    erp, case, tools, t = setup(world, perfect_reader, tmp_path, "clean")
    po = t["expected"]["po_number"]
    llm = Scripted([[("read_invoice", {})], [("check_invoice", {})], [("get_purchase_order", {"po_number": po})],
                    [("three_way_match", {"po_number": po}), ("approve_for_payment", {"po_number": po, "note": "ok"})],
                    [("hold_for_review", {"reasons": ["agent_error"], "note": "never reached"})]])
    run_agent(case, tools, llm)
    assert case.decision == "APPROVE" and case.po_number == po
    assert [s["tool"] for s in case.steps][-1] == "approve_for_payment"
    assert len(erp.postings) == 1 and erp.review_queue == []


def test_guard_refuses_unsafe_approval_and_model_must_hold(world, perfect_reader, tmp_path):
    erp, case, tools, t = setup(world, perfect_reader, tmp_path, "price_over_po")
    po = t["expected"]["po_number"]
    llm = Scripted([[("approve_for_payment", {"po_number": po, "note": "looks fine"})],
                    [("hold_for_review", {"reasons": ["price_variance"], "note": "price above PO"})]])
    run_agent(case, tools, llm)
    assert case.guard_blocks == 1
    assert case.steps[0]["result"]["approved"] is False
    assert {"code": "price_variance"} .items() <= case.steps[0]["result"]["blocked_by_policy"][0].items()
    assert case.decision == "HOLD" and case.reasons == ["price_variance"]
    assert erp.postings == []


def test_reject_only_for_confirmed_duplicates(world, perfect_reader, tmp_path):
    erp, case, tools, _ = setup(world, perfect_reader, tmp_path, "clean")
    result, finished = tools.call("reject_invoice", {"note": "spam"})
    assert not finished and result["rejected"] is False and case.guard_blocks == 1
    erp2, case2, tools2, _ = setup(world, perfect_reader, tmp_path / "b", "duplicate")
    result, finished = tools2.call("reject_invoice", {"note": "already paid"})
    assert finished and case2.decision == "REJECT"


def test_bad_reason_codes_and_unknown_tools_are_errors_not_decisions(world, perfect_reader, tmp_path):
    _, case, tools, _ = setup(world, perfect_reader, tmp_path, "clean")
    assert tools.call("hold_for_review", {"reasons": ["looks weird"], "note": ""})[1] is False
    assert tools.call("pay_now", {})[1] is False
    assert tools.call("get_purchase_order", {"wrong": 1})[1] is False
    assert case.tool_errors == 3 and case.decision is None


def test_extra_arguments_are_ignored_not_errors(world, perfect_reader, tmp_path):
    # dev run 1: the model passed invoice fields to check_invoice (no arguments) and retried forever
    _, case, tools, _ = setup(world, perfect_reader, tmp_path, "clean")
    result, finished = tools.call("check_invoice", {"subtotal": 5732.54, "line_items": [{"amount": 1}]})
    assert "passed" in result and not finished and case.tool_errors == 0


def test_repeated_tool_errors_stop_the_loop(world, perfect_reader, tmp_path):
    _, case, tools, _ = setup(world, perfect_reader, tmp_path, "clean")
    llm = Scripted([[("pay_now", {})]] * 10)
    run_agent(case, tools, llm)
    assert case.decision == "HOLD" and case.reasons == ["agent_error"] and case.tool_errors == 4


def test_model_that_never_finishes_is_put_on_hold_by_code(world, perfect_reader, tmp_path):
    erp, case, tools, _ = setup(world, perfect_reader, tmp_path, "clean")
    llm = Scripted(["Let me think.", "Still thinking.", "Done, I guess."])
    run_agent(case, tools, llm)
    assert case.decision == "HOLD" and case.reasons == ["agent_error"]
    assert case.steps[-1]["by"] == "code (fallback)"
    assert any("Continue with the tools" in str(m.get("content")) for m in llm.seen)


def test_results_state_the_next_policy_step(world, perfect_reader, tmp_path):
    _, _, tools, _ = setup(world, perfect_reader, tmp_path, "po_missing_unique")
    assert "search_purchase_orders" in tools.call("read_invoice", {})[0]["next_step"]
    _, _, tools, _ = setup(world, perfect_reader, tmp_path / "b", "bank_change")
    s = tools.call("find_supplier", {})[0]
    assert "bank_mismatch" in tools.call("verify_bank_account", {"supplier_id": s["supplier_id"]})[0]["next_step"]
    _, _, tools, _ = setup(world, perfect_reader, tmp_path / "c", "clean")
    assert "next_step" not in tools.call("read_invoice", {})[0]
    assert "next_step" not in tools.call("check_invoice", {})[0]


def test_masked_bank_account_in_tool_results(world, perfect_reader, tmp_path):
    _, _, tools, t = setup(world, perfect_reader, tmp_path, "clean")
    inv, _ = tools.call("read_invoice", {})
    assert inv["bank_account"].startswith("****") and t["invoice"]["bank_account"] not in str(inv)
