import json
import pytest

from my_own_ai.finance_module import Finance, replay
from my_own_ai.modules import CheckpointStore, ModuleController, default_modules
from my_own_ai.reasoning import Stage, Status


def test_ledger_reconciles_cash_inventory_fees_and_settlement():
    result = replay([20, 30, 50], [{"index":0,"side":"buy","units":4},
                                  {"index":1,"side":"sell","units":2}], 100, 2)
    assert result["simulated"] is True
    assert result["final_cents"] == 174
    assert result["net_cents"] == 74
    assert result["ledger"][-1]["inventory"] == 0
    assert result["ledger"][-1]["side"] == "settle"


@pytest.mark.parametrize("orders", [
    [{"index":0,"side":"buy","units":6}],
    [{"index":0,"side":"sell","units":1}],
    [{"index":0,"side":"buy","units":True}],
    [{"index":0,"side":"deposit","units":1}],
    [{"index":0,"side":"buy","units":1,"fill":999}],
    [{"index":1,"side":"buy","units":1},{"index":0,"side":"sell","units":1}],
])
def test_invalid_proposals_are_not_executed(orders):
    with pytest.raises(ValueError):
        replay([20,30], orders, 100, 2)


def test_cash_and_inventory_cannot_be_invented():
    assert replay([20,30], [], 100, 2)["final_cents"] == 100
    with pytest.raises(ValueError):
        replay([20,30], [{"index":0,"side":"buy","units":1}], True, 0)


class FixtureProposer:
    """Scripted controller fixture, not a neural competency test."""
    def __init__(self):
        self.requests = []

    def propose(self, request):
        self.requests.append(request)
        if request.stage == Stage.FORMULATE:
            return "Fixture plan"
        if request.stage == Stage.CONSTRUCT:
            return "[]"
        return '[{"index":0,"side":"buy","units":4}]'


def test_module_is_opt_in_and_replay_survives_switch_and_cold_restore(tmp_path):
    assert "finance" not in {m.name for m in default_modules(tmp_path)}
    file = tmp_path/"scenario.json"
    file.write_text('{"prices_cents":[20,50]}')
    store = CheckpointStore(tmp_path/"memory")
    inputs = dict(scenario="scenario.json",initial_cents=100,target_cents=180,fee_cents=2)
    objective = "Fixture objective: settle at least 180 cents"
    fixture = FixtureProposer()
    with ModuleController(store, [*default_modules(tmp_path), Finance(tmp_path)], fixture) as core:
        core.switch("finance")
        core.start("scenario_replay",objective,inputs=inputs)
        core.step()
        first = core.step()
        assert first["candidate"] == "[]"
        core.switch("conversation")
        assert sum(m.loaded for m in core.modules.values()) == 1
        core.switch("finance")
        failed = core.step()
        assert failed["status"] == Status.REQUIRES_CORRECTION
        assert json.loads(failed["feedback"])["final_cents"] == 100
    file.write_text('{"prices_cents":[20,1]}')
    with ModuleController(store, [Finance(tmp_path)], fixture) as core:
        core.switch("finance")
        core.step()
        result = core.step()
        assert result["status"] == Status.COMPLETED
        assert result["objective"] == objective and result["inputs"] == inputs
        proof = json.loads(result["feedback"])
        assert proof["simulated"] is True
        assert proof["final_cents"] == 216
        assert proof["target_cents"] == 180
        assert proof["source_sha256"] == first["context"]["scenario"]["sha256"]
    assert all(r.objective == objective for r in fixture.requests)


def test_owner_selected_scenario_scope_and_strict_input(tmp_path):
    finance = Finance(tmp_path)
    with pytest.raises(ValueError):
        finance.scenario("../outside.json")
    (tmp_path/"scenario.json").write_text('{"prices_cents":[20,30],"prices_cents":[1,2]}')
    with pytest.raises(ValueError):
        finance.scenario("scenario.json")
    with pytest.raises(ValueError):
        finance.validate_inputs("live_trade", {})
    assert set(finance.tool_handlers()) == {"scenario","replay","search_library","causal_replay"}
