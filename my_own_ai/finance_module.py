"""Development-only finance instruments; no broker, network, or live execution."""
import hashlib
import json
from pathlib import Path

from .modules import BasicModule, Manifest, Tool
from .reasoning import Evaluation, Stage
from .retrieval import CorpusIndex, PASSAGE_LIMIT, digest
from .functional_modules import grounded_evidence
from .finance_causal import quotes_from_rows, replay_policy, imbalance_policy

MAX_ROWS = 128
MAX_CENTS = 10**12


def integer(value, low, high, name):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"Invalid {name}")
    return value


def validate_prices(prices):
    if not isinstance(prices, list) or not 2 <= len(prices) <= MAX_ROWS:
        raise ValueError("Expected 2–128 positive integer-cent observations")
    return [integer(x, 1, MAX_CENTS, "price") for x in prices]


def replay(prices, orders, initial_cents, fee_cents):
    """Replay a proposed sequence, not a strategy generator.

    Each action is a whole-unit buy/sell at the supplied scenario price.
    Final inventory is liquidated with the same fee. No leverage, deposits,
    hidden fills or fabricated returns. This is a simplified test instrument,
    not a market execution model.
    """
    prices = validate_prices(prices)
    cash = integer(initial_cents, 1, MAX_CENTS, "initial cash")
    fee = integer(fee_cents, 0, MAX_CENTS, "fee")
    if not isinstance(orders, list) or len(orders) > len(prices):
        raise ValueError("Expected a bounded order list")
    units = 0
    last = -1
    ledger = []
    for order in orders:
        if not isinstance(order, dict) or set(order) != {"index", "side", "units"}:
            raise ValueError("Order requires index, side, units")
        index = integer(order["index"], 0, len(prices)-1, "index")
        quantity = integer(order["units"], 1, MAX_CENTS, "units")
        side = order["side"]
        if index <= last:
            raise ValueError("Orders must follow strictly increasing observation indices")
        value = prices[index]*quantity
        if side == "buy":
            if cash < value+fee:
                raise ValueError("Insufficient simulated cash")
            cash -= value+fee
            units += quantity
        elif side == "sell":
            if units < quantity or cash+value < fee:
                raise ValueError("Insufficient simulated inventory or fee balance")
            cash += value-fee
            units -= quantity
        else:
            raise ValueError("Unknown side")
        last = index
        ledger.append(dict(index=index,side=side,units=quantity,price_cents=prices[index],
                           fee_cents=fee,cash_cents=cash,inventory=units))
    if units:
        value = units*prices[-1]
        if cash+value < fee:
            raise ValueError("Cannot settle final inventory and fee")
        cash += value-fee
        ledger.append(dict(index=len(prices)-1,side="settle",units=units,price_cents=prices[-1],
                           fee_cents=fee,cash_cents=cash,inventory=0))
    return dict(simulated=True,initial_cents=initial_cents,final_cents=cash,
                net_cents=cash-initial_cents,ledger=ledger)


def strict_json(text):
    def pairs(items):
        result = {}
        for key,value in items:
            if key in result:
                raise ValueError("Duplicate JSON member")
            result[key] = value
        return result
    return json.loads(text, object_pairs_hook=pairs)


class Finance(BasicModule):
    """Explicitly registered development module; not in default_modules."""
    manifest = Manifest("finance", memory_mib=4,
                        tools=(Tool("scenario", (Stage.FORMULATE,)),
                               Tool("replay", (Stage.EVALUATE,)),
                               Tool("search_library", (Stage.FORMULATE,)),
                               Tool("causal_replay", (Stage.EVALUATE,))),
                        operations=("scenario_replay", "library_research", "causal_replay"))
    procedures = {
        Stage.FORMULATE: "Review the owner objective, cash, fees and supplied simulated price scenario.",
        Stage.CONSTRUCT: "Return JSON only: an array of orders with index, side (buy or sell), and units.",
        Stage.EVALUATE: "Replay the proposed orders and measure final simulated settled cash.",
        Stage.CORRECT: "Revise the JSON order array using the measured replay feedback.",
    }

    def __init__(self, workspace):
        super().__init__("finance")
        self.workspace = Path(workspace).resolve()

    def procedure(self, operation, stage):
        if operation == "causal_replay":
            return {Stage.FORMULATE: "Design a chronological stock strategy using the owner objective and available features; evaluation quotes are withheld.",
                    Stage.CONSTRUCT: "Return JSON with threshold_bps (integer 0 to 10000) for the experimental imbalance baseline.",
                    Stage.EVALUATE: "Evaluate the proposed threshold on withheld chronological quotes at next-quote prices.",
                    Stage.CORRECT: "Revise the threshold using the reported experimental feedback."}[stage]
        if operation == "library_research":
            return {
                Stage.FORMULATE: "Investigate the objective using the retrieved source passages.",
                Stage.CONSTRUCT: "Use Premises: then - [n] exact quote, then Inference: and your own reasoning.",
                Stage.EVALUATE: "Check quoted premises against retrieved source text.",
                Stage.CORRECT: "Correct the response using the observed citation feedback.",
            }[stage]
        return super().procedure(operation, stage)

    def search_library(self, name, query, limit):
        relative = Path(name)
        path = (self.workspace / relative).resolve()
        if (relative.is_absolute() or ".." in relative.parts or
                not path.is_relative_to(self.workspace) or not path.is_file()):
            raise ValueError("Invalid owner-selected library index")
        with CorpusIndex(path) as index:
            return index.search(query, limit)

    def scenario(self, name):
        relative = Path(name)
        path = (self.workspace/relative).resolve()
        if (relative.is_absolute() or ".." in relative.parts or
                not path.is_relative_to(self.workspace) or path.suffix != ".json" or not path.is_file()):
            raise ValueError("Invalid owner-selected scenario path")
        with path.open("rb") as source:
            raw = source.read(16385)
        if len(raw)>16384:
            raise ValueError("Scenario exceeds 16 KiB")
        data = strict_json(raw.decode("utf-8"))
        if not isinstance(data, dict) or set(data)!={"prices_cents"}:
            raise ValueError("Scenario requires prices_cents only")
        return dict(source=name,sha256=hashlib.sha256(raw).hexdigest(),
                    prices_cents=validate_prices(data["prices_cents"]))

    def validate_inputs(self, operation, inputs):
        if operation == "causal_replay":
            if set(inputs) != {"quotes", "initial_cents", "target_cents", "fee_bps"} or not isinstance(inputs["quotes"], str):
                raise ValueError("Expected quotes file, starting cash, target and fee")
            integer(inputs["initial_cents"], 1, MAX_CENTS, "initial cash")
            integer(inputs["target_cents"], 1, MAX_CENTS, "target cash")
            integer(inputs["fee_bps"], 0, 9999, "fee bps")
            return
        if operation == "library_research":
            if set(inputs) != {"index", "limit"} or not isinstance(inputs["index"], str) or not inputs["index"].strip():
                raise ValueError("Expected local library index and passage limit")
            integer(inputs["limit"], 1, 12, "passage limit")
            return
        if operation!="scenario_replay" or set(inputs)!={"scenario","initial_cents","target_cents","fee_cents"}:
            raise ValueError("Expected scenario, initial_cents, target_cents, fee_cents")
        if not isinstance(inputs["scenario"],str) or not inputs["scenario"].strip():
            raise ValueError("Expected scenario path")
        integer(inputs["initial_cents"],1,MAX_CENTS,"initial cash")
        integer(inputs["target_cents"],1,MAX_CENTS,"target cash")
        integer(inputs["fee_cents"],0,MAX_CENTS,"fee")

    def causal_replay(self, name, threshold_bps, initial_cents, fee_bps):
        from fractions import Fraction
        integer(threshold_bps, 0, 10000, "threshold bps")
        relative = Path(name)
        path = (self.workspace / relative).resolve()
        if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(self.workspace) or path.suffix != ".json":
            raise ValueError("Expected workspace quote JSON")
        raw = path.read_bytes()
        rows = strict_json(raw.decode("utf-8"))
        quotes_from_rows(rows)
        result = replay_policy(rows, imbalance_policy(Fraction(threshold_bps, 10000)), initial_cents, fee_bps)
        result["source_sha256"] = hashlib.sha256(raw).hexdigest()
        return result

    def tool_handlers(self):
        return {"scenario":self.scenario,"replay":replay,"search_library":self.search_library,
                "causal_replay":self.causal_replay}

    def prepare(self, run, tools, prior):
        o=run["inputs"]
        if run["operation"] == "causal_replay":
            return {"initial_cents":o["initial_cents"],"target_cents":o["target_cents"],"fee_bps":o["fee_bps"],"features":["bid","ask","bid_size","ask_size","imbalance","order_flow"],"execution":"next quote; simulated"}
        if run["operation"] == "library_research":
            return {"index": o["index"], "evidence": tools.call(
                "search_library", name=o["index"], query=run["objective"], limit=o["limit"])}
        return dict(simulated=True,initial_cents=o["initial_cents"],target_cents=o["target_cents"],
                    fee_cents=o["fee_cents"],scenario=tools.call("scenario",name=o["scenario"]))

    def validate_context(self, operation, inputs, context):
        if not context:
            return
        if operation == "causal_replay":
            if any(context.get(k)!=inputs[k] for k in ("initial_cents","target_cents","fee_bps")):
                raise ValueError("Owner inputs changed")
            return
        if operation == "library_research":
            if set(context) != {"index", "evidence"} or context["index"] != inputs["index"]:
                raise ValueError("Library source changed")
            evidence = context["evidence"]
            if not isinstance(evidence, list) or len(evidence) > inputs["limit"]:
                raise ValueError("Invalid library evidence")
            for source in evidence:
                if (not isinstance(source, dict) or set(source) != {"source", "locator", "text", "sha256"}
                        or not all(isinstance(v, str) and v for v in source.values())
                        or len(source["text"]) > PASSAGE_LIMIT or digest(source["text"]) != source["sha256"]):
                    raise ValueError("Invalid library passage")
            return
        if set(context)!={"simulated","initial_cents","target_cents","fee_cents","scenario"} or context["simulated"] is not True:
            raise ValueError("Invalid replay context")
        for key in ("initial_cents","target_cents","fee_cents"):
            if type(context[key]) is not int or context[key]!=inputs[key]:
                raise ValueError("Owner parameters changed")
        data=context["scenario"]
        if (not isinstance(data,dict) or set(data)!={"source","sha256","prices_cents"} or
            data["source"]!=inputs["scenario"] or not isinstance(data["sha256"],str) or
            len(data["sha256"])!=64 or any(c not in "0123456789abcdef" for c in data["sha256"])):
            raise ValueError("Invalid source evidence")
        validate_prices(data["prices_cents"])

    def evaluate(self, run, tools):
        context=run["context"]
        self.validate_context(run["operation"],run["inputs"],context)
        if run["operation"] == "causal_replay":
            from decimal import Decimal
            try:
                proposal=strict_json(run["candidate"])
                if not isinstance(proposal,dict) or set(proposal)!={"threshold_bps"}:
                    raise ValueError("Expected threshold_bps")
                result=tools.call("causal_replay",name=run["inputs"]["quotes"],threshold_bps=proposal["threshold_bps"],initial_cents=context["initial_cents"],fee_bps=context["fee_bps"])
            except (ValueError,TypeError,OSError) as exc:
                return Evaluation(False,str(exc))
            return Evaluation(Decimal(result["cash_cents"])>=context["target_cents"],json.dumps(result,sort_keys=True))
        if run["operation"] == "library_research":
            if not context["evidence"]:
                return Evaluation(False, "No matching library passages")
            verified, _, error = grounded_evidence(run["candidate"], context["evidence"])
            return Evaluation(not error, error or f"Verified {len(verified)} cited premises; substantive inference requires independent review")
        try:
            orders=strict_json(run["candidate"])
            result=tools.call("replay",prices=context["scenario"]["prices_cents"],orders=orders,
                              initial_cents=context["initial_cents"],fee_cents=context["fee_cents"])
        except (ValueError,TypeError,RecursionError) as exc:
            return Evaluation(False,"Simulated replay rejected: "+str(exc))
        result["target_cents"]=context["target_cents"]
        result["source_sha256"]=context["scenario"]["sha256"]
        # Complete means this replay met its supplied target, not live competency.
        return Evaluation(result["final_cents"]>=context["target_cents"],
                          json.dumps(result,sort_keys=True,separators=(",",":")))


def build_module(workspace):
    return Finance(workspace)
