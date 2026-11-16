"""Local business instruments; no external commitments or strategy substitution."""
import json

from .modules import BasicModule, Manifest, Tool
from .reasoning import Stage, Evaluation
from .finance_module import integer, strict_json


def calculate_scenario(units, price_cents, variable_cents, fixed_cents):
    values = [integer(n, 0, 10**12, "business input")
              for n in (units, price_cents, variable_cents, fixed_cents)]
    units, price, variable, fixed = values
    revenue = units * price
    costs = units * variable + fixed
    return {"revenue_cents": revenue, "cost_cents": costs,
            "operating_result_cents": revenue-costs,
            "scenario_only": True}


class Business(BasicModule):
    manifest = Manifest("business", memory_mib=4,
                        tools=(Tool("calculate_scenario", (Stage.EVALUATE,)),),
                        operations=("business_plan",))
    procedures = {
        Stage.FORMULATE: "Formulate a plan against the owner objective and supplied business parameters.",
        Stage.CONSTRUCT: "Return a JSON object with plan (text), units (integer), price_cents (integer).",
        Stage.EVALUATE: "Calculate the proposed scenario using the supplied costs and check the owner target.",
        Stage.CORRECT: "Revise the proposed plan and figures using the measured outcome.",
    }

    def __init__(self):
        super().__init__("business")

    def validate_inputs(self, operation, inputs):
        if operation != "business_plan" or set(inputs) != {
            "variable_cents", "fixed_cents", "target_cents", "max_units", "max_price_cents"
        }:
            raise ValueError("Supply variable cost, fixed cost, target and scenario capacity parameters")
        for key, value in inputs.items():
            integer(value, -10**24 if key=="target_cents" else 0,
                    10**24 if key=="target_cents" else 10**12, key)

    def prepare(self, run, tools, prior):
        return {"parameters":run["inputs"],"prior":prior}

    def validate_context(self, operation, inputs, context):
        if context and (set(context)!={"parameters","prior"} or context["parameters"]!=inputs):
            raise ValueError("Owner scenario parameters changed")

    def tool_handlers(self):
        return {"calculate_scenario":calculate_scenario}

    def evaluate(self, run, tools):
        try:
            proposal = strict_json(run["candidate"])
            if not isinstance(proposal,dict) or set(proposal)!={"plan","units","price_cents"}:
                raise ValueError("Expected plan, units and price_cents")
            if not isinstance(proposal["plan"],str) or not proposal["plan"].strip():
                raise ValueError("Missing proposed plan")
            parameters=run["inputs"]
            units=integer(proposal["units"],0,parameters["max_units"],"units")
            price=integer(proposal["price_cents"],0,parameters["max_price_cents"],"price")
            result=tools.call("calculate_scenario",units=units,price_cents=price,
                              variable_cents=parameters["variable_cents"],
                              fixed_cents=parameters["fixed_cents"])
        except (ValueError, TypeError, RecursionError) as exc:
            return Evaluation(False,str(exc))
        result["target_cents"]=parameters["target_cents"]
        return Evaluation(result["operating_result_cents"]>=parameters["target_cents"],
                          json.dumps(result,sort_keys=True))


def build_module(workspace):
    return Business()
