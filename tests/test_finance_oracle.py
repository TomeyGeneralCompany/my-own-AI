from fractions import Fraction
from itertools import product

import pytest

from my_own_ai.finance_oracle import benchmark, hindsight_oracle


def brute_force(quotes, initial, fee_bps):
    fee=Fraction(fee_bps,10000)
    best=Fraction(initial)
    for choices in product(("hold","buy","sell"),repeat=len(quotes)):
        cash,shares=Fraction(initial),Fraction(0)
        for side,(bid,ask) in zip(choices,quotes):
            if side=="buy" and cash:
                shares=cash/(ask*(1+fee))
                cash=Fraction(0)
            elif side=="sell" and shares:
                cash=shares*bid*(1-fee)
                shares=Fraction(0)
        if not shares:
            best=max(best,cash)
    return best


@pytest.mark.parametrize("fee",[0,25,100,1000])
def test_exact_optimum_matches_exhaustive_enumeration(fee):
    for sequence in product((10,12,15),repeat=4):
        quotes=[(p,p+1) for p in sequence]
        result=hindsight_oracle(quotes,100,fee)
        assert result["final_cents"]==brute_force(quotes,100,fee)
        cash,shares=Fraction(100),Fraction(0)
        for action in result["actions"]:
            bid,ask=quotes[action["index"]]
            if action["side"]=="buy":
                assert cash>0 and shares==0
                shares,cash=cash/(ask*(1+Fraction(fee,10000))),Fraction(0)
            else:
                assert shares>0 and cash==0
                cash,shares=shares*bid*(1-Fraction(fee,10000)),Fraction(0)
        assert cash==result["final_cents"] and shares==0


def test_synthetic_100x_is_measured_not_assumed():
    result=benchmark([(1,1),(10,10),(1,1),(10,10)])
    assert result["ideal"]["final_cents"]=="10000"
    assert result["ideal"]["target_reached"] is True
    assert result["live_performance_verified"] is False
    assert result["quote_fee_adjusted"]["target_reached"] is True


def test_spread_and_fees_reduce_available_return():
    result=benchmark([(100,102),(101,103),(100,102),(101,103)],fee_bps=10)
    assert result["ideal"]["multiple"]!="1"
    assert result["quote_fee_adjusted"]["multiple"]=="1"
    assert not result["quote_fee_adjusted"]["target_reached"]


@pytest.mark.parametrize("quotes",[[],[(2,1)],[(True,1)],[(0,1)],[(1,)]])
def test_reject_invalid_quotes(quotes):
    with pytest.raises(ValueError):
        hindsight_oracle(quotes)
