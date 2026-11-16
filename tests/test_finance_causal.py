from fractions import Fraction
from my_own_ai.finance_causal import replay_policy, imbalance_policy


def quote(t,b,a,bs=100,as_=100):
    return dict(time_ns=t,bid=b,ask=a,bid_size=bs,ask_size=as_)


def test_no_future_observations():
    rows=[quote(1,99,100),quote(2,199,200),quote(3,299,300)]
    observed=[]
    def decide(history,cash,shares):
        observed.append([q["time_ns"] for q in history])
        return "buy" if len(history)==1 else "sell"
    result=replay_policy(rows,decide)
    assert observed==[[1],[1,2]]
    assert result["cash_cents"]=="149.5"
    assert [x["price_cents"] for x in result["ledger"]]==[200,299]


def test_no_fabricated_liquidity():
    rows=[quote(1,99,100,10,1),quote(2,100,101,10,0),quote(3,200,201)]
    result=replay_policy(rows,imbalance_policy(Fraction(1,2)))
    assert result["cash_cents"]=="100"
    assert result["ledger"]==[]


def test_unliquidated_inventory_not_cash():
    rows=[quote(1,99,100,10,1),quote(2,199,200)]
    result=replay_policy(rows,imbalance_policy(Fraction(1,2)))
    assert result["cash_cents"]=="0"
    assert result["remaining_shares"]=="0.5"
