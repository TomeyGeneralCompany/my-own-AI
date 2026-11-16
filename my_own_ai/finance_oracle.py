"""Offline hindsight benchmark, never a predictive policy or live execution engine."""
from fractions import Fraction
from decimal import Decimal, localcontext


def _integer(value, low, high, label):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"Invalid {label}")
    return value


def hindsight_oracle(quotes, initial_cents=100, fee_bps=0):
    """Maximize terminal cash in a single-asset fractional long/cash model.

    Quotes are chronological (bid, ask) pairs in integer price units.
    Assumes unlimited fill capacity, instant settlement, no minimum lot,
    no leverage, no shorting, and proportional fees on both sides.
    At each observation the optimal cash and inventory states are retained.
    Exact rational arithmetic; O(n) state transitions, not O(n) bit complexity.
    """
    initial = _integer(initial_cents, 1, 10**12, "initial cents")
    fee = Fraction(_integer(fee_bps, 0, 9999, "fee bps"), 10000)
    if not isinstance(quotes, (list, tuple)) or not 1 <= len(quotes) <= 10000:
        raise ValueError("Expected 1–10000 chronological quotes")
    cash, shares = Fraction(initial), Fraction(0)
    cash_path, share_path = None, None
    for i, quote in enumerate(quotes):
        if not isinstance(quote, (list, tuple)) or len(quote) != 2:
            raise ValueError("Expected bid and ask")
        bid = _integer(quote[0], 1, 10**12, "bid")
        ask = _integer(quote[1], bid, 10**12, "ask")
        # Both transitions use the preceding states, not same-tick proceeds.
        sale = shares * bid * (1-fee)
        buy = cash / (ask * (1+fee))
        next_cash, next_shares = cash, shares
        next_cash_path, next_share_path = cash_path, share_path
        if sale > cash:
            next_cash, next_cash_path = sale, (i, "sell", share_path)
        if buy > shares:
            next_shares, next_share_path = buy, (i, "buy", cash_path)
        cash, shares = next_cash, next_shares
        cash_path, share_path = next_cash_path, next_share_path
    actions = []
    while cash_path is not None:
        i, side, cash_path = cash_path
        actions.append({"index":i,"side":side})
    actions.reverse()
    return {"hindsight_only":True,"initial_cents":Fraction(initial),
            "final_cents":cash,"multiple":cash/initial,"actions":actions,
            "fee_bps":fee_bps,"observations":len(quotes)}


def decimal_text(value, precision=24):
    """Display a rational without converting large integers to binary floats."""
    with localcontext() as ctx:
        ctx.prec=precision
        return str(Decimal(value.numerator)/Decimal(value.denominator))


def benchmark(quotes, initial_cents=100, fee_bps=0, target_cents=10000):
    """Report ideal and quote/fee-adjusted bounds separately.

    This does not expose future observations to a live or causal policy.
    The ideal series uses quote midpoints in doubled price units (ratios
    are invariant to that unit conversion).
    """
    target=_integer(target_cents,1,10**12,"target cents")
    adjusted=hindsight_oracle(quotes,initial_cents,fee_bps)
    ideal=hindsight_oracle([(b+a,b+a) for b,a in quotes],initial_cents,0)
    def report(result):
        return {**result,
                "initial_cents":decimal_text(result["initial_cents"]),
                "final_cents":decimal_text(result["final_cents"]),
                "multiple":decimal_text(result["multiple"]),
                "target_reached":result["final_cents"]>=target}
    return {"target_cents":target,"ideal":report(ideal),"quote_fee_adjusted":report(adjusted),
            "live_performance_verified":False}
