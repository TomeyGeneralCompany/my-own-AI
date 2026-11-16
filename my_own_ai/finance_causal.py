"""Chronological research instrument; no future signal input or broker execution."""
from fractions import Fraction
from .finance_oracle import decimal_text


def quotes_from_rows(rows):
    result=[]
    previous=-1
    for row in rows:
        if set(row)!={"time_ns","bid","ask","bid_size","ask_size"}:
            raise ValueError("Expected timestamp, bid/ask and displayed sizes")
        t,b,a,bs,as_=(row[k] for k in ("time_ns","bid","ask","bid_size","ask_size"))
        if any(type(x) is not int for x in (t,b,a,bs,as_)) or t<=previous or not 0<b<=a or min(bs,as_)<0:
            raise ValueError("Invalid chronological quote")
        previous=t
        result.append((t,b,a,bs,as_))
    if len(result)<2:
        raise ValueError("At least two quotes required")
    return result


def order_flow(previous,current):
    _,b0,a0,bs0,as0=previous
    _,b,a,bs,as_=current
    return ((bs if b>=b0 else 0)-(bs0 if b<=b0 else 0)
            -(as_ if a<=a0 else 0)+(as0 if a>=a0 else 0))


def replay_policy(rows, policy, initial_cents=100, fee_bps=0):
    """Policy sees current and past quote features; executes at next quote.
    
    All prices are integer cents, displayed sizes whole shares.
    Fractional holdings, proportional fees, displayed-depth-limited market
    fills, no leverage, no short inventory, instant simulated settlement.
    The policy receives no future observations or hindsight oracle.
    """
    quotes=quotes_from_rows(rows)
    if type(initial_cents) is not int or initial_cents<=0 or type(fee_bps) is not int or not 0<=fee_bps<10000:
        raise ValueError("Invalid starting cash or fee")
    cash,shares=Fraction(initial_cents),Fraction(0)
    fee=Fraction(fee_bps,10000)
    ledger=[]
    pending="hold"
    history=[]
    for i,q in enumerate(quotes):
        t,b,a,bs,as_=q
        if pending=="buy":
            quantity=min(cash/(a*(1+fee)),Fraction(as_))
            cash-=quantity*a*(1+fee)
            shares+=quantity
        elif pending=="sell":
            quantity=min(shares,Fraction(bs))
            cash+=quantity*b*(1-fee)
            shares-=quantity
        else:
            quantity=Fraction(0)
        if quantity:
            ledger.append({"time_ns":t,"side":pending,"shares":decimal_text(quantity),
                           "price_cents":a if pending=="buy" else b})
        depth=bs+as_
        observation={"time_ns":t,"bid":b,"ask":a,"bid_size":bs,"ask_size":as_,
                     "imbalance":str(Fraction(bs-as_,depth)) if depth else "0",
                     "order_flow":order_flow(quotes[i-1],q) if i else 0}
        history.append(observation)
        if i<len(quotes)-1:
            pending=policy(tuple(dict(x) for x in history),cash,shares)
            if pending not in ("buy","sell","hold"):
                raise ValueError("Invalid strategy action")
    # Mark remaining inventory separately; neither pretend it filled nor
    # count it as settled cash. The final quote depth limits liquidation.
    return {"simulated":True,"initial_cents":initial_cents,
            "cash_cents":decimal_text(cash),"remaining_shares":decimal_text(shares),
            "marked_equity_cents":decimal_text(cash+shares*quotes[-1][1]*(1-fee)),
            "ledger":ledger}


def imbalance_policy(threshold):
    """Explicit baseline, not a required Core strategy. Threshold in [0,1]."""
    threshold=Fraction(threshold)
    if not 0<=threshold<=1:
        raise ValueError("Invalid threshold")
    def decide(history,cash,shares):
        signal=Fraction(history[-1]["imbalance"])
        if signal>threshold and cash>0:
            return "buy"
        if signal<-threshold and shares>0:
            return "sell"
        return "hold"
    return decide
