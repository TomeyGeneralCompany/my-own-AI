"""Compare deterministic frozen-exam evidence without using it for training."""
def retention(previous, current, historical_best=0):
    def cases(report):
        raw = (report or {}).get("raw_results", {})
        return {x["id"]: x for x in raw.get("cases", []) if x.get("exact_match") is not None}
    old, new = cases(previous), cases(current)
    if not new:
        raise ValueError("Missing auto-scored comprehension exam cases")
    correct = {k for k, v in new.items() if v["exact_match"] is True}
    shared = set(old) & set(new)
    prior = {k for k in shared if old[k]["exact_match"] is True}
    retained = prior & correct
    gained = {k for k in shared if k not in prior and k in correct}
    lost = prior - correct
    categories = {}
    for group in sorted({v["category"] for v in new.values()}):
        ids = {k for k, v in new.items() if v["category"] == group}
        categories[group] = {
            "correct": len(correct & ids), "total": len(ids),
            "retained": len(retained & ids), "lost": len(lost & ids),
            "gained": len(gained & ids),
        }
    return {
        "correct": len(correct), "total": len(new),
        "prior_comparable": len(shared), "previous_correct": len(prior),
        "best_observed_correct": max(int(historical_best), len(prior), len(correct)),
        "retained": len(retained), "lost": len(lost), "gained": len(gained),
        "retained_ids": sorted(retained), "lost_ids": sorted(lost),
        "gained_ids": sorted(gained), "categories": categories,
    }
