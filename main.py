"""GTA VI trophy rarity forecast: model earners and players-tracked separately, then rate = earners / tracked.

Exophase refreshes on its own schedule; a refresh grows the denominator (players tracked) so rates drop. Extrapolating
the rate itself misses that, so each distinct refreshed page is turned into earners = rate * players / 100, and
ln(earners) (per trophy) and ln(players) (shared) are each fitted against ln(1 + hours since launch day) and read at
the target. The quantile spread is a residual term plus an extrapolation term that grows with how far past the last
refresh the target is. With fewer than 3 distinct refreshed pages there is no growth to fit: a prior says the next
refresh pushes rates down, with a wide spread.
"""

import json
import math
import os
from datetime import datetime
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
T0 = datetime.fromisoformat("2026-11-19T00:00:00+00:00")
K = 8  # most recent refreshed pages used in a fit
SLOPE_DAMP = 0.85  # growth slows as the launch wave passes: shrink the fitted slope past the last page
FLOOR_S = 0.35  # minimum logit spread (synthetic backtest is idealised, so not tuned tighter)
EXTRA_S = 0.30  # logit spread per unit of ln((1+h_target)/(1+h_last))
RESID_MULT = 2.0
# with <3 pages: median logit drop for the next refresh by tier (new players have done little, so the hardest trophies
# fall furthest; guesses, no refreshed data to fit), and its spread
PRIOR_SHIFT = {"platinum": -1.6, "gold": -1.3, "silver": -1.15, "bronze": -1.0}
PRIOR_S = 1.3


def hours(iso: str) -> float:
    return (datetime.fromisoformat(iso.replace("Z", "+00:00")) - T0).total_seconds() / 3600


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp(pct)
    return math.log(p / (100 - p))


def expit(v: float) -> float:
    return 100 / (1 + math.exp(-v))


def ols(xs, ys):
    """slope, intercept, residual sd (ys on xs); flat line through the mean when xs do not vary."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx > 1e-12 else 0.0
    b = max(b, 0.0)  # counts never fall
    a = my - b * mx
    res = math.sqrt(sum((y - a - b * x) ** 2 for x, y in zip(xs, ys)) / max(n - 2, 1))
    return b, a, res


def pages(case):
    """Distinct refreshed pages, oldest first, each stamped with the first read that showed it."""
    out, seen = [], None
    for h in case["history"]:
        key = (h["players_tracked"], tuple(h["rates"]))
        if key != seen:
            out.append(h)
            seen = key
    return out


def forecast_case(case: dict) -> dict:
    last = case["history"][-1]
    pg = pages(case)[-K:]
    ht = hours(case["target"])
    n_tr = len(case["trophies"])
    rows = []
    if len(pg) < 3:
        for i, t in enumerate(case["trophies"]):
            r = last["rates"][i] if last["rates"][i] is not None else 0.0
            shift = PRIOR_SHIFT[t["tier"]] if r < 99.5 else -0.2
            s = PRIOR_S if r < 99.5 else 0.5
            rows.append((t["id"], [expit(logit(r) + shift + z * s) for z in Z]))
        return rows_out(rows)
    xs = [math.log(1 + max(hours(p["at"]), 0)) for p in pg]
    xt = math.log(1 + ht)
    gap = max(xt - xs[-1], 0.0)
    bn, an, rn = ols(xs, [math.log(max(p["players_tracked"], 1)) for p in pg])
    ln_n_pred = an + bn * xs[-1] + SLOPE_DAMP * bn * gap
    for i, t in enumerate(case["trophies"]):
        pts = [(x, p["rates"][i] * p["players_tracked"] / 100) for x, p in zip(xs, pg) if p["rates"][i] is not None]
        if len(pts) < 3:
            r = last["rates"][i] if last["rates"][i] is not None else 0.0
            rows.append((t["id"], [expit(logit(r) + z * PRIOR_S) for z in Z]))
            continue
        # +0.5 players keeps ln finite for trophies nobody has earned yet
        be, ae, re = ols([p[0] for p in pts], [math.log(p[1] + 0.5) for p in pts])
        ln_e_pred = ae + be * pts[-1][0] + SLOPE_DAMP * be * gap
        rate = 100 * math.exp(ln_e_pred - ln_n_pred)
        rate = min(rate, 100.0)
        s = math.sqrt(FLOOR_S**2 + (RESID_MULT * (re + rn)) ** 2 + (EXTRA_S * gap) ** 2)
        rows.append((t["id"], [expit(logit(rate) + z * s) for z in Z]))
    return rows_out(rows)


def rows_out(rows):
    out = []
    for tid, q in rows:
        q = [round(min(100.0, max(0.0, v)), 4) for v in q]
        q = [max(q[: j + 1]) for j in range(5)]
        out.append({"id": tid, "q": q})
    return {"trophies": out}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        (OUTPUT_DIR / f"{path.stem}.out").write_text(json.dumps(forecast_case(case), separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
