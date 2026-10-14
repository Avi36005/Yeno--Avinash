"""Random-search strategy parameters.

Modes:
  synth  : full 24 h synthetic sessions across several market-maker regimes;
           objective = qualification rate, then median cash at crossing.
  replay : live recordings (a few hours); objective = projected cash at the
           1000 USD crossing from realized PnL per volume and volume rate.

python3 tune.py synth --trials 40 --runs 3
python3 tune.py replay --trials 200
Results are appended to results/tune-<mode>.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "yeno-bot"))

SPACE = {
    "entry_max_s": [90, 120, 150, 180, 240],
    "entry_min_s": [16, 20, 25, 30, 40],
    "edge_base": [0.0, 0.004, 0.008, 0.012, 0.02, 0.03],
    "model_weight": [0.3, 0.5, 0.7, 0.85, 1.0],
    "min_entry_price": [0.3, 0.5, 0.7, 0.8],
    "max_entry_price": [0.93, 0.96, 0.98, 0.99],
    "hold_margin": [0.0, 0.003, 0.006, 0.01, 0.02],
    "latency_cushion": [0.0, 0.002, 0.004, 0.008],
    "forced_exit_s": [4, 6, 9, 12, 20],
    "pace_gain": [0.0, 0.05, 0.1, 0.2],
    "max_cycles_per_market": [1, 2, 4, 8],
}

REGIMES = [
    {"mm_lag": 0.5, "mm_noise": 0.05},
    {"mm_lag": 1.0, "mm_noise": 0.06},
    {"mm_lag": 0.3, "mm_noise": 0.08},
]


def sample(rng: random.Random) -> dict:
    return {k: rng.choice(v) for k, v in SPACE.items()}


def eval_synth(params: dict, runs: int, hours: float) -> dict:
    from run_synth import run_many, summarize
    out = []
    for i, reg in enumerate(REGIMES):
        res = run_many("ours", params, reg, {}, runs=runs, hours=hours, seed=100 + i, workers=10)
        out.extend(res)
    s = summarize(out)
    q = s["qualified"] / s["runs"]
    cash = statistics.median([r["cash"] if r["qualified"] else 0.0 for r in out])
    return {"score": q * 100 + cash, **s}


_ROWS = None


def _init_rows():
    global _ROWS
    from recorded import load_rows
    _ROWS = load_rows()


def eval_replay_one(params: dict) -> dict:
    from run_replay import replay
    r, mk = replay("ours", params, "coinbase", None, {"min_order_override": 5.0}, rows=_ROWS)
    hours = max(r.hours, 1e-6)
    rate = r.volume / hours
    pnl = r.final_cash - 10.0
    per_vol = pnl / r.volume if r.volume else 0.0
    feasible = rate * 24 >= 1000 * 1.15
    proj = 10.0 + per_vol * 1000 if r.volume >= 50 else 0.0
    score = proj if feasible else proj * min(1.0, rate * 24 / 1150) - 5
    return {"score": round(score, 3), "volume": round(r.volume, 2), "rate_per_h": round(rate, 1),
            "pnl": round(pnl, 3), "per100": round(100 * per_vol, 3), "cycles": r.cycles,
            "settled": r.settled_positions, "markets": len(mk)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["synth", "replay"])
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--include-default", action="store_true")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    (HERE / "results").mkdir(exist_ok=True)
    path = HERE / "results" / f"tune-{a.mode}.jsonl"
    trials = [sample(rng) for _ in range(a.trials)]
    if a.include_default:
        trials.insert(0, {})
    if a.mode == "synth":
        for params in trials:
            r = eval_synth(params, a.runs, a.hours)
            row = {"params": params, **r}
            print(json.dumps(row), flush=True)
            with path.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
    else:
        with ProcessPoolExecutor(max_workers=10, initializer=_init_rows) as pool:
            for params, r in zip(trials, pool.map(eval_replay_one, trials)):
                row = {"params": params, **r}
                with path.open("a") as fh:
                    fh.write(json.dumps(row) + "\n")
        rows = [json.loads(l) for l in path.open()]
        rows.sort(key=lambda r: -r["score"])
        for r in rows[:10]:
            print(json.dumps(r))
