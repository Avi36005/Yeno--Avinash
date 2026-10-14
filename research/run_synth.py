"""Run the bot (in-process) through synthetic 24 h sessions in parallel.

python3 run_synth.py --bot ours|house --lag 1.0 --noise 0.1 --runs 8 --hours 24
Prints per-run results and a summary; --json writes results.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "yeno-bot"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluator import Config, Evaluator  # noqa: E402
from synth import SynthCfg, generate, windows  # noqa: E402


def make_bot(kind: str, params: dict | None):
    if kind == "house":
        spec = importlib.util.spec_from_file_location("house", ROOT / "starter-kit" / "yeno-house-bot-v1.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.HouseBot().decide
    from strategy import Strategy
    return Strategy(params).decide


def one(args):
    start, scfg, ecfg, kind, params = args
    events, markets = generate(start, scfg)
    ev = Evaluator(make_bot(kind, params), markets, ecfg)
    r = ev.run(events)
    return {"start": start, "qualified": r.qualified, "cash": r.cash_at_cross, "final_cash": r.final_cash,
            "volume": round(r.volume, 2), "cycles": r.cycles, "hours": r.time_to_target_h,
            "dd": round(r.max_drawdown, 3), "settled": r.settled_positions, "rejected": r.rejected,
            "flat": r.flat_at_end}


def run_many(kind="ours", params=None, scfg_kw=None, ecfg_kw=None, runs=8, hours=24.0, seed=0, workers=8,
             starts=None):
    ws = windows(hours)
    if not ws:
        raise SystemExit("no full window in history")
    if starts is None:
        step = max(1, len(ws) // runs)
        starts = ws[::step][:runs]
    jobs = []
    for i, s in enumerate(starts):
        scfg = SynthCfg(hours=hours, seed=seed + i, **(scfg_kw or {}))
        ecfg = Config(**(ecfg_kw or {}))
        jobs.append((s, scfg, ecfg, kind, params))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, jobs))


def summarize(res):
    q = [r for r in res if r["qualified"]]
    cash = [r["cash"] for r in q]
    vols = [r["volume"] for r in res]
    return {
        "runs": len(res), "qualified": len(q),
        "median_cash_q": round(statistics.median(cash), 3) if cash else None,
        "min_cash_q": round(min(cash), 3) if cash else None,
        "median_hours_q": round(statistics.median([r["hours"] for r in q]), 2) if q else None,
        "median_volume": round(statistics.median(vols), 1),
        "median_final_cash": round(statistics.median([r["final_cash"] for r in res]), 3),
        "settled": sum(r["settled"] for r in res),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bot", default="ours")
    ap.add_argument("--lag", type=float, default=1.0)
    ap.add_argument("--noise", type=float, default=0.10)
    ap.add_argument("--offset", type=float, default=0.0)
    ap.add_argument("--adverse", type=int, default=0)
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--params", default=None, help="JSON dict of strategy overrides")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    res = run_many(a.bot, json.loads(a.params) if a.params else None,
                   {"mm_lag": a.lag, "mm_noise": a.noise, "ref_offset": a.offset},
                   {"adverse_ticks": a.adverse}, a.runs, a.hours, a.seed)
    for r in res:
        print(r)
    print(summarize(res))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))
