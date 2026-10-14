"""Replay live recordings through the paper evaluator.

python3 run_replay.py [--bot ours|house] [--src coinbase] [--http URL] [--params JSON]
                      [--adverse N] [--mos 5] [--log]

Recordings are only a few hours long, so this reports volume and cash
trajectory (cash per $100 volume) rather than a 24 h qualification.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluator import Config, Evaluator  # noqa: E402
from recorded import build, load_rows  # noqa: E402
from run_synth import make_bot  # noqa: E402


def http_decider(url: str):
    def decide(obs):
        req = urllib.request.Request(url, data=json.dumps(obs).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=0.2) as r:
            return json.loads(r.read())
    return decide


def replay(kind="ours", params=None, src="coinbase", http=None, ecfg_kw=None, rows=None, keep_log=False,
           target=1e9):
    rows = rows if rows is not None else load_rows()
    events, markets = build(rows, ref_src=src)
    decide = http_decider(http) if http else make_bot(kind, params)
    cfg = Config(target=target, **(ecfg_kw or {}))
    ev = Evaluator(decide, markets, cfg, keep_log=keep_log)
    r = ev.run(events)
    return r, markets


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bot", default="ours")
    ap.add_argument("--src", default="coinbase")
    ap.add_argument("--http", default=None)
    ap.add_argument("--params", default=None)
    ap.add_argument("--adverse", type=int, default=0)
    ap.add_argument("--mos", type=float, default=5.0)
    ap.add_argument("--log", action="store_true")
    a = ap.parse_args()
    ecfg = {"adverse_ticks": a.adverse, "min_order_override": a.mos, "enforce_timeout": bool(a.http)}
    r, mk = replay(a.bot, json.loads(a.params) if a.params else None, a.src, a.http, ecfg, keep_log=a.log)
    pnl = r.final_cash - 10.0
    print(json.dumps({"markets": len(mk), "hours": round(r.hours, 2), "volume": round(r.volume, 2),
                      "cycles": r.cycles, "final_cash": round(r.final_cash, 4),
                      "pnl_per_100_volume": round(100 * pnl / r.volume, 3) if r.volume else None,
                      "settled": r.settled_positions, "rejected": r.rejected, "decisions": r.decisions,
                      "slow_replies": r.slow_replies, "flat": r.flat_at_end}))
    if a.log:
        for t, what, kw in r.log:
            print(round(t, 2), what, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in kw.items()})
