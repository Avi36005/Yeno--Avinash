"""Reliability test of the TWAP fair-value model on historical settlements.

Feeds Binance 1s mids into the bot's FairValue model exactly as the bot would
see them, then checks predicted P(YES) against realized outcomes.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "yeno-bot"))
from model import FairValue  # noqa: E402

btc = {}
for line in open("data/hist-btc1s.jsonl"):
    r = json.loads(line)
    btc[r[0] + 1] = r[4]  # price observed at end of second
mk = {m["start"]: m for m in (json.loads(l) for l in open("data/hist-markets.jsonl"))}
ts = sorted(btc)
checkpoints = [240, 180, 120, 90, 75, 60, 45, 30, 20, 12, 6]
step = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0

fv = FairValue()
rows = []
for t in ts:
    fv.observe_mid(float(t), btc[t])
    start = t - t % 300
    m = mk.get(start)
    if m is None:
        continue
    end = start + 300
    s = end - t
    if s == 299:
        fv.learn_basis(float(start), m["open"])
    if s in checkpoints:
        p, mean, sd = fv.prob_yes(now=float(t), end=float(end), open_target=m["open"])
        if p is None:
            continue
        rows.append((s, p, 1.0 if m["close"] >= m["open"] else 0.0, mean - m["open"], sd))
    if t % 300 == 0:
        pass

rows = np.array(rows)
print("n", len(rows))
for s in checkpoints:
    r = rows[rows[:, 0] == s]
    if not len(r):
        continue
    p = np.clip(r[:, 1], 1e-4, 1 - 1e-4)
    y = r[:, 2]
    ll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    brier = np.mean((p - y) ** 2)
    z = r[:, 3] / r[:, 4]
    print(f"S={s:4d} n={len(r)} logloss={ll:.4f} brier={brier:.4f}")
# reliability buckets on max-side probability
q = np.maximum(rows[:, 1], 1 - rows[:, 1])
win = np.where(rows[:, 1] >= 0.5, rows[:, 2], 1 - rows[:, 2])
print("bucket  n   mean_pred  realized")
for lo, hi in [(0.5, 0.7), (0.7, 0.85), (0.85, 0.93), (0.93, 0.97), (0.97, 0.99), (0.99, 0.997), (0.997, 1.01)]:
    k = (q >= lo) & (q < hi)
    if k.sum():
        print(f"{lo:.3f}-{hi:.3f} {k.sum():5d} {q[k].mean():.4f} {win[k].mean():.4f}")
