"""Book-vs-model statistics for recorded data and synthetic data (same code path).

Stats per seconds-to-close bucket:
  dev   = median |model fair - book mid|
  e2    = share of book updates where max(fair-ask, bid-fair) > 0.02
  fwd   = mean 3s-forward change of book mid in the direction of (fair - mid)  [lag signal]
"""
import sys, math
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "yeno-bot"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import FairValue

BUCKETS = [(180, 300), (90, 180), (45, 90), (15, 45)]

def stats(events, markets):
    fv = FairValue()
    rec = []
    for e in events:
        if e[0] == "btc":
            fv.observe_mid(e[1], e[2]); continue
        if e[0] != "book" or e[3] != "YES": continue
        t, slug = e[1], e[2]; m = markets[slug]
        fv.learn_basis(m["start"], m["open"])
        if not fv.basis_known or not e[4] or not e[5]: continue
        p, _, _ = fv.prob_yes(t, m["end"], m["open"])
        rec.append((t, m["end"] - t, p, e[4][0][0], e[5][0][0], slug))
    out = {}
    if not rec: return out
    t = np.array([r[0] for r in rec]); S = np.array([r[1] for r in rec]); p = np.array([r[2] for r in rec])
    bid = np.array([r[3] for r in rec]); ask = np.array([r[4] for r in rec]); mid = (bid + ask) / 2
    slugs = [r[5] for r in rec]
    # forward mid 3 s later within same market
    j = np.searchsorted(t, t + 3.0)
    fwd = np.full(len(t), np.nan)
    for i in range(len(t)):
        if j[i] < len(t) and slugs[j[i]] == slugs[i]:
            fwd[i] = (mid[j[i]] - mid[i]) * np.sign(p[i] - mid[i])
    for lo, hi in BUCKETS:
        k = (S >= lo) & (S < hi)
        if k.sum() < 20: continue
        e = np.maximum(p[k] - ask[k], bid[k] - p[k])
        f = fwd[k]; f = f[~np.isnan(f)]
        big = k & (np.abs(p - mid) > 0.03)
        fb = fwd[big]; fb = fb[~np.isnan(fb)]
        out[f"{lo}-{hi}"] = dict(n=int(k.sum()), dev=round(float(np.median(np.abs(p[k] - mid[k]))), 4),
                                 e2=round(float(np.mean(e > 0.02)), 3),
                                 fwd=round(float(np.mean(f)), 4) if len(f) else None,
                                 fwd_big=round(float(np.mean(fb)), 4) if len(fb) else None, nbig=int(len(fb)))
    return out

if __name__ == "__main__":
    from recorded import load_rows, build
    src = sys.argv[1] if len(sys.argv) > 1 else "coinbase"
    ev, mk = build(load_rows(), ref_src=src)
    print("recorded markets", len(mk))
    for k, v in stats(ev, mk).items(): print("REC", k, v)
