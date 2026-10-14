"""Where is the edge? Uses live recordings.

1. Logistic regression of outcome on logit(model) and logit(book mid) per
   time bucket -> optimal model_weight (share of model in the blend).
2. Calibration of the *ask* price of each side: realized win rate vs price
   paid, by price bucket and time bucket (favorite-longshot bias).
"""
import sys, math
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "yeno-bot"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import FairValue
from recorded import load_rows, build

def lg(p): p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))

def fit_logit(X, y, iters=50, l2=1e-3):
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        z = X @ w; p = 1 / (1 + np.exp(-z))
        g = X.T @ (p - y) + l2 * w
        H = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(len(w))
        w -= np.linalg.solve(H, g)
    return w

src = sys.argv[1] if len(sys.argv) > 1 else "coinbase"
ev, mk = build(load_rows(), ref_src=src)
fv = FairValue(); rec = []; last = {}
for e in ev:
    if e[0] == "btc": fv.observe_mid(e[1], e[2]); continue
    if e[0] != "book" or e[3] != "YES": continue
    t, slug = e[1], e[2]; m = mk[slug]
    fv.learn_basis(m["start"], m["open"])
    if not fv.basis_known: continue
    if t - last.get(slug, -9) < 2.0: continue  # thin to 1 row / 2 s
    last[slug] = t
    yb = e[4][0][0] if e[4] else None; ya = e[5][0][0] if e[5] else None
    p, _, _ = fv.prob_yes(t, m["end"], m["open"])
    y = 1.0 if m["close"] >= m["open"] else 0.0
    rec.append((m["end"] - t, p, yb, ya, y, slug))
print("markets", len(mk), "rows", len(rec))
S = np.array([r[0] for r in rec]); P = np.array([r[1] for r in rec]); Y = np.array([r[4] for r in rec])
YB = np.array([r[2] if r[2] is not None else np.nan for r in rec]); YA = np.array([r[3] if r[3] is not None else np.nan for r in rec])
print("\n[1] outcome ~ a*logit(model) + b*logit(book mid)")
for lo, hi in [(150, 300), (90, 150), (45, 90), (15, 45)]:
    k = (S >= lo) & (S < hi) & ~np.isnan(YB) & ~np.isnan(YA)
    if k.sum() < 50 or len(set(Y[k])) < 2: continue
    mid = (YB[k] + YA[k]) / 2
    X = np.column_stack([lg(P[k]), lg(mid), np.ones(k.sum())])
    w = fit_logit(X, Y[k])
    nm = len({rec[i][5] for i in np.where(k)[0]})
    print(f"S {lo}-{hi}: n={k.sum()} markets={nm} a(model)={w[0]:+.2f} b(book)={w[1]:+.2f} -> model share={w[0]/(w[0]+w[1]+1e-9):.2f}")
print("\n[2] price paid (ask) vs realized win rate, both sides pooled")
rows = []
for s, p, yb, ya, y, slug in rec:
    if ya is not None: rows.append((s, ya, y, p))
    if yb is not None: rows.append((s, 1 - yb, 1 - y, 1 - p))  # NO ask = 1 - YES bid
R = np.array(rows)
for lo, hi in [(0.5, 0.7), (0.7, 0.85), (0.85, 0.93), (0.93, 0.97), (0.97, 0.995)]:
    for slo, shi in [(90, 300), (20, 90)]:
        k = (R[:, 1] >= lo) & (R[:, 1] < hi) & (R[:, 0] >= slo) & (R[:, 0] < shi)
        if k.sum() < 30: continue
        print(f"ask {lo:.2f}-{hi:.2f} S {slo}-{shi}: n={k.sum():5d} avg ask={R[k,1].mean():.3f} win={R[k,2].mean():.3f} model={R[k,3].mean():.3f}")
