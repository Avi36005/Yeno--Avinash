"""Compare book prices with model fair value on recordings; measure lead-lag."""
import sys, math, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "yeno-bot"))
from model import FairValue
from recorded import load_rows, build
src = sys.argv[1] if len(sys.argv) > 1 else "coinbase"
rows = load_rows()
events, markets = build(rows, ref_src=src)
print("markets", len(markets), "events", len(events), "hours", (events[-1][1]-events[0][1])/3600)
fv = FairValue()
recs = []
last_book = {}
for e in events:
    if e[0] == "btc":
        fv.observe_mid(e[1], e[2]); continue
    if e[0] != "book" or e[3] != "YES": continue
    t, slug = e[1], e[2]; m = markets[slug]
    fv.learn_basis(m["start"], m["open"])
    if not fv.basis_known: continue
    bids, asks = e[4], e[5]
    if not bids or not asks: continue
    p, mean, sd = fv.prob_yes(t, m["end"], m["open"])
    if p is None: continue
    recs.append((t, m["end"]-t, p, bids[0][0], asks[0][0], 1.0 if m["close"]>=m["open"] else 0.0, fv.basis_count))
r = np.array(recs)
print("rows with basis", len(r), "basis", fv.basis)
S, p, bid, ask, y = r[:,1], r[:,2], r[:,3], r[:,4], r[:,5]
mid = (bid+ask)/2
print("spread dist", np.percentile(ask-bid,[10,50,90]))
for lo,hi in [(240,300),(180,240),(120,180),(90,120),(60,90),(30,60),(15,30),(0,15)]:
    k=(S>=lo)&(S<hi)
    if k.sum()<10: continue
    # brier of model vs book mid vs outcome
    bm = np.mean((p[k]-y[k])**2); bb = np.mean((mid[k]-y[k])**2)
    # opportunities: model says buy YES (p - ask > x) or NO (bid - p > x i.e. NO ask = 1-bid)
    edge_yes = p[k]-ask[k]; edge_no = bid[k]-p[k]
    e = np.maximum(edge_yes, edge_no)
    print(f"S {lo:3d}-{hi:3d} n={k.sum():6d} brier model={bm:.4f} book={bb:.4f}  |p-mid| med={np.median(np.abs(p[k]-mid[k])):.3f} edge>0.02: {np.mean(e>0.02):.3f} >0.05: {np.mean(e>0.05):.3f}")
