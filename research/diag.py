"""Trade-level diagnosis of one synthetic run."""
import sys, json, statistics
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "yeno-bot"))
from evaluator import Config, Evaluator
from synth import SynthCfg, generate, windows
from strategy import Strategy
lag, noise, idx, hours = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]) if len(sys.argv)>4 else 24
params = json.loads(sys.argv[5]) if len(sys.argv) > 5 else None
ws = windows(hours); step = max(1, len(ws)//5); start = ws[::step][idx]
events, markets = generate(start, SynthCfg(hours=hours, mm_lag=lag, mm_noise=noise, seed=idx))
bot = Strategy(params)
# wrap decide to record entry context
ctx = {}
def decide(o):
    a = bot.decide(o)
    if a["action"] == "BUY":
        ctx["buy"] = (o["market"]["secondsToClose"], a["outcome"])
    return a
ev = Evaluator(decide, markets, Config(), keep_log=True)
r = ev.run(events)
print({k: getattr(r, k) for k in ["qualified","cash_at_cross","final_cash","volume","cycles","settled_positions","rejected","buys"]})
# reconstruct cycles
cyc = []; cur = None
for t, what, kw in r.log:
    if what == "BUY": cur = {"t": t, "cost": kw["gross"]+kw["fees"], "avg": kw["avg"], "proceeds": 0.0, "sh": kw["shares"]}
    elif what == "SELL" and cur: 
        cur["proceeds"] += kw["gross"]-kw["fees"]; cur["exit"] = kw["avg"]
        if kw["left"] <= 1e-6: cur["pnl"] = cur["proceeds"]-cur["cost"]; cur["hold"] = t-cur["t"]; cyc.append(cur); cur=None
    elif what == "SETTLE" and cur:
        cur["pnl"] = kw["shares"]*(1 if kw["won"] else 0) + cur["proceeds"] - cur["cost"]; cur["settle"]=True; cyc.append(cur); cur=None
pn = [c["pnl"] for c in cyc]
print("cycles", len(cyc), "sum pnl", round(sum(pn),3), "mean", round(statistics.mean(pn),4) if pn else None)
print("wins", sum(p>0 for p in pn), "losses", sum(p<0 for p in pn), "worst", sorted(pn)[:8])
for lo,hi in [(0,0.6),(0.6,0.8),(0.8,0.9),(0.9,0.95),(0.95,1)]:
    s=[c["pnl"] for c in cyc if lo<=c["avg"]<hi]
    if s: print(f"entry px {lo}-{hi}: n={len(s)} sum={sum(s):.3f} mean={statistics.mean(s):.4f}")
# cumulative pnl by hour
h = {}
for c in cyc: h.setdefault(int((c["t"]-ev.t0)//3600), []).append(c["pnl"])
print("by hour:", [(k, len(v), round(sum(v),2)) for k,v in sorted(h.items())])
print("--- trades")
for c in cyc[:40]:
    m = None
    for slug, mm in markets.items():
        if mm["start"] <= c["t"] < mm["end"]: m = mm
    print(f"t+{(c['t']-ev.t0)/60:7.1f}m S={m['end']-c['t']:6.1f} sh={c['sh']:.2f} in={c['avg']:.3f} out={c.get('exit',0):.3f} hold={c.get('hold',0):6.1f} pnl={c['pnl']:+.3f} win={'Y' if m['close']>=m['open'] else 'N'}")
