import json, numpy as np
btc = {}
for l in open('data/hist-btc1s.jsonl'):
    r = json.loads(l); btc[r[0]] = r[4]   # close of second r[0] (price at r[0]+1)
mk = [json.loads(l) for l in open('data/hist-markets.jsonl')]
mk = [m for m in mk if all((m['start']+k) in btc for k in range(-61, 300))]
print("markets with full btc", len(mk))
def px(t):  # price at time t ~ close of second t-1
    return btc[t-1]
res = {}
for name, f in {
    'spot_end': lambda e: px(e),
    'twap60_end': lambda e: np.mean([px(e-k) for k in range(0, 60)]),
    'twap60_shift1': lambda e: np.mean([px(e-k) for k in range(1, 61)]),
    'twap30_end': lambda e: np.mean([px(e-k) for k in range(0, 30)]),
    'twap120_end': lambda e: np.mean([px(e-k) for k in range(0, 120)]),
}.items():
    d_close = np.array([m['close'] - f(m['start']+300) for m in mk])
    d_open = np.array([m['open'] - f(m['start']) for m in mk])
    # basis drifts: use difference of residuals
    chg_true = np.array([m['close']-m['open'] for m in mk])
    chg_est = np.array([f(m['start']+300)-f(m['start']) for m in mk])
    agree = np.mean(np.sign(chg_true) == np.sign(chg_est))
    print(f"{name:14s} close-basis mean {d_close.mean():8.2f} sd {d_close.std():6.2f} | change err sd {np.std(chg_true-chg_est):6.2f} | sign agree {agree:.3f}")
chg = np.array([m['close']-m['open'] for m in mk])
print("5m change sd", chg.std(), "median abs", np.median(np.abs(chg)))
r = np.diff([btc[t] for t in sorted(btc)])
print("1s return sd $", r.std(), "per sqrt s; robust (MAD*1.4826)", 1.4826*np.median(np.abs(r-np.median(r))))
