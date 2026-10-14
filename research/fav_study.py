"""Late-favorite carry study on live recordings.

For each market: at the first book update with S in [s_lo, s_hi] where the
favorite's best ask is in [a_lo, a_hi], buy ~$5 at the next book >= 250 ms
later (walking asks, both fees); then sell the whole position at the first
later book whose best bid >= exit_px, or at S <= forced (best available bids).
Reports per-trade PnL and per-$100-volume.
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "yeno-bot"))
from recorded import load_rows, build
from strategy import plan_buy, fees


def run(events, markets, s_lo, s_hi, a_lo, a_hi, exit_px, forced, budget=5.0, max_trades=3):
    by = {}
    for e in events:
        if e[0] == "book":
            by.setdefault(e[2], {}).setdefault(e[3], []).append((e[1], e[4], e[5]))
    trades = []
    for slug, m in markets.items():
        b = by.get(slug)
        if not b or "YES" not in b or "NO" not in b:
            continue
        n_tr = 0
        t_free = -1e18
        for side in ("YES", "NO"):
            pass
        # merge by time: candidate entries from either side
        seq = sorted([(t, side, bids, asks) for side in ("YES", "NO") for t, bids, asks in b[side]])
        i = 0
        while i < len(seq) and n_tr < max_trades:
            t, side, bids, asks = seq[i]
            S = m["end"] - t
            i += 1
            if t < t_free or not (s_lo <= S <= s_hi) or not asks or not bids:
                continue
            if not (a_lo <= asks[0][0] <= a_hi):
                continue
            # execution book: next update of same side >= t + 0.25
            j = i
            while j < len(seq) and not (seq[j][1] == side and seq[j][0] >= t + 0.25):
                j += 1
            if j >= len(seq):
                break
            et, _, _, easks = seq[j]
            if m["end"] - et <= 15:
                continue
            lv = sorted((float(p), float(q)) for p, q in easks)
            plan = plan_buy(lv, budget, 0.999)
            if plan is None or plan[0] < 5:
                continue
            shares, gross, bfee = plan
            # exit
            k = j + 1
            sold = None
            while k < len(seq):
                tt, sd, bb, aa = seq[k]
                k += 1
                if sd != side:
                    continue
                S2 = m["end"] - tt
                if S2 <= 0:
                    break
                if bb and (bb[0][0] >= exit_px or S2 <= forced):
                    # execute at next same-side book >= tt + 0.25
                    x = k
                    while x < len(seq) and not (seq[x][1] == side and seq[x][0] >= tt + 0.25):
                        x += 1
                    if x >= len(seq) or m["end"] - seq[x][0] <= 0:
                        break
                    xb = sorted(((float(p), float(q)) for p, q in seq[x][2]), reverse=True)
                    left, fills = shares, []
                    for p, q in xb:
                        kq = min(q, left); fills.append((p, kq)); left -= kq
                        if left <= 1e-9: break
                    if left > 1e-9:
                        continue
                    sg = sum(p * q for p, q in fills)
                    sold = (sg, fees(fills), seq[x][0])
                    k = x + 1
                    break
            win = (m["close"] >= m["open"]) == (side == "YES")
            if sold is None:
                pnl = (shares if win else 0) - gross - bfee
                trades.append(dict(slug=slug, S=S, ask=gross / shares, pnl=pnl, vol=0.0, settled=True, win=win))
                t_free = m["end"]
            else:
                pnl = sold[0] - sold[1] - gross - bfee
                trades.append(dict(slug=slug, S=S, ask=gross / shares, exit=sold[0] / shares, pnl=pnl,
                                   vol=gross + sold[0], settled=False, win=win))
                t_free = sold[2] + 1.0
            n_tr += 1
            i = next((z for z in range(i, len(seq)) if seq[z][0] >= t_free), len(seq))
    return trades


def summary(tr):
    if not tr:
        return "no trades"
    pnl = sum(t["pnl"] for t in tr); vol = sum(t["vol"] for t in tr)
    losses = [round(t["pnl"], 2) for t in tr if t["pnl"] < -0.3]
    return (f"trades={len(tr)} markets={len({t['slug'] for t in tr})} pnl={pnl:+.3f} vol={vol:.1f} "
            f"per100={100 * pnl / vol if vol else 0:+.3f} settled={sum(t['settled'] for t in tr)} big_losses={losses}")


if __name__ == "__main__":
    ev, mk = build(load_rows(), ref_src="coinbase")
    print("markets", len(mk))
    for s_lo, s_hi in [(20, 60), (30, 90), (45, 120), (60, 150)]:
        for a_lo, a_hi in [(0.85, 0.93), (0.90, 0.96), (0.93, 0.98), (0.95, 0.99)]:
            for exit_px in [0.99, 0.995]:
                tr = run(ev, mk, s_lo, s_hi, a_lo, a_hi, exit_px, forced=8)
                print(f"S[{s_lo},{s_hi}] ask[{a_lo},{a_hi}] exit>={exit_px}: {summary(tr)}")
