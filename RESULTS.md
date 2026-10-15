# Results

Development estimates from the local evaluator (`research/evaluator.py`), not official scores.
All runs start with $10, size BUYs at the $5 cap and use `minOrderSize` 5.

## Out-of-sample live replay (shipped parameters)

Twelve hours of live recordings (`rec-20261014-18` … `rec-20261015-05`, 144 markets)
that no tuning run had seen:

| Parameters | P&L | Volume | Volume / h | Profit per $100 | First half | Second half | Cycles |
|---|---|---|---|---|---|---|---|
| **Shipped defaults** | **+$12.23** | $914 | 76 | **+1.34** | −1.72 | +3.74 | 96 |
| Defaults, looser carry stop (0.3 / 0.4) | +$13.45 | $905 | 75 | +1.49 | −1.72 | +4.01 | 95 |
| Best tuned (carry only, value off) | −$6.10 | $14 | 1 | −43.9 | −43.9 | −4.9 | 2 |
| Best tuned + value mode | −$4.07 | $496 | 41 | −0.82 | −35.5 | +2.30 | 51 |

At 76 USD/h the shipped bot crosses $1,000 in about 13 h, comfortably inside the 24 h limit,
with roughly $23 cash at the crossing projected from the per-volume edge.

## Why the tuned parameters were not adopted

A 201-trial random search over the carry space (`tune.py replay --space carry`) on the
earlier 6 h / 68 markets put carry-only settings on top. None was profitable in both halves
of that data while keeping the 48 USD/h pace, and on the unseen 12 h above the top setting
lost $6 in two trades and stopped trading. The data per tuning run is a few dozen trades,
so search winners mostly fit noise. The defaults stay; the looser stop is within noise
(+$1.2, one cycle difference).

## Synthetic sessions

`tune.py synth` with the defaults (2026-10-14, before carry pacing was added): 24 h sessions
on real BTC paths across three market-maker regimes, 5 of 6 qualified, median cash at the
crossing $18.91 (worst qualified $13.35).
