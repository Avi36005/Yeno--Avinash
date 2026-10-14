"""Synthetic 24 h sessions: real BTC 1s paths + real settlements + modeled books.

The book is quoted by a market maker who prices the same TWAP model but sees
BTC with a lag `mm_lag` seconds, carries AR(1) logit noise `mm_noise`, quotes
a 1-tick (sometimes 2-tick) spread, and refreshes at Poisson rate `rate` per
second. Those parameters are calibrated against live recordings
(see calib_book.py) and swept for robustness.
"""

from __future__ import annotations

import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "yeno-bot"))
from model import FairValue  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"


@dataclass
class SynthCfg:
    hours: float = 24.0
    mm_lag: float = 1.0
    mm_noise: float = 0.10  # sd of logit noise
    mm_noise_tau: float = 8.0  # seconds, AR(1) correlation time
    two_tick_prob: float = 0.15
    rate: float = 3.0  # book refreshes per second
    ref_rate: float = 4.0  # reference mid updates per second
    ref_offset: float = 0.0  # added to Binance mid for the reference feed
    depth_median: float = 150.0
    levels: int = 6
    mos: float = 5.0
    seed: int = 0


_BTC = None
_MK = None


def load_hist():
    global _BTC, _MK
    if _BTC is None:
        btc = {}
        for line in open(DATA / "hist-btc1s.jsonl"):
            r = json.loads(line)
            btc[r[0] + 1] = r[4]
        _BTC = btc
        _MK = {m["start"]: m for m in (json.loads(l) for l in open(DATA / "hist-markets.jsonl"))}
    return _BTC, _MK


def windows(hours: float) -> list[int]:
    """Start times (market aligned) of fully covered windows of `hours`."""
    btc, mk = load_hist()
    starts = sorted(mk)
    need = int(hours * 3600 / 300)
    out = []
    for i, s in enumerate(starts):
        run = starts[i:i + need]
        if len(run) == need and run[-1] - s == (need - 1) * 300 and all(
                (x - 61) in btc and (x + 300) in btc for x in (run[0], run[-1])):
            out.append(s)
    return out


def logit(p):
    p = min(1 - 1e-6, max(1e-6, p))
    return math.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def generate(start: int, cfg: SynthCfg):
    """Yield evaluator events; return markets dict via attribute on generator owner."""
    btc, mk = load_hist()
    rng = random.Random(cfg.seed * 1000003 + start)
    n_markets = int(cfg.hours * 3600 / 300)
    markets = {}
    for k in range(n_markets):
        s = start + 300 * k
        m = mk.get(s)
        if m is None:
            continue
        markets[f"m{s}"] = {"start": float(s), "end": float(s + 300), "open": m["open"], "close": m["close"]}

    def price(t: float) -> float:
        a = math.floor(t)
        pa = btc.get(a)
        pb = btc.get(a + 1, pa)
        if pa is None:
            return pb
        return pa + (pb - pa) * (t - a)

    def gen():
        mm = FairValue()
        t0 = float(start - 120)
        t_end = float(start + n_markets * 300)
        # warm MM and reference history
        t = t0
        next_ref = t0
        next_book = t0 + 60
        noise = 0.0
        last_noise_t = t0
        target_emitted = set()
        while t < t_end:
            t = min(next_ref, next_book)
            if t >= t_end:
                break
            if t == next_ref:
                p = price(t)
                if p is not None:
                    yield ("btc", t, p + cfg.ref_offset + rng.gauss(0, 0.8))
                next_ref = t + rng.expovariate(cfg.ref_rate)
                continue
            # book refresh
            next_book = t + rng.expovariate(cfg.rate)
            s = start + 300 * int((t - start) // 300)
            slug = f"m{s}"
            m = markets.get(slug)
            if m is None:
                continue
            if slug not in target_emitted and t >= s + 2:
                target_emitted.add(slug)
                yield ("target", t, slug, m["open"])
            # MM view: lagged BTC, fed at 1 Hz-ish resolution
            tl = t - cfg.mm_lag
            lag_px = price(tl)
            if lag_px is None:
                continue
            mm.observe_mid(tl, lag_px)
            mm.learn_basis(float(s), m["open"])
            if not mm.basis_known:
                continue
            f, _, _ = mm.prob_yes(tl, m["end"], m["open"])
            if f is None:
                continue
            dt = t - last_noise_t
            last_noise_t = t
            rho = math.exp(-dt / cfg.mm_noise_tau)
            noise = rho * noise + math.sqrt(1 - rho * rho) * rng.gauss(0, cfg.mm_noise)
            f = sigmoid(logit(f) + noise)
            ticks = 2 if rng.random() < cfg.two_tick_prob else 1
            bid = math.floor((f - 0.005 * ticks) * 100 + 1e-9) / 100
            ask = round(bid + 0.01 * ticks, 2)
            yes_b, yes_a = [], []
            for i in range(cfg.levels):
                pb = round(bid - 0.01 * i, 2)
                pa = round(ask + 0.01 * i, 2)
                if 0.01 <= pb <= 0.99:
                    yes_b.append([pb, round(cfg.depth_median * math.exp(rng.gauss(0, 0.8)), 2)])
                if 0.01 <= pa <= 0.99:
                    yes_a.append([pa, round(cfg.depth_median * math.exp(rng.gauss(0, 0.8)), 2)])
            no_b = [[round(1 - p, 2), q] for p, q in yes_a]
            no_a = [[round(1 - p, 2), q] for p, q in yes_b]
            yield ("book", t, slug, "YES", yes_b, yes_a, cfg.mos)
            yield ("book", t, slug, "NO", no_b, no_a, cfg.mos)

    return gen(), markets
