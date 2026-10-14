"""Local paper evaluator that mirrors the published YeNo contract.

Event stream (time ordered):
  ("btc", t, mid)
  ("book", t, slug, outcome, bids, asks, min_order_size)
  ("target", t, slug, open_target)      # opening target becomes observable
Markets: dict slug -> {"start", "end", "open", "close"} (close = settlement TWAP).

Contract implemented:
  $10 start cash, $5 fee-inclusive BUY cap, one position or pending BUY,
  250 ms latency then cross the next L2 update of the same outcome,
  books older than 2 s rejected, no entries inside the final 15 s,
  dynamic taker fee 0.07*q*p*(1-p) (5 dp) + 1% overlay (5 dp) each side,
  partial fills, min order size on BUY, positions held through the close
  settle with zero volume, volume = gross BUY + gross SELL of fully closed
  cycles, stop at first clean $1,000 crossing, 24 h window, 200 ms reply.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

TICK = 0.01


def fee_for(fills: list[tuple[float, float]]) -> float:
    gross = sum(p * q for p, q in fills)
    protocol = sum(0.07 * q * p * (1 - p) for p, q in fills)
    return round(protocol + 1e-12, 5) + round(gross * 0.01 + 1e-12, 5)


@dataclass
class Config:
    start_cash: float = 10.0
    max_buy: float = 5.0
    target: float = 1000.0
    window_s: float = 24 * 3600.0
    latency_s: float = 0.25
    stale_s: float = 2.0
    entry_block_s: float = 15.0
    min_order_override: float | None = None
    adverse_ticks: int = 0  # stress: shift executed prices against the bot
    depth_haircut: float = 1.0  # stress: fraction of displayed depth available
    reply_timeout_s: float = 0.2
    enforce_timeout: bool = False
    share_step: float = 0.01


@dataclass
class Result:
    qualified: bool = False
    cash_at_cross: float | None = None
    time_to_target_h: float | None = None
    volume: float = 0.0
    cycles: int = 0
    final_cash: float = 0.0
    max_drawdown: float = 0.0
    settled_positions: int = 0
    rejected: int = 0
    buys: int = 0
    decisions: int = 0
    slow_replies: int = 0
    flat_at_end: bool = True
    hours: float = 0.0
    log: list = field(default_factory=list)


class Evaluator:
    def __init__(self, decide: Callable[[dict], dict], markets: dict, cfg: Config | None = None,
                 keep_log: bool = False) -> None:
        self.decide = decide
        self.markets = markets
        self.cfg = cfg or Config()
        self.keep_log = keep_log
        self.cash = self.cfg.start_cash
        self.volume = 0.0
        self.cycles = 0
        self.position: dict | None = None
        self.pending: dict | None = None
        self.books: dict[tuple[str, str], tuple[float, list, list, float]] = {}
        self.ref_mid: tuple[float, float] | None = None
        self.targets: dict[str, tuple[float, float]] = {}
        self.res = Result()
        self.t0: float | None = None
        self.peak = self.cash
        self.done = False

    # ------------------------------------------------------------------
    def _current_market(self, t: float) -> str | None:
        for slug, m in self._active:
            if m["start"] <= t < m["end"]:
                return slug
        return None

    def run(self, events: Iterable[tuple]) -> Result:
        # Small cache of markets sorted by start for current-market lookup.
        ms = sorted(self.markets.items(), key=lambda kv: kv[1]["start"])
        self._ms = ms
        self._mi = 0
        self._active = []
        last_t = None
        for ev in events:
            t = ev[1]
            if self.t0 is None:
                self.t0 = t
            if t - self.t0 > self.cfg.window_s:
                break
            last_t = t
            self._advance_markets(t)
            self._settle_expired(t)
            kind = ev[0]
            if kind == "btc":
                self.ref_mid = (t, ev[2])
                continue
            if kind == "target":
                self.targets[ev[2]] = (t, ev[3])
                continue
            # book
            _, _, slug, outcome, bids, asks, mos = ev
            if self.cfg.min_order_override is not None:
                mos = self.cfg.min_order_override
            self.books[(slug, outcome)] = (t, bids, asks, mos)
            if self.pending and self.pending["slug"] == slug and self.pending["outcome"] == outcome \
                    and t >= self.pending["exec_after"]:
                self._execute(t, bids, asks, mos)
                if self.done:
                    break
            if self.pending is None:
                self._maybe_decide(t, slug)
                if self.done:
                    break
        if last_t is not None:
            self.res.hours = (last_t - (self.t0 or last_t)) / 3600
        self.res.volume = self.volume
        self.res.cycles = self.cycles
        self.res.final_cash = self.cash
        self.res.flat_at_end = self.position is None and self.pending is None
        return self.res

    def _advance_markets(self, t: float) -> None:
        while self._mi < len(self._ms) and self._ms[self._mi][1]["start"] <= t + 1:
            self._active.append(self._ms[self._mi])
            self._mi += 1
        self._active = [kv for kv in self._active if kv[1]["end"] > t - 30]

    def _settle_expired(self, t: float) -> None:
        if self.pending and t >= self.markets[self.pending["slug"]]["end"]:
            self.pending = None
        p = self.position
        if p and t >= self.markets[p["market_id"]]["end"]:
            m = self.markets[p["market_id"]]
            yes_wins = m["close"] >= m["open"]
            won = (p["outcome"] == "YES") == yes_wins
            self.cash += p["shares"] * (1.0 if won else 0.0)
            self.res.settled_positions += 1
            self._log(t, "SETTLE", won=won, shares=p["shares"])
            self.position = None
            self._mark_equity()

    def _mark_equity(self) -> None:
        self.peak = max(self.peak, self.cash)
        self.res.max_drawdown = max(self.res.max_drawdown, self.peak - self.cash)

    def _log(self, t, what, **kw):
        if self.keep_log:
            self.res.log.append((t, what, kw))

    # ------------------------------------------------------------------
    def _maybe_decide(self, t: float, slug: str) -> None:
        cur = self._current_market(t)
        if cur is None or cur != slug:
            return
        if self.position and self.position["market_id"] != cur:
            return
        by = self.books.get((cur, "YES"))
        bn = self.books.get((cur, "NO"))
        if not by or not bn or t - by[0] > self.cfg.stale_s or t - bn[0] > self.cfg.stale_s:
            return
        m = self.markets[cur]
        ref = None
        if self.ref_mid:
            tgt = self.targets.get(cur)
            ref = {
                "btcMidUsd": self.ref_mid[1],
                "observedAt": self.ref_mid[0],
                "openingTargetUsd": tgt[1] if tgt else None,
                "targetObservedAt": tgt[0] if tgt else None,
                "targetProvisional": tgt is None,
            }
        pos = copy.deepcopy(self.position) if self.position else None
        if pos is not None:
            pos.pop("buy_time", None)
        obs = {
            "schemaVersion": 1,
            "timestamp": t,
            "market": {"id": cur, "secondsToClose": round(m["end"] - t, 3)},
            "account": {"cashUsd": round(self.cash, 6), "eligibleVolumeUsd": round(self.volume, 6),
                        "completedCycles": self.cycles, "position": pos},
            "rules": {"maximumBuyCashUsd": self.cfg.max_buy, "targetVolumeUsd": self.cfg.target,
                      "evaluationWindowHours": self.cfg.window_s / 3600, "onePositionAtATime": True,
                      "buyFeesIncludedInMaximum": True},
            "reference": ref,
            "books": {
                "YES": {"bids": by[1], "asks": by[2], "minOrderSize": by[3]},
                "NO": {"bids": bn[1], "asks": bn[2], "minOrderSize": bn[3]},
            },
        }
        self.res.decisions += 1
        w0 = time.perf_counter()
        try:
            act = self.decide(obs) or {"action": "HOLD"}
        except Exception as exc:  # a crashing bot holds
            self._log(t, "ERROR", err=repr(exc))
            act = {"action": "HOLD"}
        if time.perf_counter() - w0 > self.cfg.reply_timeout_s:
            self.res.slow_replies += 1
            if self.cfg.enforce_timeout:
                return
        a = act.get("action")
        if a == "BUY":
            if self.position is not None:
                self.res.rejected += 1
                return
            outcome = act.get("outcome")
            try:
                cap = float(act.get("maxCashUsd", 0))
            except (TypeError, ValueError):
                cap = 0
            if outcome not in ("YES", "NO") or not (0 < cap <= self.cfg.max_buy + 1e-9):
                self.res.rejected += 1
                return
            if m["end"] - t <= self.cfg.entry_block_s:
                self.res.rejected += 1
                return
            cap = min(cap, self.cash)
            self.pending = {"kind": "BUY", "slug": cur, "outcome": outcome, "cap": cap,
                            "exec_after": t + self.cfg.latency_s, "t": t}
        elif a == "SELL":
            if self.position is None:
                self.res.rejected += 1
                return
            self.pending = {"kind": "SELL", "slug": cur, "outcome": self.position["outcome"],
                            "exec_after": t + self.cfg.latency_s, "t": t}

    # ------------------------------------------------------------------
    def _levels(self, rows, reverse):
        out = []
        for r in rows:
            p, q = float(r[0]), float(r[1]) * self.cfg.depth_haircut
            if 0 < p < 1 and q > 0:
                out.append((p, q))
        out.sort(reverse=reverse)
        return out

    def _execute(self, t, bids, asks, mos) -> None:
        pend = self.pending
        self.pending = None
        adv = self.cfg.adverse_ticks * TICK
        step = self.cfg.share_step
        if pend["kind"] == "BUY":
            remaining_cash = pend["cap"]
            fills = []
            for p, q in self._levels(asks, False):
                p = min(0.99, p + adv)
                per = p + 0.07 * p * (1 - p) + 0.01 * p
                take = min(q, remaining_cash / per)
                if take <= 0:
                    break
                fills.append([p, take])
                remaining_cash -= take * per
                if remaining_cash <= 1e-9:
                    break
            shares = sum(q for _, q in fills)
            shares = math.floor(shares / step + 1e-9) * step
            # trim fills to quantized share count
            out, left = [], shares
            for p, q in fills:
                k = min(q, left)
                if k > 1e-12:
                    out.append((p, k))
                left -= k
            gross = sum(p * q for p, q in out)
            fees = fee_for(out)
            while out and gross + fees > pend["cap"] + 1e-9:
                shares = round(shares - step, 8)
                out, left = [], shares
                for p, q in fills:
                    k = min(q, left)
                    if k > 1e-12:
                        out.append((p, k))
                    left -= k
                gross = sum(p * q for p, q in out)
                fees = fee_for(out)
            if shares < max(mos, step) - 1e-9 or not out:
                self.res.rejected += 1
                self._log(t, "BUY_REJECT", shares=shares, cap=pend["cap"])
                return
            self.cash -= gross + fees
            self.res.buys += 1
            self.position = {"market_id": pend["slug"], "outcome": pend["outcome"], "shares": shares,
                             "buy_gross_usd": gross, "buy_fees_usd": fees,
                             "sell_gross_usd": 0.0, "sell_fees_usd": 0.0, "buy_time": t}
            self._log(t, "BUY", outcome=pend["outcome"], shares=shares, gross=gross, fees=fees,
                      avg=gross / shares)
        else:
            pos = self.position
            if pos is None:
                return
            want = pos["shares"]
            fills, left = [], want
            for p, q in self._levels(bids, True):
                p = max(0.01, p - adv)
                k = min(q, left)
                fills.append((p, k))
                left -= k
                if left <= 1e-9:
                    break
            sold = want - max(0.0, left)
            if sold <= 1e-9:
                self.res.rejected += 1
                self._log(t, "SELL_ZERO")
                return
            gross = sum(p * q for p, q in fills)
            fees = fee_for(fills)
            self.cash += gross - fees
            pos["shares"] = round(want - sold, 8)
            pos["sell_gross_usd"] += gross
            pos["sell_fees_usd"] += fees
            self._log(t, "SELL", sold=sold, gross=gross, fees=fees, avg=gross / sold, left=pos["shares"])
            if pos["shares"] <= 1e-6:
                self.volume += pos["buy_gross_usd"] + pos["sell_gross_usd"]
                self.cycles += 1
                self.position = None
                self._mark_equity()
                if self.volume >= self.cfg.target - 1e-9:
                    self.res.qualified = True
                    self.res.cash_at_cross = self.cash
                    self.res.time_to_target_h = (t - self.t0) / 3600
                    self.done = True
