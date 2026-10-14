"""Trading logic for the YeNo × Builderr volume bot.

Edge: the markets settle on the Chainlink 60-second TWAP. Late in a market a
large part of that average is already fixed, so a TWAP-aware fair value is
sharper than the order book, which tends to lag BTC moves. The bot buys the
side whose ask is below fair value after both fee layers and expected exit
cost, holds while the bid is still below fair value, and sells when the market
has caught up (or moved against it), always flattening before the close so
every cycle counts as volume.
"""

from __future__ import annotations

import math
from typing import Any

from model import FairValue

TAKER_RATE = 0.07
OVERLAY_RATE = 0.01
TICK = 0.01

DEFAULT_PARAMS: dict[str, float] = {
    # entry window (seconds to close)
    "entry_max_s": 150.0,
    "entry_min_s": 22.0,
    # required expected edge, as a fraction of gross BUY notional
    "edge_base": 0.012,
    "edge_floor": -0.004,
    "edge_ceiling": 0.03,
    "pace_gain": 0.10,
    "pace_goal_hours": 19.0,
    # extra cushion per share against moves during the 250 ms latency
    "latency_cushion": 0.004,
    # price filters for the side we buy
    "min_entry_price": 0.30,
    "max_entry_price": 0.97,
    "min_fair": 0.55,
    # weight of the TWAP model vs the book mid (logit blend); 1 = model only
    "model_weight": 0.6,
    # exit: sell when bid >= fair - hold_margin
    "hold_margin": 0.006,
    "forced_exit_s": 9.0,
    "max_ref_age_s": 1.5,
    "max_cycles_per_market": 6.0,
    "reentry_cooldown_s": 1.0,
    "size_fraction": 1.0,
    "min_share_buffer": 1.15,
}


def _levels(rows: Any, reverse: bool) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for row in rows if isinstance(rows, list) else []:
        try:
            p, q = float(row[0]), float(row[1])
        except (IndexError, TypeError, ValueError):
            continue
        if 0 < p < 1 and q > 0 and math.isfinite(p) and math.isfinite(q):
            out.append((p, q))
    out.sort(reverse=reverse)
    return out


def _logit(p: float) -> float:
    p = min(1 - 1e-6, max(1e-6, p))
    return math.log(p / (1 - p))


def fees(fills: list[tuple[float, float]]) -> float:
    gross = sum(p * q for p, q in fills)
    protocol = sum(TAKER_RATE * q * p * (1 - p) for p, q in fills)
    return round(protocol + 1e-12, 5) + round(gross * OVERLAY_RATE + 1e-12, 5)


def per_share_cost(p: float) -> float:
    return p + TAKER_RATE * p * (1 - p) + OVERLAY_RATE * p


def plan_buy(asks: list[tuple[float, float]], budget: float, limit_price: float):
    """Walk asks up to limit_price within a fee-inclusive budget."""
    fills = []
    left = budget
    for p, q in asks:
        if p > limit_price + 1e-12:
            break
        take = min(q, left / per_share_cost(p))
        if take <= 1e-9:
            break
        fills.append((p, take))
        left -= take * per_share_cost(p)
    shares = sum(q for _, q in fills)
    if shares <= 0:
        return None
    gross = sum(p * q for p, q in fills)
    return shares, gross, fees(fills)


def plan_sell(bids: list[tuple[float, float]], shares: float):
    fills = []
    left = shares
    for p, q in bids:
        k = min(q, left)
        fills.append((p, k))
        left -= k
        if left <= 1e-9:
            break
    if left > 1e-9:
        return None
    gross = sum(p * q for p, q in fills)
    return gross, fees(fills)


class Strategy:
    def __init__(self, params: dict[str, float] | None = None) -> None:
        self.p = dict(DEFAULT_PARAMS)
        if params:
            self.p.update(params)
        self.fv = FairValue()
        self.first_seen: float | None = None
        self.market_id: str | None = None
        self.cycles_here = 0
        self.had_position = False
        self.last_flat_at = -math.inf
        self.last_buy_at = -math.inf
        self.last_ref_t = -math.inf

    # ------------------------------------------------------------------
    def decide(self, obs: dict[str, Any]) -> dict[str, Any]:
        try:
            return self._decide(obs)
        except (KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError):
            return {"action": "HOLD"}

    def _required_edge(self, now: float, account: dict, rules: dict) -> float:
        p = self.p
        target = float(rules.get("targetVolumeUsd", 1000.0)) or 1000.0
        window_h = float(rules.get("evaluationWindowHours", 24.0)) or 24.0
        goal_h = min(p["pace_goal_hours"], window_h)
        elapsed_h = max(0.0, (now - (self.first_seen or now)) / 3600.0)
        progress = float(account.get("eligibleVolumeUsd", 0.0)) / target
        behind = elapsed_h / goal_h - progress  # >0 means behind schedule
        edge = p["edge_base"] - p["pace_gain"] * behind
        # Deep in the window and still short: accept near cost-only cycles.
        if elapsed_h > 0.9 * window_h and progress < 1:
            edge = p["edge_floor"]
        return min(p["edge_ceiling"], max(p["edge_floor"], edge))

    def _decide(self, obs: dict[str, Any]) -> dict[str, Any]:
        p = self.p
        now = float(obs["timestamp"])
        market = obs["market"]
        account = obs["account"]
        rules = obs.get("rules") or {}
        books = obs["books"]
        seconds = float(market["secondsToClose"])
        end = now + seconds
        start = end - 300.0
        market_id = str(market["id"])
        if self.first_seen is None:
            self.first_seen = now

        ref = obs.get("reference")
        ref_ok = False
        target = None
        if isinstance(ref, dict):
            try:
                mid = float(ref["btcMidUsd"])
                ref_t = float(ref.get("observedAt", now))
                if ref_t <= now + 1e-6 and ref_t >= self.last_ref_t:
                    self.fv.observe_mid(ref_t, mid)
                    self.last_ref_t = ref_t
                ref_ok = now - ref_t <= p["max_ref_age_s"]
            except (KeyError, TypeError, ValueError):
                ref_ok = False
            tgt = ref.get("openingTargetUsd")
            tgt_t = ref.get("targetObservedAt")
            if tgt is not None and not ref.get("targetProvisional", True):
                if tgt_t is None or float(tgt_t) <= now + 1e-6:
                    target = float(tgt)
                    self.fv.learn_basis(start, target)

        if market_id != self.market_id:
            self.market_id = market_id
            self.cycles_here = 0
            self.had_position = False
            self.last_flat_at = -math.inf
            self.last_buy_at = -math.inf

        position = account.get("position")
        if position is None and self.had_position:
            self.cycles_here += 1
            self.last_flat_at = now
        self.had_position = position is not None

        fair_yes = None
        if target is not None:
            fair_yes, _, _ = self.fv.prob_yes(now, end, target)
            if fair_yes is not None:
                fair_yes = self._blend(fair_yes, books)

        if position is not None:
            return self._exit(position, books, seconds, fair_yes, ref_ok, market_id)
        if fair_yes is None or not ref_ok or not self.fv.basis_known:
            return {"action": "HOLD"}
        return self._entry(obs, books, seconds, fair_yes, now, account, rules)

    def _blend(self, model_p: float, books: dict) -> float:
        """Shrink the model toward the book: the market is informative too."""
        w = self.p["model_weight"]
        if w >= 1.0:
            return model_p
        mid = None
        yes = books.get("YES", {})
        no = books.get("NO", {})
        yb, ya = _levels(yes.get("bids", []), True), _levels(yes.get("asks", []), False)
        if yb and ya:
            mid = (yb[0][0] + ya[0][0]) / 2.0
        else:
            nb, na = _levels(no.get("bids", []), True), _levels(no.get("asks", []), False)
            if nb and na:
                mid = 1.0 - (nb[0][0] + na[0][0]) / 2.0
        if mid is None:
            return model_p
        lm = _logit(model_p)
        lb = _logit(mid)
        return 1.0 / (1.0 + math.exp(-(w * lm + (1.0 - w) * lb)))

    # ------------------------------------------------------------------
    def _exit(self, position, books, seconds, fair_yes, ref_ok, market_id):
        p = self.p
        if str(position.get("market_id", market_id)) != market_id:
            return {"action": "HOLD"}
        side = str(position["outcome"])
        shares = float(position["shares"])
        bids = _levels(books.get(side, {}).get("bids", []), True)
        if not bids or shares <= 0:
            return {"action": "HOLD"}
        if seconds <= p["forced_exit_s"]:
            return {"action": "SELL"}
        if fair_yes is None or not ref_ok:
            # Blind: get out rather than carry unknown risk into the close.
            return {"action": "SELL"}
        fair = fair_yes if side == "YES" else 1.0 - fair_yes
        sale = plan_sell(bids, shares)
        if sale is None:
            # Not enough displayed depth for a full exit; sell what we can
            # only when the market is no longer cheap versus fair value.
            return {"action": "SELL"} if bids[0][0] >= fair - p["hold_margin"] else {"action": "HOLD"}
        gross, _ = sale
        vwap = gross / shares
        if vwap >= fair - p["hold_margin"]:
            return {"action": "SELL"}
        return {"action": "HOLD"}

    def _entry(self, obs, books, seconds, fair_yes, now, account, rules):
        p = self.p
        if not (p["entry_min_s"] < seconds <= p["entry_max_s"]):
            return {"action": "HOLD"}
        if self.cycles_here >= p["max_cycles_per_market"]:
            return {"action": "HOLD"}
        if now - self.last_flat_at < p["reentry_cooldown_s"] or now - self.last_buy_at < p["reentry_cooldown_s"]:
            return {"action": "HOLD"}
        cash = float(account["cashUsd"])
        cap = min(float(rules.get("maximumBuyCashUsd", 5.0)), cash) * p["size_fraction"]
        cap = math.floor(cap * 1e4) / 1e4 - 0.0005
        if cap <= 0.05:
            return {"action": "HOLD"}
        need = self._required_edge(now, account, rules)
        best = None
        for side in ("YES", "NO"):
            fair = fair_yes if side == "YES" else 1.0 - fair_yes
            if fair < p["min_fair"]:
                continue
            book = books.get(side, {})
            asks = _levels(book.get("asks", []), False)
            bids = _levels(book.get("bids", []), True)
            if not asks or not bids:
                continue
            if not (p["min_entry_price"] <= asks[0][0] <= p["max_entry_price"]):
                continue
            mos = float(book.get("minOrderSize", 0) or 0)
            limit = min(p["max_entry_price"], fair - p["latency_cushion"])
            plan = plan_buy(asks, cap, limit)
            if plan is None:
                continue
            shares, gross, buy_fee = plan
            if shares < max(mos, 0.01) * p["min_share_buffer"]:
                continue
            spread = max(TICK, asks[0][0] - bids[0][0])
            exit_px = max(0.01, fair - spread / 2.0)
            exit_fee = TAKER_RATE * shares * exit_px * (1 - exit_px) + OVERLAY_RATE * shares * exit_px
            ev = shares * (exit_px - p["latency_cushion"]) - exit_fee - gross - buy_fee
            score = ev / gross
            if score >= need and (best is None or score > best[0]):
                best = (score, side, gross + buy_fee)
        if best is None:
            return {"action": "HOLD"}
        self.last_buy_at = now
        return {"action": "BUY", "outcome": best[1], "maxCashUsd": round(min(cap, best[2] * 1.0), 4)}
