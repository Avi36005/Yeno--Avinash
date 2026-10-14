"""Fair-value model for YeNo BTC five-minute Up/Down markets.

The markets settle on the Chainlink BTC/USD 60-second TWAP: YES wins when the
TWAP at the close is >= the opening target (the TWAP at the open). Inside the
final minute most of that average is already observed, so the outcome is far
more certain than a spot-price model suggests.
"""

from __future__ import annotations

import math
from collections import deque

TWAP_SECONDS = 60.0
TWAP_END_OFFSET = 1.0  # settlement window is [end-61, end-1]
DEFAULT_SIGMA = 4.5  # USD per sqrt(second), long-horizon BTC median
SIGMA_FLOOR = 1.5
SIGMA_CEIL = 20.0
VOL_SAMPLE_SECONDS = 5.0
VOL_HALF_LIFE_SECONDS = 900.0
VOL_HORIZON_FACTOR = 1.12  # 5s changes understate longer-horizon variance (momentum)
HISTORY_SECONDS = 1200.0
BASIS_ALPHA = 0.35
MODEL_NOISE_USD = 3.5  # TWAP replication error measured on 863 settled markets
UNKNOWN_BASIS_NOISE_USD = 25.0

# Fat-tailed standardized mixture: weights, scale. Total variance ~= 1.
_MIX = ((0.8, 0.8), (0.2, 1.6))


def _phi(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def tail_cdf(z: float) -> float:
    return sum(w * _phi(z / k) for w, k in _MIX)


class FairValue:
    def __init__(self) -> None:
        self.hist: deque[tuple[float, float]] = deque()
        self.var = DEFAULT_SIGMA ** 2
        self.var_weight = 0.0
        self._vol_t: float | None = None
        self._vol_mid: float | None = None
        self.basis = 0.0
        self.basis_count = 0
        self._basis_markets: set[float] = set()

    # ---- observations -------------------------------------------------
    def observe_mid(self, t: float, mid: float) -> None:
        if not (mid > 0 and math.isfinite(mid)):
            return
        if self.hist and t < self.hist[-1][0]:
            return
        if self.hist and t == self.hist[-1][0]:
            self.hist[-1] = (t, mid)
        else:
            self.hist.append((t, mid))
        while self.hist and self.hist[0][0] < t - HISTORY_SECONDS:
            self.hist.popleft()
        if self._vol_t is None or t - self._vol_t > 120:
            self._vol_t, self._vol_mid = t, mid
            return
        dt = t - self._vol_t
        if dt >= VOL_SAMPLE_SECONDS:
            sample = (mid - self._vol_mid) ** 2 / dt
            sample = min(sample, 25.0 * self.var)
            decay = 0.5 ** (dt / VOL_HALF_LIFE_SECONDS)
            self.var = decay * self.var + (1 - decay) * sample
            self.var_weight = min(1.0, self.var_weight + dt / VOL_HALF_LIFE_SECONDS)
            self._vol_t, self._vol_mid = t, mid

    @property
    def sigma(self) -> float:
        s = math.sqrt(self.var) * VOL_HORIZON_FACTOR
        # Shrink toward the long-run prior until enough data has been seen.
        w = self.var_weight
        s = w * s + (1 - w) * DEFAULT_SIGMA
        return min(SIGMA_CEIL, max(SIGMA_FLOOR, s))

    def last_mid(self) -> tuple[float, float] | None:
        return self.hist[-1] if self.hist else None

    # ---- TWAP helpers -------------------------------------------------
    def integral(self, a: float, b: float) -> tuple[float, float]:
        """Integral of the step-interpolated mid over [a, b] and covered seconds."""
        if b <= a or not self.hist:
            return 0.0, 0.0
        total = 0.0
        covered = 0.0
        h = self.hist
        n = len(h)
        # binary search first index with time > a
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi) // 2
            if h[mid][0] <= a:
                lo = mid + 1
            else:
                hi = mid
        i = lo - 1
        if i < 0:
            # history starts after a; begin at first sample
            i = 0
            cur_t = h[0][0]
            if cur_t >= b:
                return 0.0, 0.0
        else:
            cur_t = a
        while i < n and cur_t < b:
            nxt = h[i + 1][0] if i + 1 < n else b
            seg_end = min(nxt, b)
            if seg_end > cur_t:
                total += h[i][1] * (seg_end - cur_t)
                covered += seg_end - cur_t
            cur_t = seg_end
            i += 1
        return total, covered

    def learn_basis(self, start: float, open_target: float) -> None:
        if start in self._basis_markets:
            return
        a, b = start - TWAP_SECONDS - TWAP_END_OFFSET, start - TWAP_END_OFFSET
        total, covered = self.integral(a, b)
        if covered < 0.8 * TWAP_SECONDS:
            return
        self._basis_markets.add(start)
        sample = open_target - total / covered
        if self.basis_count == 0:
            self.basis = sample
        else:
            # Reject a wild sample (bad target row) once a basis is known.
            if abs(sample - self.basis) > 60:
                return
            self.basis += BASIS_ALPHA * (sample - self.basis)
        self.basis_count += 1
        if len(self._basis_markets) > 64:
            self._basis_markets = set(sorted(self._basis_markets)[-32:])

    # ---- prediction ---------------------------------------------------
    def settle_distribution(self, now: float, end: float) -> tuple[float, float] | None:
        last = self.last_mid()
        if last is None:
            return None
        mid_now = last[1]
        sigma = self.sigma
        ws = end - TWAP_END_OFFSET - TWAP_SECONDS
        we = end - TWAP_END_OFFSET
        if now >= we:
            total, covered = self.integral(ws, we)
            mean = total / covered if covered > 0 else mid_now
            var = 0.0
        elif now > ws:
            total, covered = self.integral(ws, now)
            missing = (now - ws) - covered
            future = we - now
            mean = (total + (missing + future) * mid_now) / TWAP_SECONDS
            var = sigma ** 2 * (future ** 3) / (3.0 * TWAP_SECONDS ** 2)
        else:
            mean = mid_now
            var = sigma ** 2 * ((ws - now) + TWAP_SECONDS / 3.0)
        noise = MODEL_NOISE_USD if self.basis_count else UNKNOWN_BASIS_NOISE_USD
        return mean + self.basis, math.sqrt(var + noise ** 2)

    def prob_yes(self, now: float, end: float, open_target: float):
        dist = self.settle_distribution(now, end)
        if dist is None:
            return None, None, None
        mean, sd = dist
        p = tail_cdf((mean - open_target) / sd)
        return p, mean, sd
