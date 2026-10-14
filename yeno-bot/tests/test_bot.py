"""Unit and contract tests. Run: python3 -m unittest discover -s tests -v"""

from __future__ import annotations

import copy
import json
import math
import os
import sys
import threading
import time
import unittest
import urllib.request
from http.server import ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import bot  # noqa: E402
from model import FairValue, tail_cdf  # noqa: E402
from strategy import Strategy, fees, plan_buy, plan_sell  # noqa: E402

SAMPLE = {
    "schemaVersion": 1,
    "timestamp": 1789880823.619,
    "market": {"id": "sample-btc-5m-market", "secondsToClose": 84},
    "account": {"cashUsd": 10, "eligibleVolumeUsd": 0, "completedCycles": 0, "position": None},
    "rules": {"maximumBuyCashUsd": 5, "targetVolumeUsd": 1000, "evaluationWindowHours": 24,
              "onePositionAtATime": True, "buyFeesIncludedInMaximum": True},
    "reference": {"btcMidUsd": 80482.155, "openingTargetUsd": 80424.946, "observedAt": 1789880823.019,
                  "targetObservedAt": 1789880814.203, "targetProvisional": False},
    "books": {"YES": {"bids": [[0.49, 100]], "asks": [[0.50, 100]], "minOrderSize": 5},
              "NO": {"bids": [[0.49, 100]], "asks": [[0.50, 100]], "minOrderSize": 5}},
}


def valid(decision: dict) -> bool:
    a = decision.get("action")
    if a in ("HOLD", "SELL"):
        return set(decision) == {"action"}
    if a == "BUY":
        return decision.get("outcome") in ("YES", "NO") and 0 < float(decision["maxCashUsd"]) <= 5
    return False


class FeeTests(unittest.TestCase):
    def test_fee_formula_matches_contract(self):
        # 5 shares at 0.50: protocol 0.07*5*0.5*0.5 = 0.0875, overlay 1% of 2.5 = 0.025
        self.assertAlmostEqual(fees([(0.5, 5)]), 0.1125, places=5)

    def test_plan_buy_respects_budget_and_limit(self):
        shares, gross, fee = plan_buy([(0.5, 100)], 5.0, 0.6)
        self.assertLessEqual(gross + fee, 5.0 + 1e-6)
        self.assertGreater(shares, 9)
        self.assertIsNone(plan_buy([(0.7, 100)], 5.0, 0.6))

    def test_plan_sell_needs_depth(self):
        self.assertIsNone(plan_sell([(0.5, 2)], 5))
        gross, _ = plan_sell([(0.5, 2), (0.49, 10)], 5)
        self.assertAlmostEqual(gross, 0.5 * 2 + 0.49 * 3)


class ModelTests(unittest.TestCase):
    def test_tail_cdf_symmetry(self):
        self.assertAlmostEqual(tail_cdf(0.0), 0.5)
        self.assertAlmostEqual(tail_cdf(1.3) + tail_cdf(-1.3), 1.0)

    def test_integral_step_function(self):
        fv = FairValue()
        for t, m in [(0, 100.0), (10, 110.0), (20, 120.0)]:
            fv.observe_mid(t, m)
        total, covered = fv.integral(5, 15)
        self.assertAlmostEqual(covered, 10)
        self.assertAlmostEqual(total, 100 * 5 + 110 * 5)

    def test_partial_twap_reduces_uncertainty(self):
        fv = FairValue()
        for t in range(0, 400):
            fv.observe_mid(float(t), 50000.0 + (t % 7))
        fv.learn_basis(300.0 - 0, 50003.0)  # basis sample from pre-open window
        _, sd_early = fv.settle_distribution(now=200.0, end=500.0)
        _, sd_late = fv.settle_distribution(now=399.0, end=430.0)
        self.assertLess(sd_late, sd_early)


class StrategyTests(unittest.TestCase):
    def test_sample_decision_is_valid(self):
        self.assertTrue(valid(Strategy().decide(copy.deepcopy(SAMPLE))))

    def test_no_entry_without_learned_basis(self):
        self.assertEqual(Strategy().decide(copy.deepcopy(SAMPLE)), {"action": "HOLD"})

    def test_garbage_input_holds(self):
        s = Strategy()
        for obs in [{}, {"market": None}, {"timestamp": "x"}, json.loads("[]") or {}]:
            self.assertEqual(s.decide(obs), {"action": "HOLD"})

    def test_forced_exit_near_close(self):
        s = Strategy()
        obs = copy.deepcopy(SAMPLE)
        obs["market"]["secondsToClose"] = 5
        obs["account"]["position"] = {"market_id": "sample-btc-5m-market", "outcome": "YES", "shares": 5,
                                      "buy_gross_usd": 2.5, "buy_fees_usd": 0.11, "sell_gross_usd": 0,
                                      "sell_fees_usd": 0}
        self.assertEqual(s.decide(obs), {"action": "SELL"})

    def test_buy_cap_never_exceeds_cash_or_limit(self):
        s = Strategy()
        # Teach a basis: feed mids through the pre-open window of the market.
        end = SAMPLE["timestamp"] + 84
        start = end - 300
        for k in range(130):
            t = start - 70 + k
            obs = copy.deepcopy(SAMPLE)
            obs["timestamp"] = t + 0.1
            obs["market"]["secondsToClose"] = end - (t + 0.1)
            obs["reference"]["observedAt"] = t
            obs["reference"]["btcMidUsd"] = 80424.946
            obs["reference"]["targetObservedAt"] = start
            s.decide(obs)
        for cash in (10, 3.2, 0.5):
            obs = copy.deepcopy(SAMPLE)
            obs["account"]["cashUsd"] = cash
            obs["timestamp"] = end - 60
            obs["market"]["secondsToClose"] = 60
            obs["reference"]["observedAt"] = end - 60.2
            obs["reference"]["btcMidUsd"] = 80424.946 + 60
            obs["books"]["YES"]["asks"] = [[0.55, 100]]
            obs["books"]["YES"]["bids"] = [[0.54, 100]]
            d = s.decide(obs)
            self.assertTrue(valid(d))
            if d["action"] == "BUY":
                self.assertLessEqual(d["maxCashUsd"], min(5, cash))
            s.last_buy_at = -math.inf


class HttpContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), bot.Handler)
        cls.port = cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def post(self, payload: bytes):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/decide", data=payload, method="POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=1) as r:
            return r.status, json.loads(r.read())

    def test_valid_reply_and_latency(self):
        worst = 0.0
        for _ in range(50):
            t0 = time.perf_counter()
            status, body = self.post(json.dumps(SAMPLE).encode())
            worst = max(worst, time.perf_counter() - t0)
            self.assertEqual(status, 200)
            self.assertTrue(valid(body))
        self.assertLess(worst, 0.2)

    def test_malformed_json_holds(self):
        status, body = self.post(b"{not json")
        self.assertEqual((status, body), (200, {"action": "HOLD"}))


if __name__ == "__main__":
    unittest.main()
