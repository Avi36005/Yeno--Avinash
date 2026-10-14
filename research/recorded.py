"""Load live recordings (rec-*.jsonl) into evaluator events + market table."""

from __future__ import annotations

import glob
import json
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"


def load_rows(pattern: str = "rec-*.jsonl") -> list[dict]:
    rows = []
    for f in sorted(glob.glob(str(DATA / pattern))):
        with open(f) as fh:
            for line in fh:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    rows.sort(key=lambda r: r["t"])
    return rows


def load_hist_closes() -> dict[int, dict]:
    out = {}
    p = DATA / "hist-markets.jsonl"
    if p.exists():
        for line in p.open():
            m = json.loads(line)
            out[m["start"]] = m
    return out


def build(rows: list[dict], ref_src: str = "coinbase", ref_offset: float = 0.0):
    """Return (events, markets). Only markets with a known settlement are kept."""
    hist = load_hist_closes()
    markets: dict[str, dict] = {}
    target_seen: dict[str, float] = {}
    for r in rows:
        if r["k"] != "mkt":
            continue
        s = r["start"]
        m = markets.setdefault(r["slug"], {"start": float(s), "end": float(s + 300), "open": None, "close": None})
        if r.get("open") is not None:
            m["open"] = float(r["open"])
            target_seen.setdefault(r["slug"], r["t"])
        if r.get("close") is not None:
            m["close"] = float(r["close"])
    for slug, m in markets.items():
        h = hist.get(int(m["start"]))
        if h:
            m["open"] = m["open"] if m["open"] is not None else h["open"]
            m["close"] = m["close"] if m["close"] is not None else h["close"]
    markets = {k: v for k, v in markets.items() if v["open"] is not None and v["close"] is not None}
    events = []
    for slug, t in target_seen.items():
        if slug in markets:
            events.append(("target", t, slug, markets[slug]["open"]))
    for r in rows:
        k = r["k"]
        if k == "btc":
            if r["src"] == ref_src:
                events.append(("btc", r["t"], (r["b"] + r["a"]) / 2 + ref_offset))
        elif k == "book" and r["slug"] in markets:
            t = r["t"]
            mos = r.get("mos") or 1.0
            events.append(("book", t, r["slug"], "YES", r["yes"]["b"], r["yes"]["a"], mos))
            events.append(("book", t, r["slug"], "NO", r["no"]["b"], r["no"]["a"], mos))
    order = {"target": 0, "btc": 1, "book": 2}
    events.sort(key=lambda e: (e[1], order[e[0]]))
    return events, markets
