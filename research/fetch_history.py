#!/usr/bin/env python3
"""Fetch past BTC 5m market opens/closes (YeNo) and Binance 1s klines.

Usage: python3 fetch_history.py HOURS
Writes data/hist-markets.jsonl and data/hist-btc1s.jsonl (append, dedup by key).
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import aiohttp

API = "https://api.yeno.trade/api"
DATA = Path(__file__).resolve().parent / "data"


async def get(session, url, params=None):
    for attempt in range(4):
        try:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as r:
                if r.status == 200:
                    return await r.json()
                if r.status in (429, 418):
                    await asyncio.sleep(5)
        except Exception:
            await asyncio.sleep(1 + attempt)
    return None


async def main(hours: float) -> None:
    now = int(time.time())
    last = now - now % 300 - 300
    starts = list(range(last - int(hours * 3600), last + 1, 300))
    mpath = DATA / "hist-markets.jsonl"
    have = set()
    if mpath.exists():
        for line in mpath.open():
            have.add(json.loads(line)["start"])
    bpath = DATA / "hist-btc1s.jsonl"
    have_k = set()
    if bpath.exists():
        for line in bpath.open():
            have_k.add(json.loads(line)[0])
    sem = asyncio.Semaphore(6)
    async with aiohttp.ClientSession() as session:
        async def one_market(s):
            if s in have:
                return None
            async with sem:
                d = await get(session, f"{API}/markets/btc-updown-5m-{s}")
            if not d or not d.get("data"):
                return None
            x = d["data"]
            if x.get("cryptoClosePrice") is None or x.get("cryptoOpenPrice") is None:
                return None
            return {"start": s, "open": float(x["cryptoOpenPrice"]), "close": float(x["cryptoClosePrice"]),
                    "res": x.get("resolution"), "volume": x.get("volume")}

        rows = [r for r in await asyncio.gather(*(one_market(s) for s in starts)) if r]
        with mpath.open("a") as fh:
            for r in sorted(rows, key=lambda r: r["start"]):
                fh.write(json.dumps(r) + "\n")
        print("markets fetched", len(rows))

        # Binance 1s klines: [openTime, open, high, low, close, volume, ...]
        t0 = (starts[0] - 120) * 1000
        t1 = (last + 300) * 1000
        chunks = list(range(t0, t1, 1000 * 1000))
        bsem = asyncio.Semaphore(4)

        async def one_chunk(c):
            async with bsem:
                return await get(session, "https://api.binance.com/api/v3/klines",
                                 {"symbol": "BTCUSDT", "interval": "1s", "startTime": c, "limit": 1000})

        out = await asyncio.gather(*(one_chunk(c) for c in chunks))
        n = 0
        with bpath.open("a") as fh:
            for res in out:
                for k in res or []:
                    ts = k[0] // 1000
                    if ts in have_k or ts * 1000 >= t1:
                        continue
                    have_k.add(ts)
                    fh.write(json.dumps([ts, float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])]) + "\n")
                    n += 1
        print("btc seconds fetched", n)


if __name__ == "__main__":
    asyncio.run(main(float(sys.argv[1]) if len(sys.argv) > 1 else 24))
