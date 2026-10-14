#!/usr/bin/env python3
"""Record live YeNo BTC 5-minute Up/Down books plus BTC reference prices.

Development tool only (the competition bot never touches the network).
Writes hourly JSONL files to research/data/:
  {"k":"book","t":recv,"slug":..,"yes":{"b":[[p,s]..],"a":..},"no":..,"mos":1,"ts":server}
  {"k":"btc","t":recv,"src":"binance"|"coinbase","b":bid,"a":ask}
  {"k":"mkt","t":recv,"slug":..,"open":..,"close":..,"res":..}
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import aiohttp
import websockets

API = "https://api.yeno.trade/api"
DATA = Path(__file__).resolve().parent / "data"
DEPTH = 12
BOOK_WORKERS = 3


class Sink:
    def __init__(self) -> None:
        self.hour = None
        self.fh = None

    def write(self, row: dict) -> None:
        hour = time.strftime("%Y%m%d-%H", time.gmtime(row["t"]))
        if hour != self.hour:
            if self.fh:
                self.fh.close()
            self.hour = hour
            self.fh = open(DATA / f"rec-{hour}.jsonl", "a", buffering=1)
        self.fh.write(json.dumps(row, separators=(",", ":")) + "\n")


SINK = Sink()
STATE: dict = {"markets": {}, "current": None}


def slug_for(start: int) -> str:
    return f"btc-updown-5m-{start}"


async def get_json(session: aiohttp.ClientSession, url: str) -> dict | None:
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=4)) as r:
            if r.status != 200:
                return None
            return await r.json()
    except Exception:
        return None


async def market_loop(session: aiohttp.ClientSession) -> None:
    """Resolve market ids, opening targets and final closes."""
    pending_close: set[int] = set()
    while True:
        now = time.time()
        start = int(now - now % 300)
        for s in (start - 300, start, start + 300):
            m = STATE["markets"].get(s)
            need = m is None or (s <= now and m.get("open") is None) or (s in pending_close)
            if not need:
                continue
            d = await get_json(session, f"{API}/markets/{slug_for(s)}")
            if not d or not d.get("data"):
                continue
            x = d["data"]
            row = {
                "k": "mkt", "t": time.time(), "slug": x["slug"], "id": x["id"], "start": s,
                "open": x.get("cryptoOpenPrice"), "close": x.get("cryptoClosePrice"),
                "res": x.get("resolution"),
            }
            if m is None or m.get("open") != row["open"] or m.get("close") != row["close"]:
                SINK.write(row)
            STATE["markets"][s] = row
            if row["close"] is None and s < now:
                pending_close.add(s)
            else:
                pending_close.discard(s)
        # Keep checking recently finished markets until their close is published.
        for s in list(pending_close):
            if s < now - 3600:
                pending_close.discard(s)
        for s in list(STATE["markets"]):
            if s < now - 7200:
                del STATE["markets"][s]
        await asyncio.sleep(2.0)


def trim(levels: list, reverse: bool) -> list:
    out = [[float(l["price"]), float(l["size"])] for l in levels]
    out.sort(key=lambda r: r[0], reverse=reverse)
    return out[:DEPTH]


async def book_worker(session: aiohttp.ClientSession, wid: int) -> None:
    await asyncio.sleep(wid * 0.15)
    last_hash: dict[str, str] = {}
    while True:
        now = time.time()
        start = int(now - now % 300)
        m = STATE["markets"].get(start)
        if not m:
            await asyncio.sleep(0.5)
            continue
        sent = time.time()
        d = await get_json(session, f"{API}/trades/orderbook/{m['id']}")
        recv = time.time()
        if d and d.get("data"):
            x = d["data"]
            if last_hash.get(m["slug"]) != x.get("hash"):
                last_hash[m["slug"]] = x.get("hash")
                SINK.write({
                    "k": "book", "t": recv, "sent": round(sent, 3), "slug": m["slug"],
                    "ts": x.get("timestamp"), "mos": float(x.get("min_order_size") or 0),
                    "yes": {"b": trim(x["yes"]["bids"], True), "a": trim(x["yes"]["asks"], False)},
                    "no": {"b": trim(x["no"]["bids"], True), "a": trim(x["no"]["asks"], False)},
                })
        await asyncio.sleep(0.05)


async def binance_loop() -> None:
    url = "wss://stream.binance.com:9443/ws/btcusdt@bookTicker"
    while True:
        try:
            async with websockets.connect(url, ping_interval=20) as ws:
                last = 0.0
                async for msg in ws:
                    now = time.time()
                    if now - last < 0.1:
                        continue
                    last = now
                    x = json.loads(msg)
                    SINK.write({"k": "btc", "t": now, "src": "binance", "b": float(x["b"]), "a": float(x["a"])})
        except Exception:
            await asyncio.sleep(2)


async def coinbase_loop() -> None:
    url = "wss://ws-feed.exchange.coinbase.com"
    sub = {"type": "subscribe", "product_ids": ["BTC-USD"], "channels": ["ticker"]}
    while True:
        try:
            async with websockets.connect(url, ping_interval=20) as ws:
                await ws.send(json.dumps(sub))
                last = 0.0
                async for msg in ws:
                    x = json.loads(msg)
                    if x.get("type") != "ticker":
                        continue
                    now = time.time()
                    if now - last < 0.1:
                        continue
                    last = now
                    SINK.write({"k": "btc", "t": now, "src": "coinbase",
                                "b": float(x["best_bid"]), "a": float(x["best_ask"])})
        except Exception:
            await asyncio.sleep(2)


async def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    async with aiohttp.ClientSession(headers={"User-Agent": "yeno-research-recorder"}) as session:
        await asyncio.gather(
            market_loop(session),
            *(book_worker(session, i) for i in range(BOOK_WORKERS)),
            binance_loop(),
            coinbase_loop(),
        )


if __name__ == "__main__":
    asyncio.run(main())
