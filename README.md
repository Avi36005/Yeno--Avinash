# YeNo × Builderr Volume Bot

Entry for the [YeNo × Builderr Volume Bot Challenge](https://builderr.ai/challenges/yeno-volume-bots):
a paper bot that must turn **$10** into **$1,000 of completed BUY+SELL volume** on
YeNo BTC five-minute Up/Down markets within 24 hours, and be flat at the first clean crossing.
Ranking is by cash at that crossing.

The submission is the `yeno-bot/` folder: a standard-library Python server exposing
`POST /decide`. Everything else is research tooling used to build and tune it.

```
yeno-bot/            <- the submitted bot (stdlib only, no network)
  bot.py             HTTP server: POST /decide, GET /health
  strategy.py        entry / exit / pacing logic
  model.py           TWAP-aware fair value (vol, basis, settlement distribution)
  Dockerfile         frozen container build
  tests/             unit + HTTP contract tests
research/            <- development tooling (not part of the submission)
  recorder.py        live YeNo L2 books + Coinbase/Binance BTC mids -> JSONL
  fetch_history.py   past market opens/closes + Binance 1 s klines
  evaluator.py       local paper evaluator implementing the published contract
  recorded.py        recordings -> evaluator events
  synth.py           24 h synthetic sessions on real BTC paths and settlements
  run_replay.py      replay recordings (in-process or over HTTP, 200 ms enforced)
  run_synth.py       parallel synthetic sessions, ours vs House Bot v1
  tune.py            random-search tuning (synthetic and replay modes)
  calib_*.py, alpha_study.py, fav_study.py, edge_scan.py, peek.py, diag.py   analysis
```

## Run it

```bash
cd yeno-bot
python3 bot.py                    # listens on 127.0.0.1:8080
python3 bot.py --host 0.0.0.0 --port 8080
python3 -m unittest discover -s tests -v
```

Official smoke test (from the starter kit):

```bash
python3 starter-kit/yeno-smoke-test.py --url http://127.0.0.1:8080/decide   # -> "contractSmokePassed": true
```

Docker:

```bash
docker build -t yeno-volume-bot yeno-bot
docker run --rm -p 8080:8080 yeno-volume-bot
```

## How the bot works

**1. The market settles on a 60-second TWAP, not spot.** Each market resolves on the
Chainlink BTC/USD *60 s TWAP* at the close versus the TWAP at the open (the opening
target). Measured on 863 settled markets, a 60 s TWAP of BTC reproduces the 5-minute change
within $3.5 and gets the direction right 98% of the time; spot gets 84.5%.
Inside the last minute most of that average is already fixed, so uncertainty collapses
much faster than a spot model implies (variance `σ²·s³/(3·60²)` for `s` seconds left in
the window instead of `σ²·s`).

**2. Fair value** (`model.py`)
- keeps the reference mid history across markets and integrates it into the partial TWAP;
- learns the *basis* between the reference feed and the settlement feed from each market's
  opening target (`target − TWAP(mids over the minute before the open)`); entries are blocked
  until a basis is known, because a different feed (e.g. Binance USDT, ~$95 off) would invert
  the signal;
- estimates volatility online (EWMA of 5 s changes, scaled for BTC's short-horizon momentum);
- uses a fat-tailed CDF; reliability on 9,493 historical checkpoints is within a percent
  across all probability buckets.

**3. Trading** (`strategy.py`): value mode plus favorite-carry mode, both on by default
- blends the model with the book mid in logit space (`model_weight`) — the book is informative;
- walks the ask ladder with both fee layers (`0.07·q·p(1−p)` + 1% overlay), a latency cushion
  for the 250 ms delay and the expected exit cost; buys only when expected edge per dollar
  clears an adaptive bar and the share count clears `minOrderSize` with a buffer;
- holds while the bid is below fair value, sells as soon as the market has caught up or moved
  against the position, and always flattens before the close so every cycle earns volume
  (settled positions earn none);
- **carry mode** (checked first): with 30–90 s left, buys the favorite when its ask is
  0.85–0.96 and the model gives it at least 0.80, then sells into the late run-up once the
  bid reaches 0.99. It bails out if the model drops below 0.70 or the bid falls 0.08 under
  entry. When behind the volume schedule it accepts favorites up to 0.97. Live recordings
  show these late favorites are priced below their win rate (`research/fav_study.py`);
- paces itself against the 1000 USD target: it lowers the edge bar when behind schedule and
  raises it when ahead, so cash is preserved without missing qualification;
- never trades on stale or provisional reference data, never exceeds `min($5, cash)`,
  and answers `HOLD` on any malformed input.

## Contract compliance

| Rule | Where |
|---|---|
| Max $5 fee-inclusive BUY, never above cash | `Strategy._entry` (`cap`) + tests |
| One position / pending BUY | evaluator-owned; bot only buys when flat |
| BTC 5m Up/Down only | only acts on the market in the observation |
| Reply ≤ 200 ms | ~0.2 ms per decision; HTTP test asserts < 200 ms |
| Stale / provisional / future-stamped reference ignored | `Strategy._decide` |
| No entries in final 15 s; flat at crossing | `entry_min_s` > 15, forced exit before close |
| No network, no self-trading, reproducible | stdlib only, deterministic, in-memory state |

## Results

See [`RESULTS.md`](RESULTS.md) for live-replay and synthetic results and why the shipped
parameters were kept. These are development estimates, not official scores.
