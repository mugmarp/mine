# Non-Fiat — Evidence-Based Crypto Signal Bots

Signal-only bots for Binance. You execute trades manually. All signals go to
**one Telegram bot**, labelled by source.

## Bots

| File | Market | Direction | Timeframes | Signal rate |
|------|--------|-----------|------------|-------------|
| `nonfiat/swing_4h.py` | Futures | Long + Short | 4H | ~1/pair/2mo |
| `nonfiat/spot_4h.py` | Spot | Long only | 4H | ~1/pair/2mo |
| `nonfiat/intraday_long.py` | Futures | Long | 4H→30m→15m | ~1/day |
| `nonfiat/intraday_short.py` | Futures | Short | 4H→30m→15m | ~0.2/day |

## Strategy: the 4H core (validated)

```
EMA20/50 cross + EMA200 regime filter + ADX >= 20 + RSI 40-80
Stop 2.0x ATR(14)  |  Target 4.0x ATR(14)  |  Max hold 60 bars (10 days)
```

**Backtest — 2 years, realistic costs:**

| Market | n | WR | ExpR | Total R | PF | MaxDD | 95% CI | P(exp>0) |
|--------|---|-----|------|---------|-----|-------|--------|----------|
| Futures | 127 | 49.6% | +0.347R | +44.1 | 1.38 | -5.2R | [+0.10, +0.59] | 100% |
| Spot | 84 | 53.6% | +0.382R | +32.1 | 1.60 | -4.2R | [+0.08, +0.68] | 99% |

**Out-of-sample** (median split):
- Futures: 1st half +0.365R → 2nd half +0.329R
- Spot: 1st half +0.463R → 2nd half +0.301R

Both CIs **exclude zero** — the only strategy family tested that achieved this.

## Strategy: intraday (not yet significant)

4H regime → 30m EMA8/21 momentum + RSI + volume → 15m EMA cross entry.
Stop 1.5x ATR, Target 3.0x ATR, max hold 48h. Avg hold ~2.4h.

| Setup | n | WR | ExpR | PF | MaxDD | CI |
|-------|---|-----|------|-----|-------|-----|
| Long base | 77 | 46.8% | +0.120R | 1.26 | -15.8R | includes 0 |
| Long strong (vol30≥1.4) | 18 | 55.6% | +0.371R | 1.58 | -4.5R | includes 0 |
| Short | 14 | 64.3% | +0.616R | 1.71 | -2.2R | includes 0 |

Costs consume 59% of the gross edge. Treat as experimental.

## Strategies tested and REJECTED

| Strategy | Why rejected |
|----------|--------------|
| MACD cross | Negative expectancy both markets |
| Bollinger mean-reversion | PF ~1.0, huge drawdown |
| Donchian 20/50 breakout | 1000+ trades, 40% WR, -50R drawdowns |
| EMA cross + ADX25 rising + volume | Only 17 trades in 2 years — unusable |

## Deployment

Runs via cron, sequentially (each bot needs ~300MB; VPS has 842MB total).

```bash
*/5 * * * * /bin/bash /home/azureuser/BotScripts/run_cycle.sh
```

`run_cycle.sh` uses `flock` to prevent overlap and runs:
- intraday bots every cycle
- 4H bots every 15 min

### Manual run

```bash
PY=/opt/bb_screener/venv/bin/python3
$PY nonfiat/swing_4h.py --dry-run --once
$PY nonfiat/swing_4h.py            # live
```

## Analysis scripts

`nonfiat/analysis/` contains the backtests that produced these numbers:
- `strategy_search_4h.py` — compared 7 strategy families on 4H
- `validate_winner.py` — OOS, long/short split, per-symbol, sensitivity
- `futures_scalp_backtest.py` + `robustness_analysis.py` — intraday tests
- `split_and_swing_analysis.py` — directional splits

## Risk notes

- **Signals only.** No orders are placed.
- **Leverage:** ≤5x for 4H swing (multi-day holds), ≤10x for intraday.
- **Funding:** avoid holding across 00:00/08:00/16:00 UTC when funding >0.03%/8h.
- **Fees matter:** the 4H edge survives spot's 0.1% taker fee; the intraday edge
  does not survive pessimistic costs.
- **Sample sizes** are modest (84–127 trades). Re-evaluate after 3 months live.
