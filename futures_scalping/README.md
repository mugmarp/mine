# Futures Scalping Bot — Binance USDT-M Perpetuals

**Timeframes:** 4H regime → 30m momentum → 15m entry
**Direction:** LONG + SHORT
**Output:** Telegram signals only (manual execution)

## Strategy Logic

| Layer | Timeframe | Rule |
|-------|-----------|------|
| Regime | 4H | Close vs EMA200 (bull/bear) |
| Momentum | 30m | EMA8 vs EMA21 aligned with regime; RSI 45–75; volume ≥ 0.8× 20-period avg |
| Entry | 15m | EMA8/EMA21 cross in regime direction; volume ≥ 0.7× avg |
| Risk | — | Stop = 1.5× ATR(14), Target = 3.0× ATR(14), max hold 48h |

## Backtest Results (90 days, 8 pairs, realistic costs)

```
91 trades | 49.5% WR | +0.196R expectancy | PF 1.31 | MaxDD -13.5R
Gross +0.484R → Net +0.196R  (costs eat 59%)
```

| Scope | n | WR | ExpR | Total R | PF | MaxDD |
|-------|---|-----|------|---------|-----|-------|
| COMBINED | 91 | 49.5% | +0.196 | +17.9 | 1.31 | -13.5R |
| LONG only | 77 | 46.8% | +0.120 | +9.2 | 1.26 | -15.8R |
| SHORT only | 14 | 64.3% | +0.616 | +8.6 | 1.71 | -2.2R |

## Robustness Assessment

**Strengths:**
- Parameter sweep: 60/60 configs positive expectancy (not a knife-edge)
- Out-of-sample (2nd 45d): +0.234R vs in-sample +0.160R — holds up
- Short side notably stronger (64% WR) than long

**Weaknesses:**
- Bootstrap 95% CI: [-0.111R, +0.512R] — **straddles zero, NOT statistically significant**
- t-stat 1.23 (needs >1.98 for p<0.05)
- Costs consume 59% of gross edge
- At pessimistic costs (0.08% fee + 0.05% slip) expectancy goes **negative**
- BTC (-13.2R) and DOGE (-10.1R) were net negative — dropping them → +0.625R expectancy, PF 1.92

## Deployment

```bash
# Dry run
python bot.py --dry-run --once

# Live (systemd or cron every 5 min)
python bot.py
```

## Risk Notes

- **Leverage: 5–10x max.** At 10x, 0.143% round-trip cost = 1.43% of margin per trade.
- **Funding:** avoid holding across 00:00 / 08:00 / 16:00 UTC when funding > 0.03%/8h.
- **Sample size:** ~1 trade/day. To confirm +0.20R at 95% confidence needs ~218 trades (~7 months).
- Use **limit orders** where possible — taker fees are what erode the edge.
