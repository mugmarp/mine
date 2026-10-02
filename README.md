# Mine — Trading Strategy Repository

## Folders

### `non-fiat/`
Main production bots:
- `trend_screener.py` — 4H EMA200 regime + 1H EMA8/21 cross + ADX>25&rising + volume
- `daily_digest.py` — Cron at 18:00 EAT, single consolidated Telegram alert
- `futures_30m_screener.py` — Legacy reference

### `scalping/`
Pseudo-scalping 15m/30m for BTC/ETH/SOL/XRP:
- 4H regime filter (EMA200)
- 30m momentum (EMA8>21, RSI 45-75, volume)
- 15m entry timing (EMA8/21 cross)
- ATR-based SL/TP (1.5x/3x)
- Designed for 200ms VPS latency

### `spot_robust/`
Extensible Binance Spot analysis framework:
- `core.py` — Data fetching, module registry, Signal dataclass
- Built-in modules: correlation (BTC lead/lag), structure (breakouts), volume_profile (VWAP), basis (placeholder)
- Register custom modules: `sr.register_module('name', fn)`

## Deploy
```bash
# Daily digest (18:00 EAT)
0 15 * * * /opt/bb_screener/venv/bin/python /home/azureuser/BotScripts/daily_digest.py

# Scalping scan (every 5 min)
*/5 * * * * /opt/bb_screener/venv/bin/python /home/azureuser/BotScripts/scalping/scalping_15m_30m.py
```
