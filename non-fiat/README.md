# Non-Fiat Cryptocurrency Trading Bots
# Regime-filtered trend following signal generators for Binance futures/spot

## Active Bots
- **trend_screener.py** - 4H EMA200 regime filter + 1H EMA8/EMA21 cross + ADX(>25, rising) + volume confirmation
- **futures_30m_screener.py** - Legacy 30m futures EMA cross screener (deprecated)

## Configuration
Copy `.env.example` to `.env` and fill in your Telegram bot token and chat ID.

## Usage
```bash
# Dry run (logs only)
python trend_screener.py --dry-run --once

# Live immediate alerts
python trend_screener.py

# Daily digest mode (collects signals 08:00-18:00 EAT, sends one summary at 18:00 EAT)
python trend_screener.py --digest
```

## Systemd
```bash
systemctl enable trend_screener
systemctl start trend_screener
journalctl -u trend_screener -f
```
