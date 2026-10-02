#!/usr/bin/env python3
"""
4H SWING BOT — Binance Futures (USDT-M Perpetuals)

Strategy (validated 2yr, statistically significant):
  4H EMA20/50 cross + EMA200 regime filter + ADX>=20 + RSI 40-80
  Stop 2.0x ATR(14), Target 4.0x ATR(14), max hold 60 bars (10 days)

Backtest (2 years, 9 pairs, realistic futures costs):
  127 trades | 49.6% WR | +0.347R exp | +44.1R total | PF 1.38 | MaxDD -5.2R
  95% CI [+0.10, +0.59]  <- excludes zero (significant)
  OOS: 1st half +0.365R, 2nd half +0.329R

Signals only. You execute manually.
"""
import os
import sys
import time
import argparse
import signal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nonfiat.common import (setup_logging, futures_exchange, fetch_df, ema, rsi, atr, adx,
                            load_state, save_state, send_telegram, funding_rate,
                            fmt_price, eat_str)

log = setup_logging("swing4h")

STATE_FILE = os.getenv("SWING4H_STATE", "/opt/bb_screener/swing4h_state.json")

PAIRS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "XRP/USDT:USDT",
    "BNB/USDT:USDT", "ADA/USDT:USDT", "DOGE/USDT:USDT",
    "PENGU/USDT:USDT", "PEOPLE/USDT:USDT",
]

POLL_INTERVAL = 900      # scan every 15 min
MAX_STATE_ENTRIES = 50
LEVERAGE_TAG = 5         # 4H swing holds days — keep leverage low

# Validated params
EMA_FAST, EMA_SLOW, EMA_REGIME = 20, 50, 200
ADX_MIN = 20
RSI_LO, RSI_HI = 40, 80
ATR_SL, ATR_TP = 2.0, 4.0
MAX_HOLD_BARS = 60

RUNNING = True


def _sig(signum, frame):
    global RUNNING
    log.info(f"signal {signum} — shutting down")
    RUNNING = False


signal.signal(signal.SIGINT, _sig)
signal.signal(signal.SIGTERM, _sig)

exchange = futures_exchange()


def analyze(symbol, state):
    df = fetch_df(exchange, symbol, "4h", limit=300)
    if df.empty or len(df) < 210:
        return None

    df["ema20"] = ema(df["c"], EMA_FAST)
    df["ema50"] = ema(df["c"], EMA_SLOW)
    df["ema200"] = ema(df["c"], EMA_REGIME)
    df["rsi"] = rsi(df["c"])
    df["adx"] = adx(df)
    df["atr"] = atr(df)

    c, p = df.iloc[-2], df.iloc[-3]
    open_ms = int(c["t"])

    if open_ms in state.get(symbol, []):
        return None

    if not (c["adx"] >= ADX_MIN):
        return None
    if not (RSI_LO <= c["rsi"] <= RSI_HI):
        return None

    cross_up = p["ema20"] <= p["ema50"] and c["ema20"] > c["ema50"]
    cross_dn = p["ema20"] >= p["ema50"] and c["ema20"] < c["ema50"]

    if c["c"] > c["ema200"] and cross_up:
        direction = "LONG"
    elif c["c"] < c["ema200"] and cross_dn:
        direction = "SHORT"
    else:
        return None

    entry = c["c"]
    a = c["atr"]
    if a <= 0:
        return None

    if direction == "LONG":
        sl, tp = entry - ATR_SL * a, entry + ATR_TP * a
    else:
        sl, tp = entry + ATR_SL * a, entry - ATR_TP * a

    return {
        "symbol": symbol, "direction": direction, "entry": float(entry),
        "sl": float(sl), "tp": float(tp), "atr": float(a),
        "rsi": float(c["rsi"]), "adx": float(c["adx"]),
        "regime": "BULL" if c["c"] > c["ema200"] else "BEAR",
        "funding": funding_rate(exchange, symbol),
        "open_time": open_ms,
        "time_eat": eat_str(open_ms),
    }


def format_alert(s):
    arrow = "🟢" if s["direction"] == "LONG" else "🔴"
    risk = abs(s["entry"] - s["sl"]) / s["entry"] * 100
    reward = abs(s["tp"] - s["entry"]) / s["entry"] * 100
    fund = ""
    if s["funding"] is not None:
        warn = " ⚠️" if abs(s["funding"]) > 0.0003 else ""
        fund = f"\n• Funding: `{s['funding']*100:.4f}%`/8h{warn}"

    return (
        f"{arrow} *{s['direction']}* — `{s['symbol']}`\n"
        f"`FUTURES · 4H SWING`\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"• Regime: `{s['regime']}` | ADX: `{s['adx']:.1f}` | RSI: `{s['rsi']:.1f}`\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"• Entry: `${fmt_price(s['entry'])}`\n"
        f"• Stop: `${fmt_price(s['sl'])}`  (-{risk:.2f}%)\n"
        f"• Target: `${fmt_price(s['tp'])}`  (+{reward:.2f}%)\n"
        f"• R:R `1:{ATR_TP/ATR_SL:.0f}` | Leverage ≤ `{LEVERAGE_TAG}x` | Hold ≤ `10d`{fund}\n"
        f"• 4H candle: `{s['time_eat']} EAT`"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    log.info("🟢 4H Swing Bot (FUTURES) — EMA20/50 + EMA200 + ADX20 + RSI40-80")
    log.info(f"Pairs: {len(PAIRS)} | SL {ATR_SL}xATR | TP {ATR_TP}xATR | poll {POLL_INTERVAL}s")

    state = load_state(STATE_FILE)
    while RUNNING:
        changed = False
        for sym in PAIRS:
            if not RUNNING:
                break
            try:
                s = analyze(sym, state)
                if s:
                    log.info(f"SIGNAL {s['direction']} {sym} @ {s['entry']}")
                    send_telegram(format_alert(s), dry_run=args.dry_run)
                    state.setdefault(sym, []).append(s["open_time"])
                    changed = True
            except Exception as e:
                log.error(f"analyze {sym}: {e}")
        if changed and not args.dry_run:
            save_state(STATE_FILE, state, MAX_STATE_ENTRIES)
        if args.once:
            break
        for _ in range(POLL_INTERVAL):
            if not RUNNING:
                break
            time.sleep(1)
    log.info("4H swing bot stopped cleanly")


if __name__ == "__main__":
    main()
