#!/usr/bin/env python3
"""
4H SPOT BOT — Binance Spot

Same validated strategy as the futures swing bot, but for spot holdings
(no shorting, no funding, 0.1% taker fee). You hold hours to days.

Strategy: 4H EMA20/50 cross + EMA200 regime + ADX>=20 + RSI 40-80
  Stop 2.0x ATR, Target 4.0x ATR, max hold 60 bars (10 days)

Backtest (2 years, 6 pairs, spot costs):
  84 trades | 53.6% WR | +0.382R exp | +32.1R total | PF 1.60 | MaxDD -4.2R
  95% CI [+0.08, +0.68]  <- excludes zero (significant)
  OOS: 1st half +0.463R, 2nd half +0.301R

LONG-ONLY: spot cannot short.
Signals only. You execute manually.
"""
import os
import sys
import time
import argparse
import signal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nonfiat.common import (setup_logging, spot_exchange, fetch_df, ema, rsi, atr, adx,
                            load_state, save_state, send_telegram, fmt_price, eat_str)

log = setup_logging("spot4h")

STATE_FILE = os.getenv("SPOT4H_STATE", "/opt/bb_screener/spot4h_state.json")

PAIRS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "BNB/USDT", "ADA/USDT"]

POLL_INTERVAL = 900
MAX_STATE_ENTRIES = 50

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

exchange = spot_exchange()


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

    # LONG only — spot cannot short
    cross_up = p["ema20"] <= p["ema50"] and c["ema20"] > c["ema50"]
    if not (c["c"] > c["ema200"] and cross_up):
        return None

    entry = c["c"]
    a = c["atr"]
    if a <= 0:
        return None

    return {
        "symbol": symbol, "direction": "LONG", "entry": float(entry),
        "sl": float(entry - ATR_SL * a), "tp": float(entry + ATR_TP * a),
        "atr": float(a), "rsi": float(c["rsi"]), "adx": float(c["adx"]),
        "regime": "BULL",
        "open_time": open_ms, "time_eat": eat_str(open_ms),
    }


def format_alert(s):
    risk = abs(s["entry"] - s["sl"]) / s["entry"] * 100
    reward = abs(s["tp"] - s["entry"]) / s["entry"] * 100
    return (
        f"🟢 *LONG* — `{s['symbol']}`\n"
        f"`SPOT · 4H SWING`\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"• Regime: `{s['regime']}` | ADX: `{s['adx']:.1f}` | RSI: `{s['rsi']:.1f}`\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"• Entry: `${fmt_price(s['entry'])}`\n"
        f"• Stop: `${fmt_price(s['sl'])}`  (-{risk:.2f}%)\n"
        f"• Target: `${fmt_price(s['tp'])}`  (+{reward:.2f}%)\n"
        f"• R:R `1:{ATR_TP/ATR_SL:.0f}` | Hold ≤ `10d`\n"
        f"• 4H candle: `{s['time_eat']} EAT`"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    log.info("🟢 4H Spot Bot (LONG-ONLY) — EMA20/50 + EMA200 + ADX20 + RSI40-80")
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
                    log.info(f"SIGNAL LONG {sym} @ {s['entry']}")
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
    log.info("4H spot bot stopped cleanly")


if __name__ == "__main__":
    main()
