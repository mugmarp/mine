#!/usr/bin/env python3
"""
INTRADAY LONG BOT — Binance Futures (USDT-M)

Timeframes: 4H regime -> 30m momentum -> 15m entry
Backtest (90 days, 8 pairs): 77 trades | 46.8% WR | +0.120R | PF 1.26 | MaxDD -15.8R
Tightened (vol30>=1.4): 18 trades | 55.6% WR | +0.371R | PF 1.58 | MaxDD -4.5R

Note: this is INTRADAY momentum, not true scalping. Avg hold ~2.4h.
Neither variant is statistically significant yet (CI includes 0).

Signals only. You execute manually.
"""
import os
import sys
import time
import argparse
import signal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nonfiat.common import (setup_logging, futures_exchange, fetch_df, ema, rsi, atr,
                            load_state, save_state, send_telegram, funding_rate,
                            fmt_price, eat_str)

log = setup_logging("intraday_long")

STATE_FILE = os.getenv("INTRADAY_LONG_STATE", "/opt/bb_screener/intraday_long_state.json")

PAIRS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "XRP/USDT:USDT",
    "ADA/USDT:USDT", "DOGE/USDT:USDT", "PENGU/USDT:USDT", "PEOPLE/USDT:USDT",
]

POLL_INTERVAL = 300
MAX_STATE_ENTRIES = 50
LEVERAGE_TAG = 10

# Long-side params (base; set VOL30_MIN=1.4 for the "strong" variant)
RSI_LO, RSI_HI = 45, 75
VOL30_MIN = 0.8
VOL15_MIN = 0.7
ATR_SL, ATR_TP = 1.5, 3.0
MAX_HOLD_HOURS = 48

RUNNING = True


def _sig(signum, frame):
    global RUNNING
    log.info(f"signal {signum} — shutting down")
    RUNNING = False


signal.signal(signal.SIGINT, _sig)
signal.signal(signal.SIGTERM, _sig)

exchange = futures_exchange()


def analyze(symbol, state):
    df4 = fetch_df(exchange, symbol, "4h", 220)
    if df4.empty or len(df4) < 200:
        return None
    df4["ema200"] = ema(df4["c"], 200)
    if not (df4["c"].iloc[-2] > df4["ema200"].iloc[-2]):
        return None  # not bullish regime

    df30 = fetch_df(exchange, symbol, "30m", 100)
    if df30.empty or len(df30) < 30:
        return None
    df30["ema8"] = ema(df30["c"], 8)
    df30["ema21"] = ema(df30["c"], 21)
    df30["rsi"] = rsi(df30["c"])
    df30["vma"] = df30["v"].rolling(20).mean()
    c30 = df30.iloc[-2]

    df15 = fetch_df(exchange, symbol, "15m", 100)
    if df15.empty or len(df15) < 30:
        return None
    df15["ema8"] = ema(df15["c"], 8)
    df15["ema21"] = ema(df15["c"], 21)
    df15["atr"] = atr(df15)
    df15["vma"] = df15["v"].rolling(20).mean()

    c15, p15 = df15.iloc[-2], df15.iloc[-3]
    open_ms = int(c15["t"])
    if open_ms in state.get(symbol, []):
        return None

    if not (RSI_LO <= c30["rsi"] <= RSI_HI):
        return None
    if c30["v"] < c30["vma"] * VOL30_MIN:
        return None
    if c15["v"] < c15["vma"] * VOL15_MIN:
        return None
    if not (c30["ema8"] > c30["ema21"]):
        return None

    cross_up = p15["ema8"] <= p15["ema21"] and c15["ema8"] > c15["ema21"]
    if not cross_up:
        return None

    entry = c15["c"]
    a = c15["atr"]
    if a <= 0:
        return None

    return {
        "symbol": symbol, "direction": "LONG", "entry": float(entry),
        "sl": float(entry - ATR_SL * a), "tp": float(entry + ATR_TP * a),
        "atr": float(a), "rsi30": float(c30["rsi"]),
        "vol30": float(c30["v"] / c30["vma"]), "vol15": float(c15["v"] / c15["vma"]),
        "funding": funding_rate(exchange, symbol),
        "open_time": open_ms, "time_eat": eat_str(open_ms),
    }


def format_alert(s):
    risk = abs(s["entry"] - s["sl"]) / s["entry"] * 100
    reward = abs(s["tp"] - s["entry"]) / s["entry"] * 100
    fund = ""
    if s["funding"] is not None:
        warn = " ⚠️" if abs(s["funding"]) > 0.0003 else ""
        fund = f"\n• Funding: `{s['funding']*100:.4f}%`/8h{warn}"
    return (
        f"🚀 *LONG* — `{s['symbol']}`\n"
        f"`FUTURES · INTRADAY 15m/30m`\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"• 4H: `BULL` | RSI30: `{s['rsi30']:.1f}`\n"
        f"• Vol: `30m {s['vol30']:.2f}x` / `15m {s['vol15']:.2f}x`\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"• Entry: `${fmt_price(s['entry'])}`\n"
        f"• Stop: `${fmt_price(s['sl'])}`  (-{risk:.2f}%)\n"
        f"• Target: `${fmt_price(s['tp'])}`  (+{reward:.2f}%)\n"
        f"• R:R `1:{ATR_TP/ATR_SL:.0f}` | Leverage ≤ `{LEVERAGE_TAG}x` | Hold ≤ `48h`{fund}\n"
        f"• 15m candle: `{s['time_eat']} EAT`"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    log.info("🟢 Intraday LONG Bot (FUTURES) — 4H bull + 30m momentum + 15m cross")
    log.info(f"Pairs: {len(PAIRS)} | vol30>={VOL30_MIN} vol15>={VOL15_MIN}")

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
    log.info("Intraday long bot stopped cleanly")


if __name__ == "__main__":
    main()
