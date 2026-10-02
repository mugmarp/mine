#!/usr/bin/env python3
"""
Futures Scalping Signal Bot — Binance USDT-M
Timeframes: 4H regime / 30m momentum / 15m entry
Both LONG and SHORT. Signals only — you execute manually.

Backtest (90 days, 8 pairs, realistic costs):
  91 trades | 49.5% WR | +0.196R expectancy | PF 1.31 | MaxDD -13.5R
  Gross +0.484R -> Net +0.196R (costs eat 59%)
  Parameter sweep: 60/60 configs positive (edge not a knife-edge)
  Bootstrap 95% CI [-0.11, +0.51] — NOT yet statistically significant

LEVERAGE: 5-10x max. At 10x, 0.143% round-trip cost = 1.43% of margin.
Funding: avoid holding across 00:00/08:00/16:00 UTC when funding > 0.03%.
"""
import os
import sys
import time
import json
import argparse
import logging
from datetime import datetime, timezone, timedelta

import ccxt
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
STATE_FILE = os.getenv("FUTURES_SCALP_STATE", "/opt/bb_screener/futures_scalp_state.json")

# Binance USDT-M perpetual symbols
PAIRS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "XRP/USDT:USDT",
    "ADA/USDT:USDT", "DOGE/USDT:USDT", "PENGU/USDT:USDT", "PEOPLE/USDT:USDT",
]
POLL_INTERVAL = 300          # scan every 5 min
MAX_STATE_ENTRIES = 50
LEVERAGE_TAG = 10            # recommended max in alerts

# Strategy params (from backtest base case)
RSI_LO, RSI_HI = 45, 75
VOL30_MIN = 0.8
VOL15_MIN = 0.7
ATR_SL_MULT = 1.5
ATR_TP_MULT = 3.0
MAX_HOLD_HOURS = 48

EAT_TZ = timezone(timedelta(hours=3))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("FuturesScalp")

RUNNING = True


def _sig(signum, frame):
    global RUNNING
    logger.info(f"Signal {signum} — shutting down gracefully")
    RUNNING = False


import signal
signal.signal(signal.SIGINT, _sig)
signal.signal(signal.SIGTERM, _sig)


def get_exchange():
    return ccxt.binance({
        'enableRateLimit': True,
        'timeout': 20000,
        'options': {'defaultType': 'future'},
    })


exchange = get_exchange()


def fetch_df(symbol, timeframe, limit=250):
    for attempt in range(1, 4):
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            if not ohlcv:
                return pd.DataFrame()
            df = pd.DataFrame(ohlcv, columns=['t', 'o', 'h', 'l', 'c', 'v'])
            for c in ['o', 'h', 'l', 'c', 'v']:
                df[c] = df[c].astype(float)
            df['t'] = df['t'].astype(int)
            return df
        except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
            wait = 2 ** attempt
            logger.warning(f"Network {symbol} {attempt}/3, retry in {wait}s: {e}")
            time.sleep(wait)
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error {symbol}: {e}")
            break
        except Exception as e:
            logger.error(f"Unexpected {symbol}: {e}")
            break
    return pd.DataFrame()


def calc_ema(s, p):
    return s.ewm(span=p, adjust=False).mean()


def calc_rma(s, p):
    return s.ewm(alpha=1/p, adjust=False).mean()


def calc_rsi(s, p=14):
    d = s.diff()
    g = d.clip(lower=0)
    l = -d.clip(upper=0)
    rs = calc_rma(g, p) / calc_rma(l, p).replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def calc_atr(df, p=14):
    tr = pd.concat([df['h'] - df['l'],
                    (df['h'] - df['c'].shift(1)).abs(),
                    (df['l'] - df['c'].shift(1)).abs()], axis=1).max(axis=1)
    return calc_rma(tr, p)


def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                return {k: [int(x) for x in v] for k, v in json.load(f).items()}
        except Exception as e:
            logger.error(f"state load: {e}")
    return {}


def save_state(state):
    try:
        pruned = {k: sorted(v)[-MAX_STATE_ENTRIES:] for k, v in state.items()}
        tmp = f"{STATE_FILE}.tmp"
        with open(tmp, 'w') as f:
            json.dump(pruned, f, indent=2)
        os.replace(tmp, STATE_FILE)
    except Exception as e:
        logger.error(f"state save: {e}")


def send_telegram(msg):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram creds missing")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    for attempt in range(1, 4):
        try:
            r = requests.post(url, json={
                "chat_id": TELEGRAM_CHAT_ID, "text": msg,
                "parse_mode": "Markdown", "disable_web_page_preview": True
            }, timeout=15)
            if r.status_code == 200:
                return
            logger.warning(f"Telegram {r.status_code} attempt {attempt}/3")
        except Exception as e:
            logger.error(f"Telegram fail {attempt}/3: {e}")
        time.sleep(2)


def funding_rate(symbol):
    """Current funding rate (fraction). Returns None on failure."""
    try:
        fr = exchange.fetch_funding_rate(symbol.replace(':USDT', ''))
        return fr.get('fundingRate')
    except Exception:
        return None


def analyze(symbol, state):
    """Returns signal dict or None."""
    # 4H regime
    df4 = fetch_df(symbol, '4h', 220)
    if df4.empty or len(df4) < 200:
        return None
    df4['ema200'] = calc_ema(df4['c'], 200)
    macro_close = df4['c'].iloc[-2]
    macro_ema200 = df4['ema200'].iloc[-2]
    bull = macro_close > macro_ema200
    bear = macro_close < macro_ema200

    # 30m momentum
    df30 = fetch_df(symbol, '30m', 100)
    if df30.empty or len(df30) < 30:
        return None
    df30['ema8'] = calc_ema(df30['c'], 8)
    df30['ema21'] = calc_ema(df30['c'], 21)
    df30['rsi'] = calc_rsi(df30['c'])
    df30['vma'] = df30['v'].rolling(20).mean()
    c30 = df30.iloc[-2]

    # 15m entry
    df15 = fetch_df(symbol, '15m', 100)
    if df15.empty or len(df15) < 30:
        return None
    df15['ema8'] = calc_ema(df15['c'], 8)
    df15['ema21'] = calc_ema(df15['c'], 21)
    df15['atr'] = calc_atr(df15)
    df15['vma'] = df15['v'].rolling(20).mean()

    c15, p15 = df15.iloc[-2], df15.iloc[-3]
    open_time_ms = int(c15['t'])

    if open_time_ms in state.get(symbol, []):
        return None

    # --- filters (evidence-based) ---
    if not (RSI_LO <= c30['rsi'] <= RSI_HI):
        return None
    if c30['v'] < c30['vma'] * VOL30_MIN:
        return None
    if c15['v'] < c15['vma'] * VOL15_MIN:
        return None

    cross_up = p15['ema8'] <= p15['ema21'] and c15['ema8'] > c15['ema21']
    cross_dn = p15['ema8'] >= p15['ema21'] and c15['ema8'] < c15['ema21']

    direction = None
    if bull and cross_up and c30['ema8'] > c30['ema21']:
        direction = 'LONG'
    elif bear and cross_dn and c30['ema8'] < c30['ema21']:
        direction = 'SHORT'

    if not direction:
        return None

    entry = c15['c']
    atr = c15['atr']
    if not np.isfinite(atr) or atr <= 0:
        return None

    if direction == 'LONG':
        sl = entry - ATR_SL_MULT * atr
        tp = entry + ATR_TP_MULT * atr
    else:
        sl = entry + ATR_SL_MULT * atr
        tp = entry - ATR_TP_MULT * atr

    fr = funding_rate(symbol)
    regime = 'BULL' if bull else 'BEAR'

    return {
        'symbol': symbol, 'direction': direction, 'entry': float(entry),
        'sl': float(sl), 'tp': float(tp), 'atr': float(atr),
        'rsi30': float(c30['rsi']), 'vol30_r': float(c30['v'] / c30['vma']),
        'vol15_r': float(c15['v'] / c15['vma']), 'regime': regime,
        'funding': fr, 'open_time': open_time_ms,
        'time_eat': datetime.fromtimestamp(open_time_ms / 1000, tz=timezone.utc)
                             .astimezone(EAT_TZ).strftime('%Y-%m-%d %H:%M'),
    }


def format_alert(s):
    arrow = "🚀" if s['direction'] == 'LONG' else "🔻"
    risk_pct = abs(s['entry'] - s['sl']) / s['entry'] * 100
    reward_pct = abs(s['tp'] - s['entry']) / s['entry'] * 100
    fund_line = ""
    if s['funding'] is not None:
        fund_pct = s['funding'] * 100
        warn = " ⚠️ high" if abs(s['funding']) > 0.0003 else ""
        fund_line = f"\n• Funding: `{fund_pct:.4f}%` /8h{warn}"

    return (
        f"{arrow} *{s['direction']}* — `{s['symbol']}`\n"
        f"━━━━━━━━━━━━━━━\n"
        f"• Regime: `4H {s['regime']}` | RSI30: `{s['rsi30']:.1f}`\n"
        f"• Volume: `30m {s['vol30_r']:.2f}x` / `15m {s['vol15_r']:.2f}x`\n"
        f"━━━━━━━━━━━━━━━\n"
        f"• Entry: `${s['entry']:.6g}`\n"
        f"• Stop: `${s['sl']:.6g}`  ({risk_pct:.2f}%)\n"
        f"• Target: `${s['tp']:.6g}`  ({reward_pct:.2f}%)\n"
        f"• R:R = `1:{ATR_TP_MULT/ATR_SL_MULT:.1f}` | Leverage ≤ `{LEVERAGE_TAG}x`{fund_line}\n"
        f"• Candle: `{s['time_eat']} EAT`"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--once', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    logger.info("🟢 Futures Scalp Bot — 4H/30m/15m regime-filtered")
    logger.info(f"Params: RSI {RSI_LO}-{RSI_HI} | vol30≥{VOL30_MIN} | vol15≥{VOL15_MIN} | "
                f"SL {ATR_SL_MULT}xATR | TP {ATR_TP_MULT}xATR")
    logger.info(f"Pairs: {len(PAIRS)} USDT-M perpetuals")

    state = load_state()
    while RUNNING:
        changed = False
        for sym in PAIRS:
            if not RUNNING:
                break
            try:
                s = analyze(sym, state)
                if s:
                    msg = format_alert(s)
                    logger.info(f"SIGNAL {s['direction']} {sym} @ {s['entry']:.6g}")
                    if args.dry_run:
                        logger.info(f"[DRY RUN]\n{msg}")
                    else:
                        send_telegram(msg)
                    state.setdefault(sym, []).append(s['open_time'])
                    changed = True
            except Exception as e:
                logger.error(f"analyze {sym}: {e}")
        if changed and not args.dry_run:
            save_state(state)
        if args.once:
            break
        for _ in range(POLL_INTERVAL):
            if not RUNNING:
                break
            time.sleep(1)
    logger.info("Futures scalp bot stopped cleanly")


if __name__ == '__main__':
    main()
