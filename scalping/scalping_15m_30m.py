#!/usr/bin/env python3
"""
Pseudo-Scalping 15m/30m Strategy — BTC/ETH/SOL/XRP
Regime-filtered: 4H trend alignment + 15m momentum + volume confirmation
Designed for 200ms VPS latency (not HFT)
"""
import os
import sys
import time
import json
import ccxt
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
EAT_TZ = timezone(timedelta(hours=3))

# Liquid pairs only — spreads <0.5bp on Binance spot
PAIRS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]
SCAN_INTERVAL = 300  # 5 min scans

def calc_ema(s, p): return s.ewm(span=p, adjust=False).mean()
def calc_rma(s, p): return s.ewm(alpha=1/p, adjust=False).mean()
def calc_rsi(s, p=14):
    d = s.diff()
    g = d.clip(lower=0); l = -d.clip(upper=0)
    return (100 - 100/(1 + calc_rma(g,p)/calc_rma(l,p).replace(0,np.nan))).fillna(50)

def calc_atr(df, p=14):
    tr = pd.concat([df.h-df.l, (df.h-df.c.shift()).abs(), (df.l-df.c.shift()).abs()], axis=1).max(axis=1)
    return calc_rma(tr, p)

def fetch_df(ex, sym, tf, lim=200):
    for _ in range(3):
        try:
            o = ex.fetch_ohlcv(sym, timeframe=tf, limit=lim)
            if not o: return pd.DataFrame()
            df = pd.DataFrame(o, columns=['t','o','h','l','c','v'])
            for c in ['o','h','l','c','v']: df[c] = df[c].astype(float)
            return df
        except: time.sleep(2)
    return pd.DataFrame()

def regime_4h(ex, sym):
    df = fetch_df(ex, sym, '4h', 220)
    if len(df) < 200: return None
    df['ema200'] = calc_ema(df.c, 200)
    c = df.iloc[-2]
    return 'bull' if c.c > c.ema200 else 'bear'

def scan_15m_30m(ex, sym):
    """Return signal dict or None"""
    # 30m: momentum & structure
    df30 = fetch_df(ex, sym, '30m', 100)
    if len(df30) < 50: return None
    df30['ema8'] = calc_ema(df30.c, 8)
    df30['ema21'] = calc_ema(df30.c, 21)
    df30['rsi'] = calc_rsi(df30.c)
    df30['atr'] = calc_atr(df30)
    df30['vol_ma'] = df30.v.rolling(20).mean()

    # 15m: entry timing
    df15 = fetch_df(ex, sym, '15m', 100)
    if len(df15) < 30: return None
    df15['ema8'] = calc_ema(df15.c, 8)
    df15['ema21'] = calc_ema(df15.c, 21)
    df15['rsi'] = calc_rsi(df15.c)
    df15['atr'] = calc_atr(df15)
    df15['vol_ma'] = df15.v.rolling(20).mean()

    # 4H regime
    regime = regime_4h(ex, sym)
    if not regime: return None

    c30, p30 = df30.iloc[-2], df30.iloc[-3]
    c15, p15 = df15.iloc[-2], df15.iloc[-1]  # use current forming 15m for entry timing

    # Filters
    if float(c30.rsi) < 45 or float(c30.rsi) > 75: return None
    if float(c30.v) < float(c30.vol_ma) * 0.8: return None
    if float(c15.v) < float(c15.vol_ma) * 0.7: return None

    # 30m EMA alignment with regime
    ema8_30, ema21_30 = float(c30.ema8), float(c30.ema21)
    if regime == 'bull' and not (ema8_30 > ema21_30): return None
    if regime == 'bear' and not (ema8_30 < ema21_30): return None

    # 15m cross for entry
    cross_up = float(p15.ema8) <= float(p15.ema21) and float(c15.ema8) > float(c15.ema21)
    cross_down = float(p15.ema8) >= float(p15.ema21) and float(c15.ema8) < float(c15.ema21)

    if regime == 'bull' and cross_up:
        atr = float(c15.atr); px = float(c15.c)
        return {'sym': sym, 'dir': 'LONG', 'px': px, 'sl': px - 1.5*atr, 'tp': px + 3*atr,
                'tf': '15m/30m', 'regime': regime}
    if regime == 'bear' and cross_down:
        atr = float(c15.atr); px = float(c15.c)
        return {'sym': sym, 'dir': 'SHORT', 'px': px, 'sl': px + 1.5*atr, 'tp': px - 3*atr,
                'tf': '15m/30m', 'regime': regime}
    return None

def send_tg(msg):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: return
    requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
        json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"}, timeout=10)

def main():
    ex = ccxt.binance({'enableRateLimit': True, 'timeout': 10000, 'options': {'defaultType': 'spot'}})
    print(f"[{datetime.now(EAT_TZ).strftime('%H:%M EAT')}] Scalping scan...")
    for sym in PAIRS:
        sig = scan_15m_30m(ex, sym)
        if sig:
            msg = f"{'🚀' if sig['dir']=='LONG' else '🔻'} *{sig['dir']}* `{sig['sym']}` | {sig['tf']}\nPx: `{sig['px']:.4f}` | SL: `{sig['sl']:.4f}` | TP: `{sig['tp']:.4f}`"
            print(msg)
            send_tg(msg)

if __name__ == "__main__":
    main()
