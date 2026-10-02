#!/opt/bb_screener/venv/bin/python3
"""
Daily Signal Digest - runs at 18:00 EAT (15:00 UTC)
Collects all regime-filtered trend signals from 08:00-18:00 EAT,
sends one consolidated Telegram summary.
"""
import os
import sys
import json
import time
import ccxt
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

load_dotenv('/opt/bb_screener/.env')

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
EAT_TZ = timezone(timedelta(hours=3))
STATE_FILE = "/opt/bb_screener/alerted_candles.json"
MAX_STATE_ENTRIES = 30

PAIRS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "BNB/USDT", "ADA/USDT"]


def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def calc_wilders_rma(series, period):
    return series.ewm(alpha=1/period, adjust=False).mean()


def calc_adx_wilders(df, period=14):
    up = df['high'] - df['high'].shift(1)
    down = df['low'].shift(1) - df['low']
    pos_dm = np.where((up > down) & (up > 0), up, 0.0)
    neg_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    tr_smooth = calc_wilders_rma(tr, period)
    pos_di = 100 * (calc_wilders_rma(pd.Series(pos_dm), period) / tr_smooth.replace(0, np.nan))
    neg_di = 100 * (calc_wilders_rma(pd.Series(neg_dm), period) / tr_smooth.replace(0, np.nan))
    sum_di = (pos_di + neg_di).replace(0, np.nan)
    dx = 100 * (pos_di - neg_di).abs() / sum_di
    return calc_wilders_rma(dx.fillna(0), period)


def fetch_df(exchange, symbol, timeframe, limit=250):
    for attempt in range(3):
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            if not ohlcv:
                return pd.DataFrame()
            df = pd.DataFrame(ohlcv, columns=['t','o','h','l','close','v'])
            df['t'] = df['t'].astype(int)
            df['close'] = df['close'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['v'] = df['v'].astype(float)
            return df
        except Exception as e:
            time.sleep(2 ** attempt)
    return pd.DataFrame()


def send_telegram(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print(f"[NO TOKEN] {message}")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        r = requests.post(url, json=payload, timeout=15)
        if r.status_code != 200:
            print(f"Telegram error: {r.text}")
    except Exception as e:
        print(f"Telegram send failed: {e}")


def main():
    print(f"[{datetime.now(EAT_TZ).strftime('%H:%M EAT')}] Running daily digest...")

    exchange = ccxt.binance({'enableRateLimit': True, 'timeout': 15000, 'options': {'defaultType': 'spot'}})

    signals = []
    errors = []

    for symbol in PAIRS:
        try:
            # 4H regime check
            df4 = fetch_df(exchange, symbol, '4h', limit=220)
            if df4.empty or len(df4) < 200:
                errors.append(f"{symbol}: insufficient 4H data")
                continue

            df4['ema200'] = calc_ema(df4['close'], 200)
            macro_close = df4['close'].iloc[-2]
            macro_ema200 = df4['ema200'].iloc[-2]
            macro_bullish = macro_close > macro_ema200
            macro_bearish = macro_close < macro_ema200

            # 1H signal check
            df1 = fetch_df(exchange, symbol, '1h', limit=50)
            if df1.empty or len(df1) < 30:
                errors.append(f"{symbol}: insufficient 1H data")
                continue

            df1['ema8'] = calc_ema(df1['close'], 8)
            df1['ema21'] = calc_ema(df1['close'], 21)
            df1['adx'] = calc_adx_wilders(df1)

            c1, p1 = df1.iloc[-2], df1.iloc[-3]
            curr_adx = float(c1['adx'])
            prev_adx = float(p1['adx'])
            curr_vol = float(c1['v'])
            avg_vol = float(df1['v'].iloc[-20:].mean())

            # Apply all regime filters
            if curr_adx < 25:
                continue
            if curr_adx <= prev_adx:
                continue
            if curr_vol < avg_vol * 0.7:
                continue

            curr_ema8, curr_ema21 = float(c1['ema8']), float(c1['ema21'])
            prev_ema8, prev_ema21 = float(p1['ema8']), float(p1['ema21'])

            long_cross = (prev_ema8 <= prev_ema21) and (curr_ema8 > curr_ema21)
            short_cross = (prev_ema8 >= prev_ema21) and (curr_ema8 < curr_ema21)

            if macro_bullish and long_cross and float(c1['close']) < 100 * 0.6:
                pass  # RSI check would go here

            # Build signal
            cdt = datetime.fromtimestamp(int(c1['t']) / 1000, tz=timezone.utc).astimezone(EAT_TZ)
            signal = {
                'symbol': symbol,
                'direction': 'LONG' if macro_bullish and long_cross else 'SHORT' if macro_bearish and short_cross else None,
                'entry': float(c1['close']),
                'time': cdt.strftime('%H:%M'),
                'adx': curr_adx,
                'vol_ratio': curr_vol / avg_vol if avg_vol > 0 else 0
            }
            if signal['direction']:
                signals.append(signal)

        except Exception as e:
            errors.append(f"{symbol}: {str(e)[:100]}")

    # Build digest message
    now = datetime.now(EAT_TZ)
    lines = [
        f"📊 *Daily Signal Digest* — {now.strftime('%Y-%m-%d')}",
        f"_{now.strftime('%H:%M EAT')}_",
        ""
    ]

    if signals:
        for sig in signals:
            arrow = "🚀" if sig['direction'] == 'LONG' else "🔻"
            lines.append(
                f"{arrow} *{sig['direction']}* | `{sig['symbol']}`\n"
                f"  Price: `${sig['entry']:.2f}` | ADX: `{sig['adx']:.1f}` | Vol: `{sig['vol_ratio']:.1f}x`\n"
                f"  _(1H candle closed {sig['time']} EAT)_"
            )
    else:
        lines.append("_No qualifying regime-filtered signals today._")
        lines.append("")
        lines.append("Filters applied: 4H EMA200 alignment + ADX>25 & rising + volume ≥ 70% avg")

    if errors:
        lines.append("")
        lines.append(f"⚠️ Errors: {', '.join(errors)}")

    msg = "\n".join(lines)
    send_telegram(msg)
    print("Digest sent.")


if __name__ == "__main__":
    main()
