#!/usr/bin/env python3
"""
Shared utilities for Non-Fiat signal bots.
Indicators, Telegram delivery, state persistence, exchange helpers.
"""
import os
import sys
import time
import json
import logging
from datetime import datetime, timezone, timedelta

import ccxt
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Cron runs with an arbitrary cwd, so load_dotenv() above may not find the
# Telegram credentials. Search the known locations explicitly.
for _candidate in (
    os.getenv("NONFIAT_ENV", ""),
    "/opt/bb_screener/.env",
    os.path.expanduser("~/.env"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
):
    if _candidate and os.path.exists(_candidate):
        load_dotenv(_candidate, override=False)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

EAT_TZ = timezone(timedelta(hours=3))

logger = logging.getLogger("nonfiat")


def setup_logging(name):
    global logger
    logger = logging.getLogger(name)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logger


# ---------------- Exchanges ----------------

def spot_exchange():
    return ccxt.binance({"enableRateLimit": True, "timeout": 20000,
                         "options": {"defaultType": "spot"}})


def futures_exchange():
    return ccxt.binance({"enableRateLimit": True, "timeout": 20000,
                         "options": {"defaultType": "future"}})


def fetch_df(exchange, symbol, timeframe, limit=300, retries=3):
    """Fetch OHLCV with backoff. Returns DataFrame with t,o,h,l,c,v."""
    for attempt in range(1, retries + 1):
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            if not ohlcv:
                return pd.DataFrame()
            df = pd.DataFrame(ohlcv, columns=["t", "o", "h", "l", "c", "v"])
            for col in ["o", "h", "l", "c", "v"]:
                df[col] = df[col].astype(float)
            df["t"] = df["t"].astype(int)
            return df
        except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
            wait = 2 ** attempt
            logger.warning(f"{symbol} {timeframe} network {attempt}/{retries}, retry {wait}s: {e}")
            time.sleep(wait)
        except ccxt.ExchangeError as e:
            logger.error(f"{symbol} {timeframe} exchange error: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"{symbol} {timeframe} unexpected: {e}")
            return pd.DataFrame()
    return pd.DataFrame()


# ---------------- Indicators ----------------

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def rma(series, period):
    return series.ewm(alpha=1 / period, adjust=False).mean()


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = rma(gain, period) / rma(loss, period).replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def atr(df, period=14):
    tr = pd.concat([
        df["h"] - df["l"],
        (df["h"] - df["c"].shift(1)).abs(),
        (df["l"] - df["c"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    return rma(tr, period)


def adx(df, period=14):
    up = df["h"] - df["h"].shift(1)
    down = df["l"].shift(1) - df["l"]
    pos_dm = np.where((up > down) & (up > 0), up, 0.0)
    neg_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = pd.concat([
        df["h"] - df["l"],
        (df["h"] - df["c"].shift(1)).abs(),
        (df["l"] - df["c"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    tr_s = rma(tr, period)
    pos_di = 100 * rma(pd.Series(pos_dm, index=df.index), period) / tr_s.replace(0, np.nan)
    neg_di = 100 * rma(pd.Series(neg_dm, index=df.index), period) / tr_s.replace(0, np.nan)
    dx = 100 * (pos_di - neg_di).abs() / (pos_di + neg_di).replace(0, np.nan)
    return rma(dx.fillna(0), period)


# ---------------- State ----------------

def load_state(path):
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
            return {k: [int(x) for x in v] for k, v in data.items()}
        except Exception as e:
            logger.error(f"state load {path}: {e}")
    return {}


def save_state(path, state, max_entries=50):
    try:
        pruned = {k: sorted(v)[-max_entries:] for k, v in state.items()}
        tmp = f"{path}.tmp"
        with open(tmp, "w") as f:
            json.dump(pruned, f, indent=2)
        os.replace(tmp, path)
    except Exception as e:
        logger.error(f"state save {path}: {e}")


# ---------------- Telegram ----------------

def send_telegram(message, dry_run=False):
    """Send to the single shared Telegram bot. Returns True on success."""
    if dry_run:
        logger.info(f"[DRY RUN]\n{message}")
        return True
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram credentials missing — alert not sent")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    }
    for attempt in range(1, 4):
        try:
            r = requests.post(url, json=payload, timeout=15)
            if r.status_code == 200:
                return True
            logger.warning(f"Telegram {r.status_code} attempt {attempt}/3: {r.text[:150]}")
        except Exception as e:
            logger.error(f"Telegram fail attempt {attempt}/3: {e}")
        time.sleep(2)
    return False


def funding_rate(futures_ex, symbol):
    """Current funding rate as fraction, or None."""
    try:
        fr = futures_ex.fetch_funding_rate(symbol.replace(":USDT", ""))
        return fr.get("fundingRate")
    except Exception:
        return None


def fmt_price(x):
    if x >= 1000:
        return f"{x:,.2f}"
    if x >= 1:
        return f"{x:.4f}"
    if x >= 0.01:
        return f"{x:.5f}"
    return f"{x:.8f}"


def eat_str(ms):
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).astimezone(EAT_TZ).strftime("%Y-%m-%d %H:%M")
