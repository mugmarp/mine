#!/usr/bin/env python3
import os
import sys
import time
import json
import signal
import logging
import argparse
from datetime import datetime, timezone, timedelta

import ccxt
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
STATE_FILE = os.getenv("FUTURES_STATE_FILE", "/opt/bb_screener/futures_30m_alerted_candles.json")

PAIRS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT"]
POLL_INTERVAL_SECONDS = 180  # Scan every 3 minutes (suited for 30m timeframe)
MAX_STATE_ENTRIES = 100
LEVERAGE = 10  # Recommended leverage tag in alerts

# Local Timezone (Kampala / EAT - UTC+3)
EAT_TZ = timezone(timedelta(hours=3))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Futures30mScreener")

# Global graceful shutdown flag
RUNNING = True


def handle_shutdown_signals(signum, frame):
    global RUNNING
    logger.info(f"Received termination signal ({signum}). Shutting down gracefully...")
    RUNNING = False


signal.signal(signal.SIGINT, handle_shutdown_signals)
signal.signal(signal.SIGTERM, handle_shutdown_signals)


def get_exchange():
    """Initializes CCXT Binance Futures instance."""
    return ccxt.binance({
        'enableRateLimit': True,
        'timeout': 15000,
        'options': {'defaultType': 'future'}  # Configured for USDT-M Futures
    })


exchange = get_exchange()


def fetch_dataframe_safe(symbol: str, timeframe: str, limit: int = 250) -> pd.DataFrame:
    """Fetches OHLCV futures data with exponential backoff network resiliency."""
    for attempt in range(1, 4):
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            if not ohlcv:
                return pd.DataFrame()
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['open_time'] = df['timestamp'].astype(int)
            df['close'] = df['close'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            return df
        except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
            wait_time = 2 ** attempt  # 2s -> 4s -> 8s
            logger.warning(f"Network issue fetching {symbol} ({attempt}/3). Retrying in {wait_time}s: {e}")
            time.sleep(wait_time)
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error fetching {symbol}: {e}")
            break
        except Exception as e:
            logger.error(f"Unexpected error fetching {symbol}: {e}")
            break
    return pd.DataFrame()


def load_state() -> dict:
    """Loads state file to prevent duplicate alerts on the same candle."""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                data = json.load(f)
                cleaned_state = {}
                for symbol, timestamps in data.items():
                    if isinstance(timestamps, list):
                        cleaned_state[symbol] = [int(ts) for ts in timestamps]
                    else:
                        cleaned_state[symbol] = [int(timestamps)]
                return cleaned_state
        except Exception as e:
            logger.error(f"Error loading state file {STATE_FILE}: {e}")
    return {}


def save_state(state: dict):
    """Prunes and saves state atomically to disk."""
    pruned_state = {}
    for symbol, timestamps in state.items():
        if isinstance(timestamps, list):
            pruned_state[symbol] = sorted([int(ts) for ts in timestamps])[-MAX_STATE_ENTRIES:]
        else:
            pruned_state[symbol] = [int(timestamps)]

    try:
        temp_file = f"{STATE_FILE}.tmp"
        with open(temp_file, "w") as f:
            json.dump(pruned_state, f, indent=2)
        os.replace(temp_file, STATE_FILE)
    except Exception as e:
        logger.error(f"Error saving state file: {e}")


def send_telegram_alert(message: str, dry_run: bool = False):
    """Sends Markdown formatted signal alert to Telegram."""
    if dry_run:
        logger.info(f"[DRY RUN - ALERT NOT SENT]\n{message}")
        return

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram credentials missing in environment variables.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        res = requests.post(url, json=payload, timeout=10)
        if res.status_code != 200:
            logger.warning(f"Telegram API response ({res.status_code}): {res.text}")
    except Exception as e:
        logger.error(f"Failed to send Telegram alert: {e}")


# --- Technical Indicator Calculations ---

def calc_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def calc_wilders_rma(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(alpha=1/period, adjust=False).mean()


def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = calc_wilders_rma(gain, period)
    avg_loss = calc_wilders_rma(loss, period)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return calc_wilders_rma(tr, period)


def calc_adx_wilders(df: pd.DataFrame, period: int = 14) -> pd.Series:
    up = df['high'] - df['high'].shift(1)
    down = df['low'].shift(1) - df['low']

    pos_dm = np.where((up > down) & (up > 0), up, 0.0)
    neg_dm = np.where((down > up) & (down > 0), down, 0.0)

    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)

    tr_smooth = calc_wilders_rma(pd.Series(tr), period)
    pos_di = 100 * (calc_wilders_rma(pd.Series(pos_dm), period) / tr_smooth.replace(0, np.nan))
    neg_di = 100 * (calc_wilders_rma(pd.Series(neg_dm), period) / tr_smooth.replace(0, np.nan))

    sum_di = pos_di + neg_di
    sum_di = sum_di.replace(0, np.nan)

    dx = 100 * (pos_di - neg_di).abs() / sum_di
    return calc_wilders_rma(dx.fillna(0), period)


def analyze_market(symbol: str, state: dict, dry_run: bool = False) -> bool:
    # 1. Macro Trend Evaluation (1H Timeframe)
    df_1h = fetch_dataframe_safe(symbol, timeframe='1h', limit=220)
    if df_1h.empty or len(df_1h) < 200:
        return False

    df_1h['ema200'] = calc_ema(df_1h['close'], 200)
    macro_close = df_1h['close'].iloc[-2]
    macro_ema200 = df_1h['ema200'].iloc[-2]
    
    macro_bullish = macro_close > macro_ema200
    macro_bearish = macro_close < macro_ema200

    # 2. Tactical Execution Evaluation (30m Timeframe)
    df_30m = fetch_dataframe_safe(symbol, timeframe='30m', limit=60)
    if df_30m.empty or len(df_30m) < 30:
        return False

    df_30m['ema8'] = calc_ema(df_30m['close'], 8)
    df_30m['ema21'] = calc_ema(df_30m['close'], 21)
    df_30m['rsi'] = calc_rsi(df_30m['close'], 14)
    df_30m['adx'] = calc_adx_wilders(df_30m, 14)
    df_30m['atr'] = calc_atr(df_30m, 14)

    # Evaluate confirmed closed 30m candle (iloc[-2]) vs prior (iloc[-3])
    c_candle = df_30m.iloc[-2]
    p_candle = df_30m.iloc[-3]

    open_time_ms = int(c_candle['open_time'])
    
    # Check deduplication
    state_history = state.get(symbol, [])
    if open_time_ms in state_history:
        return False

    curr_close = c_candle['close']
    curr_ema8 = c_candle['ema8']
    curr_ema21 = c_candle['ema21']
    prev_ema8 = p_candle['ema8']
    prev_ema21 = p_candle['ema21']
    
    curr_adx = c_candle['adx']
    curr_rsi = c_candle['rsi']
    curr_atr = c_candle['atr']

    # Filter ranging markets
    if curr_adx < 25:
        return False

    candle_dt_eat = datetime.fromtimestamp(open_time_ms / 1000, tz=timezone.utc).astimezone(EAT_TZ)
    time_str = candle_dt_eat.strftime('%Y-%m-%d %H:%M')

    tag_symbol = symbol.replace("/", "")
    alert_triggered = False

    # LONG Entry Condition
    long_cross = (prev_ema8 <= prev_ema21) and (curr_ema8 > curr_ema21)
    if macro_bullish and long_cross and (curr_rsi < 65):
        entry_low = curr_close - (curr_atr * 0.25)
        entry_high = curr_close + (curr_atr * 0.10)
        stop_loss = curr_close - (curr_atr * 1.5)

        tp1 = curr_close + (curr_atr * 0.8)
        tp2 = curr_close + (curr_atr * 1.6)
        tp3 = curr_close + (curr_atr * 2.4)
        tp4 = curr_close + (curr_atr * 3.2)
        tp5 = curr_close + (curr_atr * 4.0)

        reasoning = f"1H Bullish (>EMA200) | 30m EMA8>EMA21 Cross | ADX={curr_adx:.1f} | RSI={curr_rsi:.1f}"

        msg = (
            f"💥 *Futures*\n\n"
            f"✅ *Long*\n\n"
            f"#{tag_symbol}\n\n"
            f"Entry zone : {entry_low:.2f} - {entry_high:.2f}\n\n"
            f"Take Profits :\n\n"
            f"{tp1:.2f}\n"
            f"{tp2:.2f}\n"
            f"{tp3:.2f}\n"
            f"{tp4:.2f}\n"
            f"{tp5:.2f}\n\n"
            f"Stop loss :{stop_loss:.2f}\n\n"
            f"Leverage: {LEVERAGE}x\n\n"
            f"💡 *Signal Info:* `{reasoning}`\n"
            f"• *Closed Candle (EAT):* `{time_str} EAT`"
        )
        logger.info(f"Triggered Futures LONG signal for {symbol}")
        send_telegram_alert(msg, dry_run=dry_run)
        alert_triggered = True

    # SHORT Entry Condition
    short_cross = (prev_ema8 >= prev_ema21) and (curr_ema8 < curr_ema21)
    if macro_bearish and short_cross and (curr_rsi > 35):
        entry_high = curr_close + (curr_atr * 0.25)
        entry_low = curr_close - (curr_atr * 0.10)
        stop_loss = curr_close + (curr_atr * 1.5)

        tp1 = curr_close - (curr_atr * 0.8)
        tp2 = curr_close - (curr_atr * 1.6)
        tp3 = curr_close - (curr_atr * 2.4)
        tp4 = curr_close - (curr_atr * 3.2)
        tp5 = curr_close - (curr_atr * 4.0)

        reasoning = f"1H Bearish (<EMA200) | 30m EMA8<EMA21 Cross | ADX={curr_adx:.1f} | RSI={curr_rsi:.1f}"

        msg = (
            f"💥 *Futures*\n\n"
            f"🔻 *Short*\n\n"
            f"#{tag_symbol}\n\n"
            f"Entry zone : {entry_low:.2f} - {entry_high:.2f}\n\n"
            f"Take Profits :\n\n"
            f"{tp1:.2f}\n"
            f"{tp2:.2f}\n"
            f"{tp3:.2f}\n"
            f"{tp4:.2f}\n"
            f"{tp5:.2f}\n\n"
            f"Stop loss :{stop_loss:.2f}\n\n"
            f"Leverage: {LEVERAGE}x\n\n"
            f"💡 *Signal Info:* `{reasoning}`\n"
            f"• *Closed Candle (EAT):* `{time_str} EAT`"
        )
        logger.info(f"Triggered Futures SHORT signal for {symbol}")
        send_telegram_alert(msg, dry_run=dry_run)
        alert_triggered = True

    if alert_triggered:
        if symbol not in state:
            state[symbol] = []
        state[symbol].append(open_time_ms)
        return True

    return False


def main():
    parser = argparse.ArgumentParser(description="30m/1H Futures Signal Screener Bot")
    parser.add_argument("--once", action="store_true", help="Run a single cycle and exit")
    parser.add_argument("--dry-run", action="store_true", help="Log output without sending Telegram messages")
    args = parser.parse_args()

    logger.info("🟢 Starting 30m/1H Futures Signal Screener...")
    state = load_state()

    while RUNNING:
        state_changed = False
        for symbol in PAIRS:
            if not RUNNING:
                break
            try:
                if analyze_market(symbol, state, dry_run=args.dry_run):
                    state_changed = True
            except Exception as e:
                logger.error(f"Error screening {symbol}: {e}")

        if state_changed and not args.dry_run:
            save_state(state)

        if args.once:
            break

        for _ in range(POLL_INTERVAL_SECONDS):
            if not RUNNING:
                break
            time.sleep(1)

    logger.info("Futures Screener execution finished cleanly.")


if __name__ == "__main__":
    main()
