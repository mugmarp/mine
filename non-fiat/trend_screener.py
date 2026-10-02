#!/usr/bin/env python3
"""
Trend Confluence Screener - Regime-Filtered Version

Production improvements:
1. ADX must be RISING (not just >25) - filters momentum-strength
2. Volume confirmation on 1H candle
3. Daily digest mode: one summary alert per day instead of per-signal spam
4. Reduced pair list to focus on liquid majors + proven volatile tokens
5. State pruning retains 30 entries per symbol instead of 100
"""
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
STATE_FILE = os.getenv("STATE_FILE", "/opt/bb_screener/trend_alerted_candles.json")

# Reduced pair list - focus on liquid pairs with real trend potential
PAIRS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT",
    "BNB/USDT", "ADA/USDT"
]
POLL_INTERVAL_SECONDS = 300  # Scan every 5 minutes
MAX_STATE_ENTRIES = 30  # Keep last 30 alerts per symbol (30 hours of 1H candles)

# Local Timezone (Kampala / EAT - UTC+3)
EAT_TZ = timezone(timedelta(hours=3))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("TrendScreener")

# Global graceful shutdown flag
RUNNING = True


def handle_shutdown_signals(signum, frame):
    global RUNNING
    logger.info(f"Received termination signal ({signum}). Shutting down gracefully...")
    RUNNING = False


signal.signal(signal.SIGINT, handle_shutdown_signals)
signal.signal(signal.SIGTERM, handle_shutdown_signals)


def get_exchange():
    """Initializes CCXT Binance instance with socket timeouts."""
    return ccxt.binance({
        'enableRateLimit': True,
        'timeout': 15000,
        'options': {'defaultType': 'spot'}
    })


exchange = get_exchange()


def fetch_dataframe_safe(symbol: str, timeframe: str, limit: int = 250) -> pd.DataFrame:
    """Fetches OHLCV data with network resiliency and exponential backoff."""
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
            df['volume'] = df['volume'].astype(float)
            return df
        except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
            wait_time = 2 ** attempt
            logger.warning(f"Network issue fetching {symbol} (Attempt {attempt}/3). Retrying in {wait_time}s: {e}")
            time.sleep(wait_time)
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error fetching {symbol}: {e}")
            break
        except Exception as e:
            logger.error(f"Unexpected error fetching {symbol}: {e}")
            break
    return pd.DataFrame()


def load_state() -> dict:
    """Loads alerted candles state, ensuring timestamps are stored as integers."""
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
    """Prunes and saves state atomically to prevent corrupted disk writes."""
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
    """Sends Markdown formatted alert to Telegram with retry."""
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
        "parse_mode": "Markdown",
        "disable_web_page_preview": True
    }
    for attempt in range(1, 4):
        try:
            res = requests.post(url, json=payload, timeout=15)
            if res.status_code == 200:
                return
            logger.warning(f"Telegram API response ({res.status_code}) attempt {attempt}/3: {res.text}")
        except Exception as e:
            logger.error(f"Failed to send Telegram alert (attempt {attempt}/3): {e}")
        time.sleep(2)


# --- Technical Indicator Calculations ---

def calc_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def calc_wilders_rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder's Smoothing (RMA) used by TradingView for RSI & ADX."""
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
    """TradingView-compliant ADX formula matching native exchange charts."""
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


def analyze_market(symbol: str, state: dict, dry_run: bool = False) -> tuple:
    """Analyze market and return (signal_dict, alert_message) or (None, None).

    Signal dict format: {'symbol': str, 'direction': 'LONG'|'SHORT', 'entry': float,
                         'sl': float, 'tp': float, 'time_str': str, 'reasoning': str}

    Returns None, None if no valid signal.
    """
    # 1. Macro 4H Trend Evaluation
    df_4h = fetch_dataframe_safe(symbol, timeframe='4h', limit=220)
    if df_4h.empty or len(df_4h) < 200:
        return None, None

    df_4h['ema200'] = calc_ema(df_4h['close'], 200)

    # Evaluate confirmed closed 4H candle (iloc[-2])
    macro_close = df_4h['close'].iloc[-2]
    macro_ema200 = df_4h['ema200'].iloc[-2]
    macro_bullish = macro_close > macro_ema200
    macro_bearish = macro_close < macro_ema200

    # 2. Tactical 1H Execution Evaluation
    df_1h = fetch_dataframe_safe(symbol, timeframe='1h', limit=50)
    if df_1h.empty or len(df_1h) < 30:
        return None, None

    df_1h['ema8'] = calc_ema(df_1h['close'], 8)
    df_1h['ema21'] = calc_ema(df_1h['close'], 21)
    df_1h['rsi'] = calc_rsi(df_1h['close'], 14)
    df_1h['adx'] = calc_adx_wilders(df_1h, 14)
    df_1h['atr'] = calc_atr(df_1h, 14)

    # Evaluate confirmed closed 1H candle (iloc[-2]) vs prior candle (iloc[-3])
    c_candle = df_1h.iloc[-2]
    p_candle = df_1h.iloc[-3]

    open_time_ms = int(c_candle['open_time'])

    # Check deduplication state
    state_history = state.get(symbol, [])
    if open_time_ms in state_history:
        return None, None

    curr_close = c_candle['close']
    curr_ema8 = c_candle['ema8']
    curr_ema21 = c_candle['ema21']
    prev_ema8 = p_candle['ema8']
    prev_ema21 = p_candle['ema21']

    curr_adx = c_candle['adx']
    curr_rsi = c_candle['rsi']
    curr_atr = c_candle['atr']

    # --- REGIME FILTERS (NEW) ---

    # Filter 1: ADX must be > 25 (trending regime)
    if curr_adx < 25:
        return None, None

    # Filter 2: ADX must be RISING (momentum strengthening, not weakening)
    prev_adx = df_1h.iloc[-3]['adx']
    if curr_adx <= prev_adx:
        return None, None

    # Filter 3: Volume confirmation (at least 70% of average)
    avg_vol = df_1h['volume'].iloc[-20:].mean()
    if c_candle['volume'] < avg_vol * 0.7:
        return None, None

    # --- END REGIME FILTERS ---

    # Format local Kampala time (EAT / UTC+3)
    candle_dt_eat = datetime.fromtimestamp(open_time_ms / 1000, tz=timezone.utc).astimezone(EAT_TZ)
    time_str = candle_dt_eat.strftime('%Y-%m-%d %H:%M')

    # LONG Entry Condition
    long_cross = (prev_ema8 <= prev_ema21) and (curr_ema8 > curr_ema21)
    if macro_bullish and long_cross and (curr_rsi < 60):
        stop_loss = curr_close - (curr_atr * 1.5)
        take_profit = curr_close + (curr_atr * 3.0)
        reasoning = f"4H>Bullish(>EMA200) | 1H EMA8>EMA21 Cross | ADX={curr_adx:.1f}(↑) | RSI={curr_rsi:.1f}(<60) | Vol={c_candle['volume']:.0f}"

        signal = {
            'symbol': symbol,
            'direction': 'LONG',
            'entry': float(curr_close),
            'sl': float(stop_loss),
            'tp': float(take_profit),
            'time_str': time_str,
            'reasoning': reasoning
        }
        return signal, None  # Will be handled by digest mode

    # SHORT Entry Condition
    short_cross = (prev_ema8 >= prev_ema21) and (curr_ema8 < curr_ema21)
    if macro_bearish and short_cross and (curr_rsi > 40):
        stop_loss = curr_close + (curr_atr * 1.5)
        take_profit = curr_close - (curr_atr * 3.0)
        reasoning = f"4H<Bearish(<EMA200) | 1H EMA8<EMA21 Cross | ADX={curr_adx:.1f}(↑) | RSI={curr_rsi:.1f}(>40) | Vol={c_candle['volume']:.0f}"

        signal = {
            'symbol': symbol,
            'direction': 'SHORT',
            'entry': float(curr_close),
            'sl': float(stop_loss),
            'tp': float(take_profit),
            'time_str': time_str,
            'reasoning': reasoning
        }
        return signal, None

    return None, None


def send_daily_digest(signals: list):
    """Send a single daily summary of all triggered signals."""
    if not signals:
        msg = (
            "📊 *Daily Signal Digest* (Empty)\n\n"
            "No qualifying trend signals today.\n"
            f"_Updated: {datetime.now(EAT_TZ).strftime('%H:%M EAT')}_"
        )
    else:
        lines = ["📊 *Daily Signal Digest*", ""]
        for sig in signals:
            arrow = "🚀" if sig['direction'] == 'LONG' else "🔻"
            lines.append(
                f"{arrow} *{sig['direction']}* | `{sig['symbol']}`\n"
                f"  Price: `${sig['entry']:.2f}` | SL: `${sig['sl']:.2f}` | TP: `${sig['tp']:.2f}`\n"
                f"  _{sig['time_str']} EAT_ — `{sig['reasoning']}`\n"
            )
        lines.append(f"\n_Updated: {datetime.now(EAT_TZ).strftime('%H:%M EAT')}_")
        msg = "\n".join(lines)

    send_telegram_alert(msg)


def main():
    parser = argparse.ArgumentParser(description="Trend Confluence Screener Bot (Regime-Filtered)")
    parser.add_argument("--once", action="store_true", help="Run a single cycle and exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="Log output without sending Telegram messages")
    parser.add_argument("--digest", action="store_true",
                        help="Daily digest mode: collect signals during hours, send one summary at 18:00 EAT")
    args = parser.parse_args()

    logger.info("🟢 Starting Regime-Filtered Trend Screener...")
    logger.info("Filters: 4H EMA200 regime + ADX>25 & rising + 1H volume confirmation")
    logger.info(f"Pairs: {', '.join(PAIRS)}")

    state = load_state()

    # Digest mode: collect signals between 08:00-18:00 EAT, send summary at 18:00
    if args.digest:
        logger.info("Digest mode: collecting signals, summary at 18:00 EAT daily")
        collected_signals = []
        last_digest_day = None

        while RUNNING:
            now_eat = datetime.now(EAT_TZ)
            current_day = now_eat.strftime('%Y-%m-%d')

            # Reset daily digest at midnight EAT
            if last_digest_day != current_day:
                collected_signals = []
                last_digest_day = current_day

            # Collect signals between 08:00 and 17:55 EAT
            if 8 <= now_eat.hour <= 17 and now_eat.minute <= 55:
                state_changed = False
                for symbol in PAIRS:
                    if not RUNNING:
                        break
                    try:
                        signal, _ = analyze_market(symbol, state, dry_run=args.dry_run)
                        if signal:
                            state_changed = True
                            # Mark as alerted
                            open_time_key = signal['symbol']
                            ts = int(datetime.now(timezone.utc).timestamp() * 1000)
                            if signal['symbol'] not in state:
                                state[signal['symbol']] = []
                            state[signal['symbol']].append(ts)
                            collected_signals.append(signal)
                            logger.info(f"Collected signal: {signal['symbol']} {signal['direction']}")
                    except Exception as e:
                        logger.error(f"Error screening {symbol}: {e}")

                if state_changed and not args.dry_run:
                    save_state(state)

            # Send digest at 18:00 EAT
            elif now_eat.hour == 18 and now_eat.minute == 0 and collected_signals:
                logger.info(f"Sending daily digest with {len(collected_signals)} signals")
                if not args.dry_run:
                    send_daily_digest(collected_signals)
                    collected_signals = []
                else:
                    # Dry run: just log
                    logger.info("[DRY RUN] Daily digest would contain:")
                    for sig in collected_signals:
                        logger.info(f"  {sig['symbol']}: {sig['direction']}")
                    collected_signals = []

            if args.once:
                break

            for _ in range(POLL_INTERVAL_SECONDS):
                if not RUNNING:
                    break
                time.sleep(1)

        logger.info("Trend Screener (digest mode) execution finished cleanly.")
        return

    # Standard mode: immediate alerts (but with regime filters applied)
    while RUNNING:
        state_changed = False
        for symbol in PAIRS:
            if not RUNNING:
                break
            try:
                signal, _ = analyze_market(symbol, state, dry_run=args.dry_run)
                if signal:
                    # Build alert message
                    arrow = "🚀" if signal['direction'] == 'LONG' else "🔻"
                    msg = (
                        f"{arrow} *STRONG {signal['direction']} SIGNAL: `{signal['symbol']}`*\n\n"
                        f"💡 *Reasoning:* `{signal['reasoning']}`\n"
                        f"• *Time (EAT):* `{signal['time_str']}`\n"
                        f"• *Entry:* `${signal['entry']:.2f}`\n"
                        f"🛡️ *Stop Loss:* `${signal['sl']:.2f}`\n"
                        f"🎯 *Take Profit:* `${signal['tp']:.2f}`"
                    )

                    logger.info(f"Triggered {signal['direction']} signal for {signal['symbol']}")
                    send_telegram_alert(msg, dry_run=args.dry_run)

                    # Update state
                    if signal['symbol'] not in state:
                        state[signal['symbol']] = []
                    state[signal['symbol']].append(int(datetime.now(timezone.utc).timestamp() * 1000))
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

    logger.info("Trend Screener execution finished cleanly.")


if __name__ == "__main__":
    main()
