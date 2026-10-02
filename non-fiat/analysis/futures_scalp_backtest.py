#!/usr/bin/env python3
"""
Futures Scalping Backtest — Long/Short robustness analysis
- Realistic costs: taker fees + slippage + funding
- Entry at next-candle open (no look-ahead)
- Tests long-only, short-only, combined
- Parameter sensitivity sweep for robustness
"""
import os, sys, time, pickle, itertools
import ccxt
import pandas as pd
import numpy as np

CACHE = "/home/azureuser/BotScripts/futures_15m_cache.pkl"
PAIRS = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "XRP/USDT:USDT",
         "ADA/USDT:USDT", "DOGE/USDT:USDT", "PENGU/USDT:USDT", "PEOPLE/USDT:USDT"]
DAYS = 90

# Cost model (conservative)
TAKER_FEE = 0.0005      # 0.05% each side
SLIPPAGE = 0.0002       # 0.02% each side
FUNDING_PER_8H = 0.0001 # 0.01% per 8h held


def fetch_paginated(ex, symbol, timeframe='15m', days=90):
    ms_per_candle = 15 * 60 * 1000
    total = int(days * 24 * 60 / 15)
    since = ex.milliseconds() - total * ms_per_candle
    out = []
    guard = 0
    while len(out) < total and guard < 40:
        guard += 1
        try:
            batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1500)
        except Exception as e:
            print(f"  fetch error {symbol}: {e}")
            time.sleep(2)
            continue
        if not batch:
            break
        out += batch
        since = batch[-1][0] + ms_per_candle
        # stop once we've reached the present
        if batch[-1][0] >= ex.milliseconds() - ms_per_candle:
            break
        time.sleep(ex.rateLimit / 1000)
    if not out:
        return pd.DataFrame()
    df = pd.DataFrame(out, columns=['t', 'o', 'h', 'l', 'c', 'v']).drop_duplicates('t')
    df['t'] = pd.to_datetime(df['t'], unit='ms', utc=True)
    df.set_index('t', inplace=True)
    for c in ['o', 'h', 'l', 'c', 'v']:
        df[c] = df[c].astype(float)
    return df.sort_index()


def load_data():
    if os.path.exists(CACHE):
        with open(CACHE, 'rb') as f:
            return pickle.load(f)
    ex = ccxt.binance({'enableRateLimit': True, 'timeout': 20000,
                       'options': {'defaultType': 'future'}})
    data = {}
    for sym in PAIRS:
        print(f"Fetching {sym} ({DAYS}d 15m)...")
        df = fetch_paginated(ex, sym, '15m', DAYS)
        if len(df) > 500:
            data[sym] = df
            print(f"  {len(df)} candles  {df.index[0].date()} -> {df.index[-1].date()}")
        else:
            print(f"  insufficient: {len(df)}")
    with open(CACHE, 'wb') as f:
        pickle.dump(data, f)
    return data


def ema(s, p): return s.ewm(span=p, adjust=False).mean()


def rsi(s, p=14):
    d = s.diff()
    g = d.clip(lower=0); l = -d.clip(upper=0)
    rma = lambda x, n: x.ewm(alpha=1/n, adjust=False).mean()
    return (100 - 100 / (1 + rma(g, p) / rma(l, p).replace(0, np.nan))).fillna(50)


def atr(df, p=14):
    tr = pd.concat([df.h - df.l, (df.h - df.c.shift()).abs(),
                    (df.l - df.c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/p, adjust=False).mean()


def prepare(df):
    """Build 4H/30m/15m frames with indicators."""
    d4 = df.resample('4h').agg({'o': 'first', 'h': 'max', 'l': 'min', 'c': 'last', 'v': 'sum'}).dropna()
    d4['ema200'] = ema(d4.c, 200)

    d30 = df.resample('30min').agg({'o': 'first', 'h': 'max', 'l': 'min', 'c': 'last', 'v': 'sum'}).dropna()
    d30['ema8'] = ema(d30.c, 8)
    d30['ema21'] = ema(d30.c, 21)
    d30['rsi'] = rsi(d30.c)
    d30['vma'] = d30.v.rolling(20).mean()

    d15 = df.copy()
    d15['ema8'] = ema(d15.c, 8)
    d15['ema21'] = ema(d15.c, 21)
    d15['atr'] = atr(d15)
    d15['vma'] = d15.v.rolling(20).mean()

    return d4, d30, d15


def generate_signals(d4, d30, d15, params):
    """params: dict with rsi_lo, rsi_hi, vol30, vol15, allow_long, allow_short"""
    sigs = []
    # map 15m timestamps to closed 4h / 30m indices
    idx4 = d4.index
    idx30 = d30.index

    for i in range(60, len(d15) - 1):
        t = d15.index[i]
        # last CLOSED 4h candle
        j4 = idx4.searchsorted(t, side='right') - 2
        if j4 < 200:
            continue
        c4 = d4.iloc[j4]
        bull = c4.c > c4.ema200
        bear = c4.c < c4.ema200

        j30 = idx30.searchsorted(t, side='right') - 2
        if j30 < 21:
            continue
        c30 = d30.iloc[j30]

        if c30.rsi < params['rsi_lo'] or c30.rsi > params['rsi_hi']:
            continue
        if c30.v < c30.vma * params['vol30']:
            continue

        c15 = d15.iloc[i]; p15 = d15.iloc[i-1]
        if c15.v < c15.vma * params['vol15']:
            continue

        cross_up = p15.ema8 <= p15.ema21 and c15.ema8 > c15.ema21
        cross_dn = p15.ema8 >= p15.ema21 and c15.ema8 < c15.ema21

        if params['allow_long'] and bull and cross_up and c30.ema8 > c30.ema21:
            sigs.append((i, 'LONG'))
        elif params['allow_short'] and bear and cross_dn and c30.ema8 < c30.ema21:
            sigs.append((i, 'SHORT'))
    return sigs


def simulate(d15, sigs, atr_sl=1.5, atr_tp=3.0, max_hold=192):
    """Enter next candle open. Return trade list."""
    trades = []
    for i, direction in sigs:
        if i + 1 >= len(d15):
            continue
        entry = d15.o.iloc[i + 1]
        a = d15.atr.iloc[i]
        if not np.isfinite(a) or a <= 0:
            continue
        if direction == 'LONG':
            sl, tp = entry - atr_sl * a, entry + atr_tp * a
        else:
            sl, tp = entry + atr_sl * a, entry - atr_tp * a

        fwd = d15.iloc[i + 1:i + 1 + max_hold]
        if len(fwd) == 0:
            continue

        exit_px, reason, held = None, None, 0
        for k, (_, row) in enumerate(fwd.iterrows()):
            held = k + 1
            if direction == 'LONG':
                if row.l <= sl:
                    exit_px, reason = sl, 'SL'; break
                if row.h >= tp:
                    exit_px, reason = tp, 'TP'; break
            else:
                if row.h >= sl:
                    exit_px, reason = sl, 'SL'; break
                if row.l <= tp:
                    exit_px, reason = tp, 'TP'; break
        if exit_px is None:
            exit_px, reason = fwd.c.iloc[-1], 'TIME'

        gross = (exit_px - entry) / entry if direction == 'LONG' else (entry - exit_px) / entry
        fees = 2 * (TAKER_FEE + SLIPPAGE)
        funding = (held * 15 / 60 / 8) * FUNDING_PER_8H
        net = gross - fees - funding
        r_mult = (net * entry) / (atr_sl * a)  # R in ATR-risk units

        trades.append({'dir': direction, 'reason': reason, 'net_pct': net,
                       'r': r_mult, 'held_h': held * 15 / 60})
    return trades


def stats(trades, label):
    if not trades:
        return {'label': label, 'n': 0}
    df = pd.DataFrame(trades)
    wins = df[df.net_pct > 0]
    losses = df[df.net_pct <= 0]
    n = len(df)
    wr = len(wins) / n * 100
    exp_r = df.r.mean()
    total_r = df.r.sum()
    # max drawdown on cumulative R
    cum = df.r.cumsum()
    dd = (cum - cum.cummax()).min()
    pf = wins.net_pct.sum() / abs(losses.net_pct.sum()) if len(losses) and losses.net_pct.sum() != 0 else float('inf')
    return {
        'label': label, 'n': n, 'wr': wr, 'exp_r': exp_r, 'total_r': total_r,
        'pf': pf, 'max_dd_r': dd, 'avg_hold_h': df.held_h.mean(),
        'tp': int((df.reason == 'TP').sum()), 'sl': int((df.reason == 'SL').sum()),
        'time': int((df.reason == 'TIME').sum()),
        'avg_win_r': wins.r.mean() if len(wins) else 0,
        'avg_loss_r': losses.r.mean() if len(losses) else 0,
    }


def main():
    data = load_data()
    print(f"\nLoaded {len(data)} symbols\n")

    BASE = {'rsi_lo': 45, 'rsi_hi': 75, 'vol30': 0.8, 'vol15': 0.7,
            'allow_long': True, 'allow_short': True}

    all_trades = []
    print("=" * 78)
    print(f"{'Symbol':<16}{'n':>5}{'WR%':>8}{'ExpR':>8}{'TotalR':>9}{'PF':>7}{'MaxDD':>8}{'Hold':>7}")
    print("=" * 78)
    for sym, df in data.items():
        d4, d30, d15 = prepare(df)
        sigs = generate_signals(d4, d30, d15, BASE)
        tr = simulate(d15, sigs)
        s = stats(tr, sym)
        if s['n']:
            print(f"{sym:<16}{s['n']:>5}{s['wr']:>7.1f}%{s['exp_r']:>8.2f}{s['total_r']:>9.1f}"
                  f"{s['pf']:>7.2f}{s['max_dd_r']:>8.1f}{s['avg_hold_h']:>6.1f}h")
        else:
            print(f"{sym:<16}{'0':>5}   (no trades)")
        all_trades += tr
    print("=" * 78)

    combined = stats(all_trades, 'COMBINED')
    longs = stats([t for t in all_trades if t['dir'] == 'LONG'], 'LONG only')
    shorts = stats([t for t in all_trades if t['dir'] == 'SHORT'], 'SHORT only')

    print("\n### DIRECTIONAL BREAKDOWN (base params) ###")
    for s in (combined, longs, shorts):
        if s['n']:
            print(f"  {s['label']:<12} n={s['n']:<4} WR={s['wr']:.1f}%  Exp={s['exp_r']:.3f}R  "
                  f"Total={s['total_r']:.1f}R  PF={s['pf']:.2f}  MaxDD={s['max_dd_r']:.1f}R  "
                  f"[TP:{s['tp']} SL:{s['sl']} TIME:{s['time']}]")
        else:
            print(f"  {s['label']:<12} NO TRADES")

    # Save for further analysis
    with open('/home/azureuser/BotScripts/backtest_results.pkl', 'wb') as f:
        pickle.dump({'all': all_trades, 'combined': combined, 'long': longs, 'short': shorts}, f)
    print("\nResults saved to backtest_results.pkl")


if __name__ == "__main__":
    main()
