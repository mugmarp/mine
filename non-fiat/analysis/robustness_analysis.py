#!/usr/bin/env python3
"""
Robustness analysis (VECTORIZED) for the futures scalping setup.
Precomputes all per-bar conditions once, then evaluates configs as boolean masks.
"""
import sys, pickle
import pandas as pd
import numpy as np

sys.path.insert(0, '/home/azureuser/BotScripts')
from futures_scalp_backtest import load_data, prepare, ema, rsi, atr, TAKER_FEE, SLIPPAGE, FUNDING_PER_8H

with open('/home/azureuser/BotScripts/futures_15m_cache.pkl', 'rb') as f:
    DATA = pickle.load(f)

BASE = {'rsi_lo': 45, 'rsi_hi': 75, 'vol30': 0.8, 'vol15': 0.7}


def build_matrix(df):
    """One row per 15m bar with all precomputed conditions."""
    d4 = df.resample('4h').agg({'o':'first','h':'max','l':'min','c':'last','v':'sum'}).dropna()
    d4['ema200'] = ema(d4.c, 200)
    d30 = df.resample('30min').agg({'o':'first','h':'max','l':'min','c':'last','v':'sum'}).dropna()
    d30['ema8'] = ema(d30.c, 8)
    d30['ema21'] = ema(d30.c, 21)
    d30['rsi'] = rsi(d30.c)
    d30['vma'] = d30.v.rolling(20).mean()

    d15 = df.copy()
    d15['ema8'] = ema(d15.c, 8)
    d15['ema21'] = ema(d15.c, 21)
    d15['atr'] = atr(d15)
    d15['vma'] = d15.v.rolling(20).mean()

    # align closed 4h / 30m values onto 15m index
    i4 = d4.index.searchsorted(d15.index, side='right') - 2
    i30 = d30.index.searchsorted(d15.index, side='right') - 2

    m = pd.DataFrame(index=d15.index)
    m['atr'] = d15['atr'].values
    m['vol15_r'] = (d15['v'] / d15['vma']).values
    m['o_next'] = d15['o'].shift(-1).values
    m['cross_up'] = ((d15.ema8.shift(1) <= d15.ema21.shift(1)) & (d15.ema8 > d15.ema21)).values
    m['cross_dn'] = ((d15.ema8.shift(1) >= d15.ema21.shift(1)) & (d15.ema8 < d15.ema21)).values
    m['fwd_hi'] = d15['h'].values
    m['fwd_lo'] = d15['l'].values
    m['fwd_c'] = d15['c'].values

    ok4 = i4 >= 200
    bull = np.zeros(len(d15), bool); bear = np.zeros(len(d15), bool)
    bull[ok4] = (d4.c.values[i4[ok4]] > d4.ema200.values[i4[ok4]])
    bear[ok4] = (d4.c.values[i4[ok4]] < d4.ema200.values[i4[ok4]])
    m['bull'] = bull; m['bear'] = bear

    ok30 = i30 >= 21
    r30 = np.full(len(d15), np.nan); v30 = np.full(len(d15), np.nan); align = np.zeros(len(d15))
    r30[ok30] = d30.rsi.values[i30[ok30]]
    with np.errstate(divide='ignore', invalid='ignore'):
        v30[ok30] = d30.v.values[i30[ok30]] / d30.vma.values[i30[ok30]]
    align[ok30] = np.where(d30.ema8.values[i30[ok30]] > d30.ema21.values[i30[ok30]], 1, -1)
    m['rsi30'] = r30; m['vol30_r'] = v30; m['align30'] = align
    return m


MAT = {sym: build_matrix(df) for sym, df in DATA.items()}
print(f"Built matrices for {len(MAT)} symbols\n")


def simulate_mask(m, mask, direction, atr_sl=1.5, atr_tp=3.0, max_hold=192,
                  fee=None, slip=None, fund=None):
    fee = TAKER_FEE if fee is None else fee
    slip = SLIPPAGE if slip is None else slip
    fund = FUNDING_PER_8H if fund is None else fund
    idxs = np.where(mask)[0]
    trades = []
    for i in idxs:
        if i + 1 >= len(m):
            continue
        entry = m.o_next.iloc[i]
        a = m.atr.iloc[i]
        if not np.isfinite(entry) or not np.isfinite(a) or a <= 0:
            continue
        if direction == 'LONG':
            sl, tp = entry - atr_sl * a, entry + atr_tp * a
        else:
            sl, tp = entry + atr_sl * a, entry - atr_tp * a
        end = min(i + 1 + max_hold, len(m))
        hi = m.fwd_hi.iloc[i+1:end].values
        lo = m.fwd_lo.iloc[i+1:end].values
        cl = m.fwd_c.iloc[i+1:end].values
        if len(hi) == 0:
            continue
        exit_px, reason, held = None, None, 0
        if direction == 'LONG':
            for k in range(len(hi)):
                held = k + 1
                if lo[k] <= sl: exit_px, reason = sl, 'SL'; break
                if hi[k] >= tp: exit_px, reason = tp, 'TP'; break
        else:
            for k in range(len(hi)):
                held = k + 1
                if hi[k] >= sl: exit_px, reason = sl, 'SL'; break
                if lo[k] <= tp: exit_px, reason = tp, 'TP'; break
        if exit_px is None:
            exit_px, reason = cl[-1], 'TIME'
        gross = (exit_px - entry) / entry if direction == 'LONG' else (entry - exit_px) / entry
        net = gross - 2 * (fee + slip) - (held * 15 / 60 / 8) * fund
        r_mult = (net * entry) / (atr_sl * a)
        trades.append({'sym': m.attrs.get('sym',''), 't': m.index[i], 'dir': direction,
                       'net_pct': net, 'r': r_mult, 'reason': reason, 'held_h': held*15/60})
    return trades


def evaluate(params, atr_sl=1.5, atr_tp=3.0, fee=None, slip=None, fund=None,
             allow_long=True, allow_short=True, tag_time=True):
    all_t = []
    for sym, m in MAT.items():
        long_mask = (m.bull & m.cross_up & (m.align30 == 1) &
                     (m.rsi30 >= params['rsi_lo']) & (m.rsi30 <= params['rsi_hi']) &
                     (m.vol30_r >= params['vol30']) & (m.vol15_r >= params['vol15'])).fillna(False).values
        short_mask = (m.bear & m.cross_dn & (m.align30 == -1) &
                      (m.rsi30 >= params['rsi_lo']) & (m.rsi30 <= params['rsi_hi']) &
                      (m.vol30_r >= params['vol30']) & (m.vol15_r >= params['vol15'])).fillna(False).values
        if allow_long:
            tr = simulate_mask(m, long_mask, 'LONG', atr_sl, atr_tp, fee=fee, slip=slip, fund=fund)
            for t in tr: t['sym'] = sym
            all_t += tr
        if allow_short:
            tr = simulate_mask(m, short_mask, 'SHORT', atr_sl, atr_tp, fee=fee, slip=slip, fund=fund)
            for t in tr: t['sym'] = sym
            all_t += tr
    return all_t


def summarize(trades, label=""):
    if not trades:
        return {'label': label, 'n': 0, 'wr': 0, 'exp': 0, 'total': 0, 'pf': 0, 'dd': 0}
    g = pd.DataFrame(trades)
    wins = g[g.net_pct > 0]; losses = g[g.net_pct <= 0]
    cum = g.r.cumsum()
    dd = (cum - cum.cummax()).min()
    pf = wins.net_pct.sum() / abs(losses.net_pct.sum()) if len(losses) and losses.net_pct.sum() != 0 else float('inf')
    return {'label': label, 'n': len(g), 'wr': len(wins)/len(g)*100, 'exp': g.r.mean(),
            'total': g.r.sum(), 'pf': pf, 'dd': dd}


print("=" * 82)
print("1. MONTHLY BREAKDOWN — regime dependence (base params)")
print("=" * 82)
base_t = evaluate(BASE)
bdf = pd.DataFrame(base_t)
bdf['month'] = pd.to_datetime(bdf.t).dt.strftime('%Y-%m')
print(f"\n{'Month':<10}{'n':>5}{'WR%':>8}{'ExpR':>9}{'TotalR':>9}{'PF':>7}")
print("-" * 50)
for month, g in bdf.groupby('month'):
    wins = g[g.net_pct > 0]; losses = g[g.net_pct <= 0]
    pf = wins.net_pct.sum()/abs(losses.net_pct.sum()) if len(losses) and losses.net_pct.sum() != 0 else float('inf')
    print(f"{month:<10}{len(g):>5}{len(wins)/len(g)*100:>7.1f}%{g.r.mean():>9.3f}{g.r.sum():>9.1f}{pf:>7.2f}")

print("\n" + "=" * 82)
print("2. PARAMETER SENSITIVITY — is the edge a knife-edge?")
print("=" * 82)
print(f"\n{'vol30':>7}{'vol15':>7}{'rsi_hi':>8}{'n':>6}{'WR%':>8}{'ExpR':>9}{'TotalR':>9}{'PF':>7}")
print("-" * 62)
rows = []
for vol30 in [0.6, 0.8, 1.0, 1.2, 1.4]:
    for vol15 in [0.5, 0.7, 0.9, 1.1]:
        for rsi_hi in [70, 75, 80]:
            p = dict(BASE, vol30=vol30, vol15=vol15, rsi_hi=rsi_hi)
            s = summarize(evaluate(p))
            if s['n'] == 0: continue
            print(f"{vol30:>7.1f}{vol15:>7.1f}{rsi_hi:>8}{s['n']:>6}{s['wr']:>7.1f}%{s['exp']:>9.3f}{s['total']:>9.1f}{s['pf']:>7.2f}")
            rows.append({'vol30':vol30,'vol15':vol15,'rsi_hi':rsi_hi,'n':s['n'],'wr':s['wr'],'exp':s['exp'],'total':s['total']})

sr = pd.DataFrame(rows)
print(f"\nSweep: {len(sr)} configs | positive expectancy: {(sr.exp>0).sum()}/{len(sr)} ({(sr.exp>0).mean()*100:.0f}%)")
print(f"  Median ExpR: {sr.exp.median():.3f} | range {sr.exp.min():.3f} to {sr.exp.max():.3f}")

print("\n" + "=" * 82)
print("3. COST SENSITIVITY — does the edge survive costs?")
print("=" * 82)
print(f"\n{'Scenario':<36}{'n':>5}{'WR%':>8}{'ExpR':>9}{'TotalR':>9}")
print("-" * 68)
for label, fee, slip, fund in [
    ("Base (0.05% fee + 0.02% slip)", 0.0005, 0.0002, 0.0001),
    ("High fee (0.08% taker)", 0.0008, 0.0002, 0.0001),
    ("High slippage (0.05%)", 0.0005, 0.0005, 0.0001),
    ("Pessimistic (0.08% + 0.05% + fund)", 0.0008, 0.0005, 0.0002),
    ("Zero costs (theoretical)", 0.0, 0.0, 0.0),
]:
    s = summarize(evaluate(BASE, fee=fee, slip=slip, fund=fund))
    print(f"{label:<36}{s['n']:>5}{s['wr']:>7.1f}%{s['exp']:>9.3f}{s['total']:>9.1f}")

print("\n" + "=" * 82)
print("4. DIRECTIONAL + PER-SYMBOL ROBUSTNESS (base params)")
print("=" * 82)
sl_ = summarize(evaluate(BASE, allow_short=False), "LONG only")
ss_ = summarize(evaluate(BASE, allow_long=False), "SHORT only")
sc_ = summarize(base_t, "COMBINED")
print(f"\n{'Scope':<12}{'n':>5}{'WR%':>8}{'ExpR':>9}{'TotalR':>9}{'PF':>7}{'MaxDD':>8}")
print("-" * 58)
for s in (sc_, sl_, ss_):
    print(f"{s['label']:<12}{s['n']:>5}{s['wr']:>7.1f}%{s['exp']:>9.3f}{s['total']:>9.1f}{s['pf']:>7.2f}{s['dd']:>8.1f}")

print(f"\n{'Symbol':<18}{'n':>5}{'WR%':>8}{'ExpR':>9}{'TotalR':>9}{'Long':>6}{'Short':>7}")
print("-" * 62)
for sym in MAT:
    g = bdf[bdf.sym == sym]
    if len(g) == 0:
        print(f"{sym:<18}{'0':>5}"); continue
    wins = g[g.net_pct > 0]
    print(f"{sym:<18}{len(g):>5}{len(wins)/len(g)*100:>7.1f}%{g.r.mean():>9.3f}{g.r.sum():>9.1f}"
          f"{len(g[g.dir=='LONG']):>6}{len(g[g.dir=='SHORT']):>7}")
print("\nDone.")
