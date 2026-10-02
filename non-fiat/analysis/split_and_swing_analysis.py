#!/usr/bin/env python3
"""
1. Critical review of prior decisions
2. Long-only vs Short-only as SEPARATE setups (accuracy/WR impact)
3. 4H swing setup backtest (2 years)
"""
import sys, pickle
import pandas as pd
import numpy as np

sys.path.insert(0, '/home/azureuser/BotScripts')

# ============ PART A: SPLIT SETUPS on the 15m/30m futures universe ============
from robustness_analysis import evaluate, summarize, BASE, MAT
from futures_scalp_backtest import TAKER_FEE, SLIPPAGE, FUNDING_PER_8H

print("=" * 86)
print("PART A: LONG vs SHORT AS SEPARATE SETUPS (15m/30m futures, 90 days)")
print("=" * 86)

# A1: same params, split by direction
long_t = evaluate(BASE, allow_short=False)
short_t = evaluate(BASE, allow_long=False)
comb_t = evaluate(BASE)

def s(trades, label):
    if not trades:
        return {'label': label, 'n': 0}
    g = pd.DataFrame(trades)
    w = g[g.net_pct > 0]; l = g[g.net_pct <= 0]
    cum = g.r.cumsum(); dd = (cum - cum.cummax()).min()
    pf = w.net_pct.sum()/abs(l.net_pct.sum()) if len(l) and l.net_pct.sum() != 0 else float('inf')
    # per-trade $ if risking 1% of 10k = $100 per R
    return {'label': label, 'n': len(g), 'wr': len(w)/len(g)*100, 'exp': g.r.mean(),
            'total': g.r.sum(), 'pf': pf, 'dd': dd, 'avg_hold': g.held_h.mean()}

print(f"\n{'Setup':<22}{'n':>5}{'WR%':>8}{'ExpR':>9}{'TotalR':>9}{'PF':>7}{'MaxDD':>9}{'Hold':>7}")
print("-" * 76)
for t, lab in [(comb_t,'COMBINED'), (long_t,'LONG-only'), (short_t,'SHORT-only')]:
    r = s(t, lab)
    if r['n']:
        print(f"{r['label']:<22}{r['n']:>5}{r['wr']:>7.1f}%{r['exp']:>9.3f}{r['total']:>9.1f}"
              f"{r['pf']:>7.2f}{r['dd']:>9.1f}{r['avg_hold']:>6.1f}h")

# A2: "STRONG" variants — tighter filters to improve accuracy
print("\n--- TIGHTENED 'STRONG' VARIANTS (higher selectivity) ---")
strong_sets = [
    ('LONG vol30>=1.2', dict(BASE, vol30=1.2, vol15=0.9), False, True),
    ('LONG vol30>=1.4', dict(BASE, vol30=1.4, vol15=0.9), False, True),
    ('SHORT vol30>=0.8', dict(BASE), True, False),
    ('SHORT vol30>=1.2', dict(BASE, vol30=1.2, vol15=0.7), True, False),
]
print(f"\n{'Setup':<22}{'n':>5}{'WR%':>8}{'ExpR':>9}{'TotalR':>9}{'PF':>7}{'MaxDD':>9}")
print("-" * 69)
for lab, p, allow_long, allow_short in strong_sets:
    t = evaluate(p, allow_long=allow_long, allow_short=allow_short)
    r = s(t, lab)
    if r['n']:
        print(f"{lab:<22}{r['n']:>5}{r['wr']:>7.1f}%{r['exp']:>9.3f}{r['total']:>9.1f}{r['pf']:>7.2f}{r['dd']:>9.1f}")
    else:
        print(f"{lab:<22}{'0':>5}  (no trades)")

# A3: per-pair split
print("\n--- PER-PAIR, BY DIRECTION (base params) ---")
cdf = pd.DataFrame(comb_t)
print(f"\n{'Symbol':<20}{'L n':>5}{'L WR':>7}{'L R':>7}{'S n':>5}{'S WR':>7}{'S R':>7}")
print("-" * 58)
for sym in MAT:
    g = cdf[cdf.sym == sym]
    lg = g[g.dir == 'LONG']; sg = g[g.dir == 'SHORT']
    lwr = f"{len(lg[lg.net_pct>0])/len(lg)*100:.0f}%" if len(lg) else '-'
    swr = f"{len(sg[sg.net_pct>0])/len(sg)*100:.0f}%" if len(sg) else '-'
    print(f"{sym:<20}{len(lg):>5}{lwr:>7}{lg.r.sum() if len(lg) else 0:>7.1f}"
          f"{len(sg):>5}{swr:>7}{sg.r.sum() if len(sg) else 0:>7.1f}")

# ============ PART B: 4H SWING SETUP ============
print("\n" + "=" * 86)
print("PART B: 4H SWING SETUP (2 years, 9 pairs, futures)")
print("=" * 86)

with open('/home/azureuser/BotScripts/swing_4h_cache.pkl', 'rb') as f:
    S4 = pickle.load(f)


def ema(x, p): return x.ewm(span=p, adjust=False).mean()


def rsi14(x, p=14):
    d = x.diff(); g = d.clip(lower=0); l = -d.clip(upper=0)
    rma = lambda v, n: v.ewm(alpha=1/n, adjust=False).mean()
    return (100 - 100/(1 + rma(g, p)/rma(l, p).replace(0, np.nan))).fillna(50)


def atr14(df, p=14):
    tr = pd.concat([df.h-df.l, (df.h-df.c.shift()).abs(), (df.l-df.c.shift()).abs()],
                   axis=1).max(axis=1)
    return tr.ewm(alpha=1/p, adjust=False).mean()


def swing_backtest(df, sl_mult=2.0, tp_mult=4.0, max_hold=60,
                   fee=TAKER_FEE, slip=SLIPPAGE, fund=FUNDING_PER_8H,
                   rsi_lo=45, rsi_hi=75, vol_min=0.8, allow_long=True, allow_short=True):
    """4H swing: EMA20/50 cross + EMA200 regime + ADX + volume."""
    d = df.copy()
    d['ema20'] = ema(d.c, 20)
    d['ema50'] = ema(d.c, 50)
    d['ema200'] = ema(d.c, 200)
    d['rsi'] = rsi14(d.c)
    d['atr'] = atr14(d)
    d['vma'] = d.v.rolling(20).mean()
    # ADX
    up = d.h - d.h.shift(1); dn = d.l.shift(1) - d.l
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    ndm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([d.h-d.l, (d.h-d.c.shift()).abs(), (d.l-d.c.shift()).abs()], axis=1).max(axis=1)
    rma = lambda v, n: v.ewm(alpha=1/n, adjust=False).mean()
    atr_s = rma(tr, 14)
    pdi = 100*rma(pd.Series(pdm, index=d.index), 14)/atr_s.replace(0, np.nan)
    ndi = 100*rma(pd.Series(ndm, index=d.index), 14)/atr_s.replace(0, np.nan)
    dx = 100*(pdi-ndi).abs()/(pdi+ndi).replace(0, np.nan)
    d['adx'] = rma(dx.fillna(0), 14)

    trades = []
    for i in range(210, len(d)-1):
        r = d.iloc[i]; p = d.iloc[i-1]
        if not np.isfinite(r.atr) or r.atr <= 0:
            continue
        if r.adx < 25 or r.adx <= p.adx:
            continue
        if not (rsi_lo <= r.rsi <= rsi_hi):
            continue
        if r.v < r.vma * vol_min:
            continue
        cu = p.ema20 <= p.ema50 and r.ema20 > r.ema50
        cd = p.ema20 >= p.ema50 and r.ema20 < r.ema50
        entry = d.o.iloc[i+1]
        if allow_long and r.c > r.ema200 and cu:
            direction = 'LONG'
        elif allow_short and r.c < r.ema200 and cd:
            direction = 'SHORT'
        else:
            continue
        sl = entry - sl_mult*r.atr if direction == 'LONG' else entry + sl_mult*r.atr
        tp = entry + tp_mult*r.atr if direction == 'LONG' else entry - tp_mult*r.atr
        fwd = d.iloc[i+1:i+1+max_hold]
        if len(fwd) == 0:
            continue
        exit_px, reason, held = None, None, 0
        for k, (_, row) in enumerate(fwd.iterrows()):
            held = k+1
            if direction == 'LONG':
                if row.l <= sl: exit_px, reason = sl, 'SL'; break
                if row.h >= tp: exit_px, reason = tp, 'TP'; break
            else:
                if row.h >= sl: exit_px, reason = sl, 'SL'; break
                if row.l <= tp: exit_px, reason = tp, 'TP'; break
        if exit_px is None:
            exit_px, reason = fwd.c.iloc[-1], 'TIME'
        gross = (exit_px-entry)/entry if direction == 'LONG' else (entry-exit_px)/entry
        net = gross - 2*(fee+slip) - (held*4/24/8)*fund
        rm = (net*entry)/(sl_mult*r.atr)
        trades.append({'sym': df.attrs.get('sym',''), 't': d.index[i+1], 'dir': direction,
                       'net_pct': net, 'r': rm, 'reason': reason, 'held_h': held*4})
    return trades


for sym, df in S4.items():
    df.attrs['sym'] = sym

print(f"\n{'Scope':<22}{'n':>5}{'WR%':>8}{'ExpR':>9}{'TotalR':>9}{'PF':>7}{'MaxDD':>9}{'Hold':>8}")
print("-" * 77)
for label, kw in [
    ('4H COMBINED', {}),
    ('4H LONG-only', {'allow_short': False}),
    ('4H SHORT-only', {'allow_long': False}),
]:
    all_t = []
    for sym, df in S4.items():
        tr = swing_backtest(df, **kw)
        for t in tr: t['sym'] = sym
        all_t += tr
    r = s(all_t, label)
    if r['n']:
        print(f"{label:<22}{r['n']:>5}{r['wr']:>7.1f}%{r['exp']:>9.3f}{r['total']:>9.1f}"
              f"{r['pf']:>7.2f}{r['dd']:>9.1f}{r['avg_hold']/24:>7.1f}d")
    else:
        print(f"{label:<22}{'0':>5}  (no trades)")

# per-symbol 4H
print("\n--- 4H PER-SYMBOL (combined) ---")
print(f"\n{'Symbol':<20}{'n':>5}{'WR%':>8}{'ExpR':>9}{'TotalR':>9}{'PF':>7}")
print("-" * 58)
for sym, df in S4.items():
    tr = swing_backtest(df)
    if not tr:
        print(f"{sym:<20}{'0':>5}"); continue
    g = pd.DataFrame(tr)
    w = g[g.net_pct > 0]; l = g[g.net_pct <= 0]
    pf = w.net_pct.sum()/abs(l.net_pct.sum()) if len(l) and l.net_pct.sum() != 0 else float('inf')
    print(f"{sym:<20}{len(g):>5}{len(w)/len(g)*100:>7.1f}%{g.r.mean():>9.3f}{g.r.sum():>9.1f}{pf:>7.2f}")

print("\nDone.")
