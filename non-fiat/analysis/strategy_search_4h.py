#!/usr/bin/env python3
"""
4H Strategy Search — find what actually works on futures AND spot.
Tests multiple strategy families, reports trade count + expectancy + significance.
Cost model differs: spot (0.1% fee, no funding) vs futures (0.05% fee + funding).
"""
import sys, pickle
import pandas as pd
import numpy as np

np.random.seed(11)

FUT = pickle.load(open('/home/azureuser/BotScripts/swing_4h_cache.pkl', 'rb'))
SPOT = pickle.load(open('/home/azureuser/BotScripts/spot_4h_cache.pkl', 'rb'))

# Costs
SPOT_FEE, SPOT_SLIP = 0.0010, 0.0002     # 0.1% taker, no funding
FUT_FEE, FUT_SLIP, FUT_FUND = 0.0005, 0.0002, 0.0001


def ema(x, p): return x.ewm(span=p, adjust=False).mean()


def rsi14(x, p=14):
    d = x.diff(); g = d.clip(lower=0); l = -d.clip(upper=0)
    rma = lambda v, n: v.ewm(alpha=1/n, adjust=False).mean()
    return (100 - 100/(1 + rma(g, p)/rma(l, p).replace(0, np.nan))).fillna(50)


def atr14(df, p=14):
    tr = pd.concat([df.h-df.l, (df.h-df.c.shift()).abs(), (df.l-df.c.shift()).abs()],
                   axis=1).max(axis=1)
    return tr.ewm(alpha=1/p, adjust=False).mean()


def adx14(df, p=14):
    up = df.h - df.h.shift(1); dn = df.l.shift(1) - df.l
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    ndm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([df.h-df.l, (df.h-df.c.shift()).abs(), (df.l-df.c.shift()).abs()],
                   axis=1).max(axis=1)
    rma = lambda v, n: v.ewm(alpha=1/n, adjust=False).mean()
    atr_s = rma(tr, p)
    pdi = 100*rma(pd.Series(pdm, index=df.index), p)/atr_s.replace(0, np.nan)
    ndi = 100*rma(pd.Series(ndm, index=df.index), p)/atr_s.replace(0, np.nan)
    dx = 100*(pdi-ndi).abs()/(pdi+ndi).replace(0, np.nan)
    return rma(dx.fillna(0), p)


def simulate(d, signals, sl_mult, tp_mult, max_hold, fee, slip, fund):
    trades = []
    for i, direction in signals:
        if i+1 >= len(d): continue
        entry = d.o.iloc[i+1]; a = d.atr.iloc[i]
        if not np.isfinite(a) or a <= 0: continue
        sl = entry - sl_mult*a if direction == 'LONG' else entry + sl_mult*a
        tp = entry + tp_mult*a if direction == 'LONG' else entry - tp_mult*a
        fwd = d.iloc[i+1:i+1+max_hold]
        if len(fwd) == 0: continue
        ex, reason, held = None, None, 0
        for k, (_, row) in enumerate(fwd.iterrows()):
            held = k+1
            if direction == 'LONG':
                if row.l <= sl: ex, reason = sl, 'SL'; break
                if row.h >= tp: ex, reason = tp, 'TP'; break
            else:
                if row.h >= sl: ex, reason = sl, 'SL'; break
                if row.l <= tp: ex, reason = tp, 'TP'; break
        if ex is None: ex, reason = fwd.c.iloc[-1], 'TIME'
        gross = (ex-entry)/entry if direction == 'LONG' else (entry-ex)/entry
        net = gross - 2*(fee+slip) - (held*4/24/8)*fund
        trades.append({'dir': direction, 'net_pct': net, 'r': (net*entry)/(sl_mult*a),
                       'reason': reason, 'held_h': held*4, 't': d.index[i+1]})
    return trades


def prep(df):
    d = df.copy()
    d['ema20'] = ema(d.c, 20); d['ema50'] = ema(d.c, 50); d['ema200'] = ema(d.c, 200)
    d['ema9'] = ema(d.c, 9); d['ema21'] = ema(d.c, 21)
    d['rsi'] = rsi14(d.c); d['atr'] = atr14(d); d['adx'] = adx14(d)
    d['vma'] = d.v.rolling(20).mean()
    d['hh20'] = d.h.rolling(20).max(); d['ll20'] = d.l.rolling(20).min()
    d['hh50'] = d.h.rolling(50).max(); d['ll50'] = d.l.rolling(50).min()
    d['sma20'] = d.c.rolling(20).mean(); d['std20'] = d.c.rolling(20).std()
    d['bbu'] = d.sma20 + 2*d.std20; d['bbl'] = d.sma20 - 2*d.std20
    d['macd'] = ema(d.c, 12) - ema(d.c, 26); d['macd_sig'] = ema(d['macd'], 9)
    return d


# ---- STRATEGY DEFINITIONS (each returns list of (idx, 'LONG'|'SHORT')) ----

def strat_ema_cross(d):
    """EMA20/50 cross + EMA200 regime, loose filters."""
    out = []
    for i in range(210, len(d)-1):
        r, p = d.iloc[i], d.iloc[i-1]
        if r.adx < 20: continue
        if not (40 <= r.rsi <= 80): continue
        cu = p.ema20 <= p.ema50 and r.ema20 > r.ema50
        cd = p.ema20 >= p.ema50 and r.ema20 < r.ema50
        if r.c > r.ema200 and cu: out.append((i, 'LONG'))
        elif r.c < r.ema200 and cd: out.append((i, 'SHORT'))
    return out


def strat_ema_cross_vol(d):
    """EMA cross + ADX rising + volume (the strict one from before)."""
    out = []
    for i in range(210, len(d)-1):
        r, p = d.iloc[i], d.iloc[i-1]
        if r.adx < 25 or r.adx <= p.adx: continue
        if not (45 <= r.rsi <= 75): continue
        if r.v < r.vma * 0.8: continue
        cu = p.ema20 <= p.ema50 and r.ema20 > r.ema50
        cd = p.ema20 >= p.ema50 and r.ema20 < r.ema50
        if r.c > r.ema200 and cu: out.append((i, 'LONG'))
        elif r.c < r.ema200 and cd: out.append((i, 'SHORT'))
    return out


def strat_donchian20(d):
    """20-bar breakout + EMA200 regime."""
    out = []
    for i in range(210, len(d)-1):
        r = d.iloc[i]
        if r.c > d.hh20.iloc[i-1] and r.c > r.ema200: out.append((i, 'LONG'))
        elif r.c < d.ll20.iloc[i-1] and r.c < r.ema200: out.append((i, 'SHORT'))
    return out


def strat_donchian50(d):
    """50-bar breakout (slower, fewer but stronger)."""
    out = []
    for i in range(210, len(d)-1):
        r = d.iloc[i]
        if r.c > d.hh50.iloc[i-1] and r.c > r.ema200: out.append((i, 'LONG'))
        elif r.c < d.ll50.iloc[i-1] and r.c < r.ema200: out.append((i, 'SHORT'))
    return out


def strat_bb_meanrev(d):
    """Bollinger mean reversion: buy lower band, sell upper, in trend direction."""
    out = []
    for i in range(210, len(d)-1):
        r, p = d.iloc[i], d.iloc[i-1]
        if not (25 <= r.rsi <= 75): continue
        # long: price poked below lower band then closed back above, above EMA200
        if p.c < p.bbl and r.c > r.bbl and r.c > r.ema200: out.append((i, 'LONG'))
        elif p.c > p.bbu and r.c < r.bbu and r.c < r.ema200: out.append((i, 'SHORT'))
    return out


def strat_macd(d):
    """MACD cross + EMA200 regime."""
    out = []
    for i in range(210, len(d)-1):
        r, p = d.iloc[i], d.iloc[i-1]
        if r.adx < 20: continue
        cu = p.macd <= p.macd_sig and r.macd > r.macd_sig
        cd = p.macd >= p.macd_sig and r.macd < r.macd_sig
        if r.c > r.ema200 and cu: out.append((i, 'LONG'))
        elif r.c < r.ema200 and cd: out.append((i, 'SHORT'))
    return out


def strat_ema_cross_loose(d):
    """EMA20/50 cross, NO adx/rsi/vol filter — maximum trades."""
    out = []
    for i in range(210, len(d)-1):
        r, p = d.iloc[i], d.iloc[i-1]
        cu = p.ema20 <= p.ema50 and r.ema20 > r.ema50
        cd = p.ema20 >= p.ema50 and r.ema20 < r.ema50
        if r.c > r.ema200 and cu: out.append((i, 'LONG'))
        elif r.c < r.ema200 and cd: out.append((i, 'SHORT'))
    return out


STRATS = {
    'EMA cross loose (no filters)': strat_ema_cross_loose,
    'EMA cross + ADX20/RSI40-80': strat_ema_cross,
    'EMA cross + ADX25rise+vol': strat_ema_cross_vol,
    'Donchian 20 breakout': strat_donchian20,
    'Donchian 50 breakout': strat_donchian50,
    'Bollinger mean-reversion': strat_bb_meanrev,
    'MACD cross + ADX20': strat_macd,
}


def run_all(data, market):
    fee, slip, fund = (SPOT_FEE, SPOT_SLIP, 0) if market == 'SPOT' else (FUT_FEE, FUT_SLIP, FUT_FUND)
    rows = []
    for name, fn in STRATS.items():
        all_t = []
        for sym, df in data.items():
            d = prep(df)
            sigs = fn(d)
            tr = simulate(d, sigs, 2.0, 4.0, 60, fee, slip, fund)
            for t in tr: t['sym'] = sym
            all_t += tr
        if not all_t:
            rows.append({'strat': name, 'n': 0}); continue
        g = pd.DataFrame(all_t)
        w = g[g.net_pct > 0]; l = g[g.net_pct <= 0]
        cum = g.r.cumsum(); dd = (cum - cum.cummax()).min()
        pf = w.net_pct.sum()/abs(l.net_pct.sum()) if len(l) and l.net_pct.sum() != 0 else float('inf')
        rs = g.r.values
        ppos = (np.array([np.random.choice(rs, len(rs), True).mean() for _ in range(3000)]) > 0).mean()*100 if len(rs) > 5 else np.nan
        rows.append({'strat': name, 'n': len(g), 'wr': len(w)/len(g)*100, 'exp': g.r.mean(),
                     'total': g.r.sum(), 'pf': pf, 'dd': dd, 'ppos': ppos,
                     'hold_d': g.held_h.mean()/24,
                     'longs': len(g[g.dir == 'LONG']), 'shorts': len(g[g.dir == 'SHORT'])})
    return rows


for market, data in [('FUTURES', FUT), ('SPOT', SPOT)]:
    print("=" * 100)
    print(f"4H STRATEGY SEARCH — {market}  (2 years, {len(data)} pairs)")
    print("=" * 100)
    rows = run_all(data, market)
    print(f"\n{'Strategy':<30}{'n':>5}{'WR%':>7}{'ExpR':>8}{'TotalR':>8}{'PF':>6}{'MaxDD':>8}{'P>0':>6}{'Hold':>7}{'L':>4}{'S':>4}")
    print("-" * 93)
    for r in rows:
        if r.get('n', 0) == 0:
            print(f"{r['strat']:<30}{'0':>5}  (no trades)"); continue
        p = f"{r['ppos']:.0f}%" if not np.isnan(r.get('ppos', np.nan)) else '-'
        print(f"{r['strat']:<30}{r['n']:>5}{r['wr']:>6.1f}%{r['exp']:>8.3f}{r['total']:>8.1f}"
              f"{r['pf']:>6.2f}{r['dd']:>8.1f}{p:>6}{r['hold_d']:>6.1f}d{r['longs']:>4}{r['shorts']:>4}")
    print()
