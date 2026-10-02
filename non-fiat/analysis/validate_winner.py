#!/usr/bin/env python3
"""Deep validation of the winning 4H config: OOS, long/short split, per-symbol, sensitivity."""
import sys, pickle
import pandas as pd
import numpy as np

np.random.seed(23)

FUT = pickle.load(open('/home/azureuser/BotScripts/swing_4h_cache.pkl', 'rb'))
SPOT = pickle.load(open('/home/azureuser/BotScripts/spot_4h_cache.pkl', 'rb'))
SPOT_FEE, SPOT_SLIP = 0.0010, 0.0002
FUT_FEE, FUT_SLIP, FUT_FUND = 0.0005, 0.0002, 0.0001

import importlib.util
spec = importlib.util.spec_from_file_location("ss", "/home/azureuser/BotScripts/strategy_search_4h.py")
# instead just redefine the needed pieces
def ema(x, p): return x.ewm(span=p, adjust=False).mean()
def rsi14(x, p=14):
    d = x.diff(); g = d.clip(lower=0); l = -d.clip(upper=0)
    rma = lambda v, n: v.ewm(alpha=1/n, adjust=False).mean()
    return (100 - 100/(1 + rma(g, p)/rma(l, p).replace(0, np.nan))).fillna(50)
def atr14(df, p=14):
    tr = pd.concat([df.h-df.l, (df.h-df.c.shift()).abs(), (df.l-df.c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/p, adjust=False).mean()
def adx14(df, p=14):
    up = df.h - df.h.shift(1); dn = df.l.shift(1) - df.l
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    ndm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([df.h-df.l, (df.h-df.c.shift()).abs(), (df.l-df.c.shift()).abs()], axis=1).max(axis=1)
    rma = lambda v, n: v.ewm(alpha=1/n, adjust=False).mean()
    atr_s = rma(tr, p)
    pdi = 100*rma(pd.Series(pdm, index=df.index), p)/atr_s.replace(0, np.nan)
    ndi = 100*rma(pd.Series(ndm, index=df.index), p)/atr_s.replace(0, np.nan)
    dx = 100*(pdi-ndi).abs()/(pdi+ndi).replace(0, np.nan)
    return rma(dx.fillna(0), p)

def prep(df):
    d = df.copy()
    d['ema20']=ema(d.c,20); d['ema50']=ema(d.c,50); d['ema200']=ema(d.c,200)
    d['rsi']=rsi14(d.c); d['atr']=atr14(d); d['adx']=adx14(d); d['vma']=d.v.rolling(20).mean()
    return d

def signals(d, adx_min=20, rsi_lo=40, rsi_hi=80, allow_long=True, allow_short=True):
    out=[]
    for i in range(210, len(d)-1):
        r,p = d.iloc[i], d.iloc[i-1]
        if r.adx < adx_min: continue
        if not (rsi_lo <= r.rsi <= rsi_hi): continue
        cu = p.ema20 <= p.ema50 and r.ema20 > r.ema50
        cd = p.ema20 >= p.ema50 and r.ema20 < r.ema50
        if allow_long and r.c > r.ema200 and cu: out.append((i,'LONG'))
        elif allow_short and r.c < r.ema200 and cd: out.append((i,'SHORT'))
    return out

def simulate(d, sigs, sl_m=2.0, tp_m=4.0, mh=60, fee=0.0005, slip=0.0002, fund=0.0001):
    tr=[]
    for i, dr in sigs:
        if i+1 >= len(d): continue
        e=d.o.iloc[i+1]; a=d.atr.iloc[i]
        if not np.isfinite(a) or a<=0: continue
        sl = e-sl_m*a if dr=='LONG' else e+sl_m*a
        tp = e+tp_m*a if dr=='LONG' else e-tp_m*a
        fwd=d.iloc[i+1:i+1+mh]
        if len(fwd)==0: continue
        ex,rs,hd=None,None,0
        for k,(_,row) in enumerate(fwd.iterrows()):
            hd=k+1
            if dr=='LONG':
                if row.l<=sl: ex,rs=sl,'SL'; break
                if row.h>=tp: ex,rs=tp,'TP'; break
            else:
                if row.h>=sl: ex,rs=sl,'SL'; break
                if row.l<=tp: ex,rs=tp,'TP'; break
        if ex is None: ex,rs=fwd.c.iloc[-1],'TIME'
        g=(ex-e)/e if dr=='LONG' else (e-ex)/e
        net=g-2*(fee+slip)-(hd*4/24/8)*fund
        tr.append({'dir':dr,'net_pct':net,'r':(net*e)/(sl_m*a),'reason':rs,'held_h':hd*4,'t':d.index[i+1]})
    return tr

def collect(data, market, **kw):
    fee,slip,fund = (SPOT_FEE,SPOT_SLIP,0) if market=='SPOT' else (FUT_FEE,FUT_SLIP,FUT_FUND)
    all_t=[]
    for sym,df in data.items():
        d=prep(df); sigs=signals(d, **kw)
        tr=simulate(d, sigs, fee=fee, slip=slip, fund=fund)
        for t in tr: t['sym']=sym
        all_t+=tr
    return all_t

def summ(trades, label):
    if not trades: return {'label':label,'n':0}
    g=pd.DataFrame(trades); w=g[g.net_pct>0]; l=g[g.net_pct<=0]
    cum=g.r.cumsum(); dd=(cum-cum.cummax()).min()
    pf=w.net_pct.sum()/abs(l.net_pct.sum()) if len(l) and l.net_pct.sum()!=0 else float('inf')
    rs=g.r.values
    boot=np.array([np.random.choice(rs,len(rs),True).mean() for _ in range(5000)]) if len(rs)>5 else None
    return {'label':label,'n':len(g),'wr':len(w)/len(g)*100,'exp':g.r.mean(),'total':g.r.sum(),
            'pf':pf,'dd':dd,'ci':np.percentile(boot,[2.5,97.5]) if boot is not None else None,
            'ppos':(boot>0).mean()*100 if boot is not None else np.nan,
            'hold':g.held_h.mean()/24}

for market, data in [('FUTURES',FUT),('SPOT',SPOT)]:
    print("="*94)
    print(f"DEEP VALIDATION — {market} — EMA20/50 + EMA200 + ADX20 + RSI40-80")
    print("="*94)
    # combined + split
    print(f"\n{'Setup':<24}{'n':>5}{'WR%':>7}{'ExpR':>8}{'TotalR':>8}{'PF':>6}{'MaxDD':>8}{'P>0':>6}{'95% CI':>20}")
    print("-"*92)
    for lab, kw in [('COMBINED',{}), ('LONG-only',{'allow_short':False}), ('SHORT-only',{'allow_long':False})]:
        t=collect(data,market,**kw); s=summ(t,lab)
        if s['n']==0: print(f"{lab:<24}{'0':>5}"); continue
        ci=f"[{s['ci'][0]:+.2f},{s['ci'][1]:+.2f}]" if s['ci'] is not None else '-'
        print(f"{lab:<24}{s['n']:>5}{s['wr']:>6.1f}%{s['exp']:>8.3f}{s['total']:>8.1f}{s['pf']:>6.2f}{s['dd']:>8.1f}{s['ppos']:>5.0f}%{ci:>20}")

    # OOS split
    t=collect(data,market); g=pd.DataFrame(t); g['t']=pd.to_datetime(g.t); g=g.sort_values('t')
    mid=g.t.quantile(0.5)
    print(f"\n  OUT-OF-SAMPLE (split {mid.date()}):")
    for lab,sub in [('1st half',g[g.t<=mid]),('2nd half',g[g.t>mid])]:
        if len(sub)==0: continue
        w=sub[sub.net_pct>0]
        print(f"    {lab:<10} n={len(sub):<4} WR={len(w)/len(sub)*100:.1f}%  Exp={sub.r.mean():+.3f}R  Total={sub.r.sum():+.1f}R")

    # per-symbol
    print(f"\n  {'Symbol':<16}{'n':>5}{'WR%':>7}{'ExpR':>8}{'TotalR':>8}{'L':>4}{'S':>4}")
    print("  "+"-"*50)
    for sym in data:
        sub=g[g.sym==sym]
        if len(sub)==0: print(f"  {sym:<16}{'0':>5}"); continue
        w=sub[sub.net_pct>0]
        print(f"  {sym:<16}{len(sub):>5}{len(w)/len(sub)*100:>6.1f}%{sub.r.mean():>8.3f}{sub.r.sum():>8.1f}{len(sub[sub.dir=='LONG']):>4}{len(sub[sub.dir=='SHORT']):>4}")

    # sensitivity
    print(f"\n  PARAMETER SENSITIVITY:")
    print(f"    {'adx':>4}{'rsi_lo':>7}{'rsi_hi':>7}{'n':>5}{'WR%':>7}{'ExpR':>8}{'PF':>6}")
    print("    "+"-"*44)
    for adx in [15,20,25]:
        for rl,rh in [(35,80),(40,80),(45,75)]:
            t=collect(data,market,adx_min=adx,rsi_lo=rl,rsi_hi=rh)
            if not t: continue
            s=summ(t,'')
            print(f"    {adx:>4}{rl:>7}{rh:>7}{s['n']:>5}{s['wr']:>6.1f}%{s['exp']:>8.3f}{s['pf']:>6.2f}")
    print()
