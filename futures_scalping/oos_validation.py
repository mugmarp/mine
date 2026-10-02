#!/usr/bin/env python3
"""
Out-of-sample validation + statistical significance for the futures scalp setup.
"""
import sys, pickle
import pandas as pd
import numpy as np

sys.path.insert(0, '/home/azureuser/BotScripts')
from robustness_analysis import evaluate, summarize, BASE, MAT

np.random.seed(42)

# ---------- 1. OUT OF SAMPLE SPLIT ----------
print("=" * 84)
print("1. OUT-OF-SAMPLE SPLIT (first 45 days = in-sample, last 45 days = out-of-sample)")
print("=" * 84)

all_t = evaluate(BASE)
g = pd.DataFrame(all_t)
g['t'] = pd.to_datetime(g['t'])
g = g.sort_values('t')
mid = g['t'].quantile(0.5)
IS = g[g.t <= mid]
OOS = g[g.t > mid]

print(f"\nSplit at: {mid}")
print(f"\n{'Scope':<16}{'n':>5}{'WR%':>8}{'ExpR':>9}{'TotalR':>9}{'PF':>7}")
print("-" * 54)
for label, sub in [('IN-SAMPLE (1st half)', IS), ('OUT-OF-SAMPLE (2nd)', OOS), ('FULL PERIOD', g)]:
    if len(sub) == 0: continue
    wins = sub[sub.net_pct > 0]; losses = sub[sub.net_pct <= 0]
    pf = wins.net_pct.sum()/abs(losses.net_pct.sum()) if len(losses) and losses.net_pct.sum() != 0 else float('inf')
    print(f"{label:<16}{len(sub):>5}{len(wins)/len(sub)*100:>7.1f}%{sub.r.mean():>9.3f}{sub.r.sum():>9.1f}{pf:>7.2f}")

# ---------- 2. BOOTSTRAP CONFIDENCE INTERVAL ----------
print("\n" + "=" * 84)
print("2. BOOTSTRAP 95% CI ON EXPECTANCY (10,000 resamples)")
print("=" * 84)
rs = g.r.values
n = len(rs)
boot = np.array([np.random.choice(rs, n, replace=True).mean() for _ in range(10000)])
lo, hi = np.percentile(boot, [2.5, 97.5])
print(f"\n  Mean expectancy: {rs.mean():.4f}R")
print(f"  95% CI: [{lo:.4f}R, {hi:.4f}R]")
print(f"  P(expectancy > 0): {(boot > 0).mean()*100:.1f}%")
if lo < 0 < hi:
    print("  >>> CI STRADDLES ZERO — edge NOT statistically significant at 95%")
else:
    print("  >>> CI excludes zero — statistically significant")

# ---------- 3. T-TEST ----------
from math import sqrt
se = rs.std(ddof=1) / sqrt(n)
t_stat = rs.mean() / se
print(f"\n  t-statistic: {t_stat:.3f}  (n={n}, se={se:.4f})")
print(f"  Rough threshold for p<0.05: |t| > 1.98  ->  {'PASS' if abs(t_stat) > 1.98 else 'FAIL'}")

# ---------- 4. PER-TRADE COST IMPACT ----------
print("\n" + "=" * 84)
print("3. WHERE DOES THE MONEY GO? (gross vs costs)")
print("=" * 84)
GROSS = evaluate(BASE, fee=0, slip=0, fund=0)
gg = pd.DataFrame(GROSS)
print(f"\n  Gross expectancy:   {gg.r.mean():.4f}R  (total {gg.r.sum():.1f}R)")
print(f"  Net expectancy:     {g.r.mean():.4f}R  (total {g.r.sum():.1f}R)")
print(f"  Cost drag:          {gg.r.mean() - g.r.mean():.4f}R per trade ({(1 - g.r.mean()/gg.r.mean())*100:.0f}% of gross)")
print(f"  Avg trade size (gross): {gg.net_pct.mean()*100:.3f}%")
print(f"  Round-trip cost:        {(gg.net_pct.mean() - g.net_pct.mean())*100:.3f}%")
print(f"  Avg holding time:       {g.held_h.mean():.2f}h")

# ---------- 5. STATISTICAL POWER CHECK ----------
print("\n" + "=" * 84)
print("4. SAMPLE SIZE REALITY CHECK")
print("=" * 84)
trades_per_day = n / 90
print(f"\n  Total trades: {n} over 90 days = {trades_per_day:.2f} trades/day")
print(f"  To detect expectancy 0.20R with 95% confidence (std={rs.std():.2f}R):")
needed = (1.96 * rs.std() / 0.20) ** 2
print(f"    Need ~{int(needed)} trades = ~{int(needed/trades_per_day)} days = ~{needed/trades_per_day/365:.1f} years")
print(f"    (Current: {n} trades)")

# ---------- 6. SYMBOL FILTER TEST ----------
print("\n" + "=" * 84)
print("5. WHAT IF WE DROP THE LOSERS? (BTC & DOGE were net negative)")
print("=" * 84)
print(f"\n{'Universe':<26}{'n':>5}{'WR%':>8}{'ExpR':>9}{'TotalR':>9}{'PF':>7}")
print("-" * 64)
universes = {
    'All 8 pairs': None,
    'Drop BTC & DOGE': ['BTC/USDT:USDT', 'DOGE/USDT:USDT'],
    'Majors only (BTC,ETH,SOL,XRP)': ['ADA/USDT:USDT','DOGE/USDT:USDT','PENGU/USDT:USDT','PEOPLE/USDT:USDT'],
}
for label, drop in universes.items():
    sub = g if drop is None else g[~g.sym.isin(drop)]
    if len(sub) == 0: continue
    wins = sub[sub.net_pct > 0]; losses = sub[sub.net_pct <= 0]
    pf = wins.net_pct.sum()/abs(losses.net_pct.sum()) if len(losses) and losses.net_pct.sum() != 0 else float('inf')
    print(f"{label:<26}{len(sub):>5}{len(wins)/len(sub)*100:>7.1f}%{sub.r.mean():>9.3f}{sub.r.sum():>9.1f}{pf:>7.2f}")

# ---------- 7. TIME-BASED (long vs short by month) ----------
print("\n" + "=" * 84)
print("6. DIRECTION BY MONTH (is short edge persistent or a one-off?)")
print("=" * 84)
g['month'] = g.t.dt.strftime('%Y-%m')
print(f"\n{'Month':<9}{'Dir':<7}{'n':>5}{'WR%':>8}{'ExpR':>9}{'TotalR':>9}")
print("-" * 47)
for (m, d), sub in g.groupby(['month', 'dir']):
    wins = sub[sub.net_pct > 0]
    print(f"{m:<9}{d:<7}{len(sub):>5}{len(wins)/len(sub)*100:>7.1f}%{sub.r.mean():>9.3f}{sub.r.sum():>9.1f}")

print("\nDone.")
