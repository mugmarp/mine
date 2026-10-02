#!/usr/bin/env python3
"""
Spot Robust Core — Foundation for Binance Spot Analysis
Modular design: plug in correlation, orderbook, basis, structure, volume_profile
"""
import ccxt
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime, timezone, timedelta

EAT_TZ = timezone(timedelta(hours=3))

@dataclass
class Signal:
    symbol: str
    timestamp: datetime
    source: str  # 'correlation', 'orderbook', 'basis', 'structure', 'vp'
    direction: str  # 'LONG'/'SHORT'/'NEUTRAL'
    strength: float  # 0-1
    metadata: dict

class SpotRobust:
    def __init__(self, symbols: List[str] = None):
        self.symbols = symbols or ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "BNB/USDT"]
        self.ex = ccxt.binance({'enableRateLimit': True, 'timeout': 10000, 'options': {'defaultType': 'spot'}})
        self.modules = []

    def register_module(self, name: str, fn):
        """Register analysis module: fn(symbol, data_dict) -> Optional[Signal]"""
        self.modules.append((name, fn))

    def fetch_data(self, tf='1h', limit=500) -> Dict[str, pd.DataFrame]:
        """Fetch OHLCV for all symbols"""
        data = {}
        for sym in self.symbols:
            try:
                o = self.ex.fetch_ohlcv(sym, timeframe=tf, limit=limit)
                if o:
                    df = pd.DataFrame(o, columns=['t','o','h','l','c','v'])
                    for c in ['o','h','l','c','v']: df[c] = df[c].astype(float)
                    df['t'] = pd.to_datetime(df['t'], unit='ms', utc=True)
                    df.set_index('t', inplace=True)
                    data[sym] = df
            except Exception as e:
                print(f"Fetch error {sym}: {e}")
        return data

    def fetch_orderbook(self, symbol: str, limit=100) -> dict:
        """Fetch orderbook snapshot"""
        try:
            return self.ex.fetch_order_book(symbol, limit=limit)
        except: return {}

    def run(self, tf='1h') -> List[Signal]:
        data = self.fetch_data(tf)
        if len(data) < 2: return []
        signals = []
        for name, fn in self.modules:
            for sym in self.symbols:
                if sym not in data: continue
                try:
                    sig = fn(sym, {'ohlcv': data[sym], 'all': data})
                    if sig: signals.append(sig)
                except Exception as e:
                    print(f"Module {name} error on {sym}: {e}")
        return signals


# ===== EXAMPLE MODULES (plug into register_module) =====

def module_correlation_lead_lag(symbol: str, data: dict) -> Optional[Signal]:
    """BTC leads ETH/SOL/XRP by 1-3 candles on 1h?"""
    all_data = data['all']
    if 'BTC/USDT' not in all_data or symbol == 'BTC/USDT': return None
    btc = all_data['BTC/USDT'].c.pct_change().dropna()
    alt = data['ohlcv'].c.pct_change().dropna()
    # Find max correlation at lag 1-3
    best_corr, best_lag = 0, 0
    for lag in range(1, 4):
        c = btc.shift(lag).corr(alt)
        if c > best_corr: best_corr, best_lag = c, lag
    if best_corr > 0.6 and best_lag > 0:
        # BTC moved, alt likely to follow
        last_btc_ret = btc.iloc[-1]
        direction = 'LONG' if last_btc_ret > 0 else 'SHORT'
        return Signal(symbol, datetime.now(EAT_TZ), 'correlation', direction, best_corr,
                     {'lag': best_lag, 'btc_ret': last_btc_ret})
    return None


def module_market_structure(symbol: str, data: dict) -> Optional[Signal]:
    """Higher highs / lower lows on 1h + 4h alignment"""
    df = data['ohlcv']
    if len(df) < 100: return None
    c = df.c
    # 1h structure
    hh = c.rolling(20).max()
    ll = c.rolling(20).min()
    last_c = c.iloc[-1]
    break_up = last_c > hh.iloc[-2]
    break_down = last_c < ll.iloc[-2]
    if break_up:
        return Signal(symbol, datetime.now(EAT_TZ), 'structure', 'LONG', 0.7,
                     {'type': '20h_breakout', 'level': hh.iloc[-2]})
    if break_down:
        return Signal(symbol, datetime.now(EAT_TZ), 'structure', 'SHORT', 0.7,
                     {'type': '20h_breakdown', 'level': ll.iloc[-2]})
    return None


def module_volume_profile(symbol: str, data: dict) -> Optional[Signal]:
    """Simple VWAP distance + volume spike"""
    df = data['ohlcv']
    if len(df) < 50: return None
    # Session VWAP (rolling 24h)
    tp = (df.h + df.l + df.c) / 3
    vwap = (tp * df.v).rolling(24).sum() / df.v.rolling(24).sum()
    dist = (df.c.iloc[-1] - vwap.iloc[-1]) / vwap.iloc[-1]
    vol_ratio = df.v.iloc[-1] / df.v.rolling(24).mean().iloc[-1]
    if abs(dist) < 0.005 and vol_ratio > 1.5:  # near VWAP, volume spike
        direction = 'LONG' if dist > 0 else 'SHORT'
        return Signal(symbol, datetime.now(EAT_TZ), 'volume_profile', direction, min(vol_ratio/3, 1),
                     {'dist_pct': dist*100, 'vol_ratio': vol_ratio})
    return None


def module_spot_futures_basis(symbol: str, data: dict) -> Optional[Signal]:
    """Check spot-futures basis for funding arb / trend confirmation"""
    # Would need futures data - placeholder for now
    return None


# ===== USAGE EXAMPLE =====
if __name__ == "__main__":
    sr = SpotRobust()
    sr.register_module('correlation', module_correlation_lead_lag)
    sr.register_module('structure', module_market_structure)
    sr.register_module('volume_profile', module_volume_profile)
    # sr.register_module('basis', module_spot_futures_basis)

    sigs = sr.run('1h')
    for s in sigs:
        print(f"{s.timestamp.strftime('%H:%M')} | {s.symbol} | {s.source} | {s.direction} | str={s.strength:.2f} | {s.metadata}")
