# Spot Robust Analysis Framework

Modular, extensible framework for Binance Spot analysis. Plug in modules:

- **correlation** — BTC lead/lag on altcoins (1-3 candle lag)
- **structure** — Market structure breaks (higher highs/lower lows)
- **volume_profile** — VWAP distance + volume spike detection
- **basis** — Spot-futures basis / funding rate (placeholder)

## Usage
```python
from spot_robust.core import SpotRobust, module_correlation_lead_lag, module_market_structure

sr = SpotRobust(["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"])
sr.register_module('correlation', module_correlation_lead_lag)
sr.register_module('structure', module_market_structure)

signals = sr.run('1h')
for s in signals:
    print(s.symbol, s.direction, s.strength, s.metadata)
```

## Extend
Add your own module:
```python
def my_module(symbol: str, data: dict) -> Optional[Signal]:
    # data['ohlcv'] = symbol's DataFrame
    # data['all'] = dict of all symbols' DataFrames
    return Signal(...)

sr.register_module('my_name', my_module)
```
