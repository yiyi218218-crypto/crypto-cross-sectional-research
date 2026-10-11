# Crypto cross-sectional research core

A data-free, reproducible research snapshot from a local Freqtrade-based study. The repository includes only the project's own Python research modules. It excludes the upstream Freqtrade checkout, exchange data, backtest archives, local configuration, secrets, and Git history.

The code constructs historical liquid universes, causal 4-hour-bar features, predeclared ranking policies, and a next-bar portfolio simulator with explicit turnover costs. It is research infrastructure, not an executable exchange strategy.

## Research status

The local study's final assessment was **no validated alpha and no live trading**. This snapshot intentionally does not quote profit or return claims. Its purpose is to show point-in-time universe construction, lookahead controls, and cost-aware evaluation.

## Install and inspect

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python -m unittest discover -s tests
```

Tests use generated fixtures only. No exchange connection, private API key, or historical dataset is required.
