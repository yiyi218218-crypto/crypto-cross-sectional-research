"""Small causal cross-sectional feature set."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .universe import CUTOFF, parse_selected


ASSET_FEATURES = [
    "momentum_24h",
    "momentum_72h",
    "momentum_7d",
    "momentum_14d",
    "momentum_30d",
    "volatility_7d",
    "volatility_30d",
    "vol_adjusted_momentum",
    "relative_return_vs_btc",
    "relative_return_vs_universe_median",
    "drawdown_recent_high",
    "trailing_liquidity",
]
CONTEXT_FEATURES = ["btc_trend_7d", "btc_volatility_7d", "btc_drawdown_30d", "breadth_above_ma30"]


def _asset_features(pair: str, frame: pd.DataFrame) -> pd.DataFrame:
    close = pd.to_numeric(frame["close"], errors="coerce")
    quote = pd.to_numeric(frame["quote_volume"], errors="coerce")
    log_return = np.log(close).diff()
    out = pd.DataFrame(index=frame.index)
    out["pair"] = pair
    out["momentum_24h"] = close / close.shift(6) - 1.0
    out["momentum_72h"] = close / close.shift(18) - 1.0
    out["momentum_7d"] = close / close.shift(42) - 1.0
    out["momentum_14d"] = close / close.shift(84) - 1.0
    out["momentum_30d"] = close / close.shift(180) - 1.0
    out["volatility_7d"] = log_return.rolling(42, min_periods=42).std(ddof=0) * np.sqrt(6.0)
    out["volatility_30d"] = log_return.rolling(180, min_periods=180).std(ddof=0) * np.sqrt(6.0)
    out["vol_adjusted_momentum"] = out["momentum_7d"] / out["volatility_7d"].replace(0, np.nan)
    out["drawdown_recent_high"] = close / close.rolling(180, min_periods=180).max() - 1.0
    out["trailing_liquidity"] = quote.rolling(180, min_periods=180).sum()
    out["close"] = close
    out["above_ma30"] = close > close.rolling(180, min_periods=180).mean()
    return out.reset_index(names="timestamp")


def build_feature_panel(frames: dict[str, pd.DataFrame], membership: pd.DataFrame) -> pd.DataFrame:
    """Build causal asset and global-context features.

    Every rolling operation includes only rows at or before the feature
    timestamp.  Orders use the next 4h bar, so the inclusion of the current
    completed bar is executable under the declared timing convention.
    """
    pieces = [_asset_features(pair, frame) for pair, frame in sorted(frames.items())]
    panel = pd.concat(pieces, ignore_index=True)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"], utc=True)
    panel = panel[panel["timestamp"] < CUTOFF].copy()
    selected50 = membership[membership["universe_size"] == membership["universe_size"].max()].copy()
    selected_map = {row.timestamp: set(parse_selected(row.selected_liquid_universe)) for row in selected50.itertuples()}
    pivot = panel.pivot(index="timestamp", columns="pair", values="momentum_7d")
    btc_mom = panel.loc[panel["pair"] == "BTC/USDT", ["timestamp", "momentum_7d", "volatility_7d", "drawdown_recent_high"]].rename(columns={"momentum_7d": "btc_trend_7d", "volatility_7d": "btc_volatility_7d", "drawdown_recent_high": "btc_drawdown_30d"})
    panel = panel.merge(btc_mom, on="timestamp", how="left", validate="many_to_one")
    panel["relative_return_vs_btc"] = panel["momentum_7d"] - panel["btc_trend_7d"]
    medians = []
    breadths = []
    for timestamp, group in panel.groupby("timestamp", sort=False):
        selected = selected_map.get(timestamp, set())
        values = group[group["pair"].isin(selected)]["momentum_7d"].dropna()
        above = group[group["pair"].isin(selected)]["above_ma30"].dropna()
        medians.append((timestamp, float(values.median()) if len(values) else np.nan))
        breadths.append((timestamp, float(above.mean()) if len(above) else np.nan))
    median_frame = pd.DataFrame(medians, columns=["timestamp", "universe_median_7d"])
    breadth_frame = pd.DataFrame(breadths, columns=["timestamp", "breadth_above_ma30"])
    panel = panel.merge(median_frame, on="timestamp", how="left", validate="many_to_one")
    panel = panel.merge(breadth_frame, on="timestamp", how="left", validate="many_to_one")
    panel["relative_return_vs_universe_median"] = panel["momentum_7d"] - panel["universe_median_7d"]
    panel = panel.drop(columns=["universe_median_7d", "above_ma30"])
    panel = panel.sort_values(["timestamp", "pair"]).reset_index(drop=True)
    assert set(ASSET_FEATURES).issubset(panel.columns)
    assert panel["timestamp"].max() < CUTOFF
    return panel


def feature_row(panel: pd.DataFrame, timestamp: pd.Timestamp, pairs: list[str]) -> pd.DataFrame:
    timestamp = pd.Timestamp(timestamp)
    subset = panel[(panel["timestamp"] == timestamp) & panel["pair"].isin(pairs)].copy()
    return subset
