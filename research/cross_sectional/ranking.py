"""Predeclared, low-dimensional cross-sectional ranking families."""

from __future__ import annotations

import math

import pandas as pd

from .features import feature_row
from .universe import parse_selected


def strategy_specs() -> list[dict]:
    specs = []
    for family in ["X0_EQUAL_WEIGHT_CONTROL", "X1_MOMENTUM_7D", "X2_MOMENTUM_14D", "X3_VOL_ADJUSTED_MOMENTUM", "X4_RELATIVE_STRENGTH_BTC", "X5_MOMENTUM_REGIME", "X6_MOMENTUM_LIQUIDITY"]:
        for size in [20, 30, 50]:
            positions = 5 if family == "X0_EQUAL_WEIGHT_CONTROL" else 3
            specs.append({"experiment_id": f"{family}_U{size}", "family": family, "universe_size": size, "position_count": positions, "rebalance": "24h", "execution_lag": "1x4h_bar"})
    assert len(specs) == 21
    return specs


def select_targets(panel: pd.DataFrame, membership_row, spec: dict) -> list[str]:
    selected = parse_selected(membership_row.selected_liquid_universe)
    if isinstance(panel, dict):
        rows = panel.get(membership_row.timestamp, pd.DataFrame()).reindex(selected).reset_index().copy()
    else:
        rows = feature_row(panel, membership_row.timestamp, selected).copy()
    if rows.empty:
        return []
    family = spec["family"]
    if family == "X5_MOMENTUM_REGIME":
        regime_ok = bool(rows["btc_trend_7d"].iloc[0] > 0 and rows["breadth_above_ma30"].iloc[0] >= 0.5)
        if not regime_ok:
            return []
    if family == "X6_MOMENTUM_LIQUIDITY":
        rows = rows.sort_values(["trailing_liquidity", "pair"], ascending=[False, True]).head(max(1, math.ceil(len(rows) / 2)))
    score = {
        "X0_EQUAL_WEIGHT_CONTROL": "trailing_liquidity",
        "X1_MOMENTUM_7D": "momentum_7d",
        "X2_MOMENTUM_14D": "momentum_14d",
        "X3_VOL_ADJUSTED_MOMENTUM": "vol_adjusted_momentum",
        "X4_RELATIVE_STRENGTH_BTC": "relative_return_vs_btc",
        "X5_MOMENTUM_REGIME": "momentum_7d",
        "X6_MOMENTUM_LIQUIDITY": "momentum_7d",
    }[family]
    rows = rows.dropna(subset=[score]).sort_values([score, "pair"], ascending=[False, True])
    return list(rows.head(spec["position_count"])["pair"])
