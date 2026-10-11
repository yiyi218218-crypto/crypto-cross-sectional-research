"""Deterministic long-only portfolio simulation with explicit turnover costs."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from .universe import BAR, CUTOFF


@dataclass
class Simulation:
    equity: pd.DataFrame
    attribution: pd.DataFrame
    metrics: dict


def build_price_lookup(frames: Mapping[str, pd.DataFrame]) -> dict[str, dict[str, dict[pd.Timestamp, float]]]:
    """Build a reusable exact-price lookup for repeated policy simulations."""
    return {pair: {field: frame[field].astype(float).to_dict() for field in ("open", "close")} for pair, frame in frames.items()}


def equal_weights(pairs: list[str]) -> dict[str, float]:
    """Return equal weights with a hard sum-to-one invariant."""
    unique = list(dict.fromkeys(pairs))
    if not unique:
        return {}
    weight = 1.0 / len(unique)
    result = {pair: weight for pair in unique}
    assert abs(sum(result.values()) - 1.0) < 1e-12
    return result


def _price(frames: Mapping[str, pd.DataFrame] | Mapping[str, Mapping[str, Mapping[pd.Timestamp, float]]], pair: str, timestamp: pd.Timestamp, field: str = "open", fallback: float | None = None) -> float:
    frame = frames[pair]
    try:
        if isinstance(frame, dict):
            value = float(frame[field].get(timestamp, np.nan))
        else:
            value = float(frame.loc[timestamp, field])
        if np.isfinite(value) and value > 0:
            return value
    except (KeyError, TypeError, ValueError):
        pass
    return float(fallback) if fallback is not None and np.isfinite(fallback) else float("nan")


def _metrics(equity: pd.DataFrame, attribution: pd.DataFrame, initial_cash: float, days: float, fee_rate: float) -> dict:
    if equity.empty:
        return {"net_return": np.nan, "cagr": np.nan, "annualized_vol": np.nan, "sharpe": np.nan, "sortino": np.nan, "max_drawdown": np.nan, "calmar": np.nan, "positive_month_ratio": np.nan, "positive_quarter_ratio": np.nan, "turnover": 0.0, "trade_rebalance_count": 0, "average_holding_period_hours": np.nan, "costs": 0.0, "cost_as_percent_of_gross_alpha": np.nan, "best_month": np.nan, "worst_month": np.nan, "top_asset_contribution": np.nan, "top_3_asset_contribution": np.nan, "cash_exposure": np.nan}
    values = equity.set_index("timestamp")["equity"].astype(float)
    returns = values.pct_change().dropna()
    net_return = float(values.iloc[-1] / initial_cash - 1.0)
    years = max(days / 365.25, 1 / 365.25)
    cagr = float((values.iloc[-1] / initial_cash) ** (1 / years) - 1.0) if values.iloc[-1] > 0 else -1.0
    ann_vol = float(returns.std(ddof=0) * np.sqrt(365.25)) if len(returns) > 1 else 0.0
    sharpe = float(returns.mean() / returns.std(ddof=0) * np.sqrt(365.25)) if len(returns) > 1 and returns.std(ddof=0) > 0 else 0.0
    downside = returns[returns < 0]
    sortino = float(returns.mean() / downside.std(ddof=0) * np.sqrt(365.25)) if len(downside) > 1 and downside.std(ddof=0) > 0 else 0.0
    drawdown = values / values.cummax() - 1.0
    max_dd = float(-drawdown.min())
    calmar = float(cagr / max_dd) if max_dd > 0 else 0.0
    monthly = values.resample("ME").last().pct_change().dropna()
    quarterly = values.resample("QE").last().pct_change().dropna()
    positive_month_ratio = float((monthly > 0).mean()) if len(monthly) else np.nan
    positive_quarter_ratio = float((quarterly > 0).mean()) if len(quarterly) else np.nan
    gross_pnl = float(attribution["gross_pnl"].sum()) if not attribution.empty else 0.0
    positive = attribution["gross_pnl"].clip(lower=0).sort_values(ascending=False) if not attribution.empty else pd.Series(dtype=float)
    positive_total = float(positive.sum())
    top_share = float(positive.iloc[0] / positive_total) if len(positive) and positive_total > 0 else 0.0
    top3_share = float(positive.head(3).sum() / positive_total) if len(positive) and positive_total > 0 else 0.0
    costs = float(equity["cost"].sum())
    turnover_dollars = float(equity["turnover_dollars"].sum())
    avg_equity = float(values.mean()) if len(values) else initial_cash
    hold = equity["holding_period_hours"].dropna()
    return {
        "net_return": net_return,
        "cagr": cagr,
        "annualized_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "positive_month_ratio": positive_month_ratio,
        "positive_quarter_ratio": positive_quarter_ratio,
        "turnover": turnover_dollars / avg_equity if avg_equity else 0.0,
        "trade_rebalance_count": int((equity["turnover_dollars"] > 0).sum()),
        "average_holding_period_hours": float(hold.mean()) if len(hold) else np.nan,
        "costs": costs,
        "cost_as_percent_of_gross_alpha": float(costs / gross_pnl) if gross_pnl > 0 else np.nan,
        "best_month": float(monthly.max()) if len(monthly) else np.nan,
        "worst_month": float(monthly.min()) if len(monthly) else np.nan,
        "top_asset_contribution": top_share,
        "top_3_asset_contribution": top3_share,
        "cash_exposure": float(equity["cash_exposure"].mean()),
        "gross_pnl": gross_pnl,
        "fees_per_side": fee_rate,
    }


def simulate_signals(frames: Mapping[str, pd.DataFrame], signals: Mapping[pd.Timestamp, list[str]], start: pd.Timestamp, end: pd.Timestamp, fee_rate: float, initial_cash: float = 10000.0, price_lookup: Mapping | None = None) -> Simulation:
    """Simulate signals calculated at T and executed at the next 4h bar.

    `signals` keys are bar-open timestamps.  The first execution is exactly
    one bar later.  The final mark is at the last allowed 4h bar at or before
    `end`, without performing a future rebalance.
    """
    start = pd.Timestamp(start).tz_convert("UTC")
    end = pd.Timestamp(end).tz_convert("UTC")
    assert end < CUTOFF
    ordered = sorted((pd.Timestamp(ts).tz_convert("UTC"), list(targets)) for ts, targets in signals.items() if start <= pd.Timestamp(ts).tz_convert("UTC") <= end and pd.Timestamp(ts).tz_convert("UTC") + BAR < CUTOFF)
    if not ordered:
        empty = pd.DataFrame(columns=["timestamp", "equity", "cash", "cost", "turnover_dollars", "cash_exposure", "holding_period_hours"])
        return Simulation(empty, pd.DataFrame(columns=["pair", "gross_pnl"]), _metrics(empty, pd.DataFrame(columns=["gross_pnl"]), initial_cash, 1.0, fee_rate))

    cash = float(initial_cash)
    positions: dict[str, float] = {}
    last_prices: dict[str, float] = {}
    entry_times: dict[str, pd.Timestamp] = {}
    holding_periods: list[float] = []
    rows: list[dict] = []
    asset_pnl: defaultdict[str, float] = defaultdict(float)
    previous_equity = cash
    prices_source = price_lookup or frames

    for signal_ts, requested in ordered:
        exec_ts = signal_ts + BAR
        lookup_pairs = set(positions) | set(requested)
        current_prices = {pair: _price(prices_source, pair, exec_ts, "open", last_prices.get(pair)) for pair in lookup_pairs}
        exact_execution_prices = {pair: _price(prices_source, pair, exec_ts, "open", None) for pair in lookup_pairs}
        current_values: dict[str, float] = {}
        for pair, units in positions.items():
            price = current_prices.get(pair, float("nan"))
            if not np.isfinite(price):
                price = last_prices.get(pair, float("nan"))
            if np.isfinite(price):
                current_values[pair] = units * price
                if pair in last_prices and np.isfinite(last_prices[pair]):
                    asset_pnl[pair] += units * (price - last_prices[pair])
                last_prices[pair] = price
        pre_equity = cash + sum(current_values.values())
        # A new order needs an actual next-bar price.  A prior mark may be
        # used only to value an already-held position through a source gap.
        available = [pair for pair in requested if np.isfinite(exact_execution_prices.get(pair, float("nan")))]
        weights = equal_weights(available)
        target_values = {pair: pre_equity * weight for pair, weight in weights.items()}
        all_pairs = set(current_values) | set(target_values)
        buys = sum(max(target_values.get(pair, 0.0) - current_values.get(pair, 0.0), 0.0) for pair in all_pairs)
        sells = sum(max(current_values.get(pair, 0.0) - target_values.get(pair, 0.0), 0.0) for pair in all_pairs)
        turnover = buys + sells
        cost = turnover * fee_rate
        post_equity = max(pre_equity - cost, 0.0)
        post_target_values = {pair: post_equity * weight for pair, weight in weights.items()}

        closed_holds: list[float] = []
        for pair in set(positions) - set(available):
            if pair in entry_times:
                duration = (exec_ts - entry_times.pop(pair)).total_seconds() / 3600.0
                holding_periods.append(duration)
                closed_holds.append(duration)
        for pair in set(available) - set(positions):
            entry_times[pair] = exec_ts
        positions = {pair: value / current_prices[pair] for pair, value in post_target_values.items() if current_prices[pair] > 0}
        cash = post_equity - sum(post_target_values.values())
        cash = max(cash, 0.0)
        for pair in positions:
            last_prices[pair] = current_prices[pair]
        rows.append({"timestamp": exec_ts, "equity": post_equity, "cash": cash, "cost": cost, "turnover_dollars": turnover, "cash_exposure": cash / post_equity if post_equity else 1.0, "holding_period_hours": float(np.mean(closed_holds)) if closed_holds else np.nan, "signal_timestamp": signal_ts})
        previous_equity = post_equity

    # Mark the surviving book at the final completed 4h bar in the requested period.
    mark_ts = min(end.floor("4h"), CUTOFF - BAR)
    if mark_ts < rows[-1]["timestamp"]:
        mark_ts = rows[-1]["timestamp"]
    final_prices = {pair: _price(prices_source, pair, mark_ts, "close", last_prices.get(pair)) for pair in positions}
    final_equity = cash + sum(units * price for pair, units in positions.items() if np.isfinite((price := final_prices.get(pair, float("nan")))))
    for pair, units in positions.items():
        price = final_prices.get(pair, float("nan"))
        if np.isfinite(price) and pair in last_prices:
            asset_pnl[pair] += units * (price - last_prices[pair])
    if mark_ts > rows[-1]["timestamp"]:
        rows.append({"timestamp": mark_ts, "equity": final_equity, "cash": cash, "cost": 0.0, "turnover_dollars": 0.0, "cash_exposure": cash / final_equity if final_equity else 1.0, "holding_period_hours": np.nan, "signal_timestamp": pd.NaT})
    equity = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    attribution = pd.DataFrame([{"pair": pair, "gross_pnl": pnl} for pair, pnl in sorted(asset_pnl.items())])
    metrics = _metrics(equity, attribution, initial_cash, max((equity["timestamp"].iloc[-1] - equity["timestamp"].iloc[0]).total_seconds() / 86400, 1.0), fee_rate)
    return Simulation(equity, attribution, metrics)
