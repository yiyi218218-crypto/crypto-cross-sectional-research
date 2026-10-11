"""Historical, point-in-time liquid-universe construction."""

from __future__ import annotations

import json
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


CUTOFF = pd.Timestamp("2025-01-01T00:00:00Z")
BAR = pd.Timedelta(hours=4)
MIN_LISTING_AGE = pd.Timedelta(days=90)
LIQUIDITY_BARS = 180


def prepare_frames(frames: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Normalize source frames and index them by UTC bar-open timestamp."""
    result: dict[str, pd.DataFrame] = {}
    for pair, frame in frames.items():
        required = {"timestamp", "open", "high", "low", "close", "volume", "quote_volume"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{pair}: missing columns {sorted(missing)}")
        clean = frame.copy()
        clean["timestamp"] = pd.to_datetime(clean["timestamp"], utc=True)
        clean = clean.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        clean = clean[clean["timestamp"] < CUTOFF].copy()
        if not clean.empty:
            assert clean["timestamp"].max() < CUTOFF
        result[pair] = clean.set_index("timestamp")
    return result


def _record(frame: pd.DataFrame, timestamp: pd.Timestamp) -> tuple[str, float, str]:
    """Return (status, trailing quote volume, first timestamp) at timestamp."""
    if frame.empty:
        return "NO_DATA", float("nan"), ""
    first = frame.index[0]
    if timestamp < first:
        return "NOT_LISTED", float("nan"), first.isoformat()
    if timestamp < first + MIN_LISTING_AGE:
        return "LISTING_AGE", float("nan"), first.isoformat()
    if timestamp not in frame.index:
        return "MISSING_CURRENT_BAR", float("nan"), first.isoformat()
    history = frame.loc[:timestamp].tail(LIQUIDITY_BARS)
    if len(history) < LIQUIDITY_BARS:
        return "INSUFFICIENT_LIQUIDITY_HISTORY", float("nan"), first.isoformat()
    current = frame.loc[timestamp]
    if not np.isfinite(float(current["close"])) or not np.isfinite(float(current["volume"])):
        return "INVALID_CURRENT_BAR", float("nan"), first.isoformat()
    quote = pd.to_numeric(history["quote_volume"], errors="coerce")
    if quote.isna().any() or float(quote.sum()) <= 0:
        return "INVALID_LIQUIDITY_HISTORY", float("nan"), first.isoformat()
    return "ELIGIBLE", float(quote.sum()), first.isoformat()


def snapshot(frames: Mapping[str, pd.DataFrame], timestamp: pd.Timestamp, universe_size: int) -> dict:
    """Build one historical universe snapshot using only data through timestamp."""
    timestamp = pd.Timestamp(timestamp)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    assert timestamp < CUTOFF
    records, eligible = _evaluate(frames, timestamp)
    eligible.sort(key=lambda value: (-value[1], value[0]))
    selected = [pair for pair, _ in eligible[:universe_size]]
    for rank, (pair, liquidity) in enumerate(eligible, start=1):
        records[pair]["liquidity_rank"] = rank
        if pair in selected:
            records[pair]["status"] = "SELECTED"
    excluded = {pair: rec["status"] for pair, rec in records.items() if pair not in selected}
    return {
        "timestamp": timestamp,
        "universe_size": int(universe_size),
        "eligible_pair_count": len(eligible),
        "selected": selected,
        "excluded": excluded,
        "records": records,
    }


def _evaluate(frames: Mapping[str, pd.DataFrame], timestamp: pd.Timestamp) -> tuple[dict[str, dict], list[tuple[str, float]]]:
    records: dict[str, dict] = {}
    eligible: list[tuple[str, float]] = []
    for pair, frame in sorted(frames.items()):
        status, liquidity, first = _record(frame, timestamp)
        records[pair] = {
            "pair": pair,
            "status": status,
            "trailing_liquidity": liquidity,
            "listing_start": first,
        }
        if status == "ELIGIBLE":
            eligible.append((pair, liquidity))
    return records, eligible


def build_membership(frames: Mapping[str, pd.DataFrame], timestamps: Iterable[pd.Timestamp], universe_sizes: Iterable[int]) -> pd.DataFrame:
    """Create compact snapshot rows for each rebalance and universe size."""
    rows: list[dict] = []
    for timestamp in timestamps:
        timestamp = pd.Timestamp(timestamp)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")
        records, eligible = _evaluate(frames, timestamp)
        eligible.sort(key=lambda value: (-value[1], value[0]))
        for size in universe_sizes:
            selected = [pair for pair, _ in eligible[: int(size)]]
            excluded = {pair: rec["status"] for pair, rec in records.items() if pair not in selected}
            rows.append(
                {
                    "timestamp": timestamp,
                    "universe_size": int(size),
                    "eligible_pair_count": len(eligible),
                    "selected_liquid_universe": json.dumps(selected, separators=(",", ":")),
                    "excluded_pairs": json.dumps(excluded, sort_keys=True, separators=(",", ":")),
                }
            )
    result = pd.DataFrame(rows)
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
    assert result["timestamp"].max() < CUTOFF
    return result.sort_values(["timestamp", "universe_size"]).reset_index(drop=True)


def parse_selected(value: str) -> list[str]:
    return list(json.loads(value))


def parse_excluded(value: str) -> dict[str, str]:
    return dict(json.loads(value))
