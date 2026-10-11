"""Generated-data checks for point-in-time selection and execution timing."""

import unittest

import numpy as np
import pandas as pd

from research.cross_sectional.features import build_feature_panel
from research.cross_sectional.portfolio import simulate_signals
from research.cross_sectional.universe import BAR, build_membership, prepare_frames, snapshot


class SyntheticResearchTests(unittest.TestCase):
    @staticmethod
    def frames():
        times = pd.date_range("2024-01-01", periods=600, freq="4h", tz="UTC")
        result = {}
        for pair, start in (("BTC/USDT", 30_000.0), ("ETH/USDT", 2_000.0)):
            close = start * (1 + np.arange(len(times)) * 0.0001)
            result[pair] = pd.DataFrame({
                "timestamp": times,
                "open": close,
                "high": close * 1.001,
                "low": close * 0.999,
                "close": close,
                "volume": np.full(len(times), 10.0),
                "quote_volume": np.full(len(times), 100_000.0),
            })
        return result

    def test_future_data_cannot_change_historical_universe_or_features(self):
        raw = self.frames()
        time = raw["BTC/USDT"].iloc[550]["timestamp"]
        baseline = prepare_frames(raw)
        before = snapshot(baseline, time, 2)
        membership = build_membership(baseline, [time], [2])
        features = build_feature_panel(baseline, membership)
        at_time = features.loc[features["timestamp"] == time, ["pair", "momentum_7d"]].reset_index(drop=True)

        raw["ETH/USDT"].loc[raw["ETH/USDT"]["timestamp"] > time, "close"] *= 100
        altered = prepare_frames(raw)
        after = snapshot(altered, time, 2)
        changed_features = build_feature_panel(altered, membership)
        changed_at_time = changed_features.loc[changed_features["timestamp"] == time, ["pair", "momentum_7d"]].reset_index(drop=True)

        self.assertEqual(before["selected"], after["selected"])
        pd.testing.assert_frame_equal(at_time, changed_at_time)

    def test_signals_execute_next_bar_and_apply_turnover_cost(self):
        frames = prepare_frames(self.frames())
        time = frames["BTC/USDT"].index[550]
        result = simulate_signals(frames, {time: ["BTC/USDT"]}, time, time + 2 * BAR, fee_rate=0.001)
        self.assertEqual(result.equity.iloc[0]["timestamp"], time + BAR)
        self.assertGreater(result.equity.iloc[0]["cost"], 0)
        self.assertEqual(result.metrics["trade_rebalance_count"], 1)


if __name__ == "__main__":
    unittest.main()
