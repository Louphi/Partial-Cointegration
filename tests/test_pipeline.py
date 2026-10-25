"""Regression checks for signal extraction and training/test isolation."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'PCI Framework'))
from PCIPipeline import Computation, Definition


class PipelineTests(unittest.TestCase):
    def test_future_missing_price_does_not_change_training_universe(self):
        dates = pd.to_datetime(['2019-01-01', '2022-12-30', '2023-01-03', '2023-06-30', '2023-07-01'])
        data = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [2, 3, np.nan, 5, 6]}, index=dates)
        original = data.copy()
        train, test = Definition.f_insample_outsample(data, 2023)
        self.assertEqual(list(train.columns), ['A', 'B'])
        self.assertEqual(list(test.index), list(dates[2:4]))
        self.assertTrue(pd.isna(test.loc[dates[2], 'B']))
        pd.testing.assert_frame_equal(data, original)
        changed = data.copy()
        changed.loc[dates[2], 'B'] = 4
        pd.testing.assert_frame_equal(train, Definition.f_insample_outsample(changed, 2023)[0])

    def test_signal_uses_transient_component_and_preserves_dates(self):
        dates = pd.date_range('2023-01-01', periods=5)
        x = np.array([10., 11., 12., 13., 14.])
        residual = np.array([0., 1., -.5, .75, -.25])
        prices = pd.DataFrame({'A': x, 'B': 2 * x + residual}, index=dates)
        pairs = Definition.pairDatabase(['A', 'B'])
        estimates = pd.DataFrame([[0, 2, .6, 1, 0]], index=['A/B'])
        mt, hedge = Computation.f_create_Mt(['A/B'], estimates, pairs, prices)
        # For a pure AR process, M equals the observed residual and R is zero.
        np.testing.assert_allclose(mt['A/B'], residual)
        pd.testing.assert_index_equal(mt.index, dates)
        pd.testing.assert_index_equal(hedge.index, dates)
        np.testing.assert_allclose(hedge['A/B'], 2 / prices['B'])

    def test_filter_random_walk_boundary_and_reconstruction(self):
        w = np.array([2., 3., -1., 4., 5.])
        c = Computation(w, w)
        m, r, errors = c.kalman_estimate(w, .7, 0, 1)
        np.testing.assert_allclose(m, 0)
        np.testing.assert_allclose(r, w)
        np.testing.assert_allclose(errors[1:], np.diff(w))
        m, r, _ = c.kalman_estimate(w, .7, 1, .4)
        np.testing.assert_allclose(m + r, w)

    def test_failed_optimizer_returns_missing_estimates(self):
        w = np.array([0., 1., -.2, 2., .5, 3.])
        with patch('PCIPipeline.opt.minimize', return_value=SimpleNamespace(success=False)):
            result = Computation(w, w).fit_mle(w)
        self.assertEqual(len(result), 3)
        self.assertTrue(np.isnan(result).all())

    def test_failed_pair_does_not_look_like_zero_estimates(self):
        pairs = Definition.pairDatabase(['A', 'B'])
        with patch.object(Computation, 'f_get_elligibility', side_effect=ValueError('invalid prices')):
            eligibility, estimates = Computation.f_compute_estimate(pairs['pair'], pd.DataFrame(), pairs)
        self.assertTrue(np.isnan(eligibility).all())
        self.assertTrue(np.isnan(estimates).all())

    def test_rolling_score_is_causal_and_handles_constant_windows(self):
        values = pd.Series([1., 1., 1., 2., 3., 4.])
        score = Computation.rolling_z_score(values, 3)
        self.assertTrue(np.isnan(score[:3]).all())
        self.assertAlmostEqual(score[-1], 1.)
        changed = values.copy()
        changed.iloc[-1] = 1000
        np.testing.assert_allclose(score[:-1], Computation.rolling_z_score(changed, 3)[:-1], equal_nan=True)
        with self.assertRaises(ValueError):
            Computation.rolling_z_score(values, 1)


if __name__ == '__main__':
    unittest.main()
