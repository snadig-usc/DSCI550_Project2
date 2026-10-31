"""Independent numerical checks for scoring and time-series summaries.

Run: .venv/Scripts/python.exe -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from numerai_tools.scoring import neutral_correlation, numerai_corr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_experiments import rank_ensemble, summarize_scores


class MetricsTests(unittest.TestCase):
    def test_corr_matches_independent_formula_with_ties(self):
        pred = np.array([.1, .3, .3, .7, .8, .9, .9, 1.0])
        target = np.array([0, .25, .5, .25, .75, 1, .5, .75])
        gaussian = norm.ppf((rankdata(pred, method="average") - .5) / len(pred))
        pred_power = np.sign(gaussian) * np.abs(gaussian) ** 1.5
        centered = target - target.mean()
        target_power = np.sign(centered) * np.abs(centered) ** 1.5
        expected = np.corrcoef(pred_power, target_power)[0, 1]
        actual = numerai_corr(pd.DataFrame({"p": pred}), pd.Series(target, name="target"),
                              max_filtered_index_ratio=0)["p"]
        self.assertAlmostEqual(expected, actual, places=12)

    def test_ncorr_matches_independent_projection(self):
        rng = np.random.default_rng(17)
        n = 100
        neutralizers = rng.normal(size=(n, 3))
        prediction = rng.normal(size=n) + neutralizers[:, 0]
        target = rng.choice([0, .25, .5, .75, 1], size=n)
        transformed = norm.ppf((rankdata(prediction) - .5) / n)
        design = np.column_stack([neutralizers, np.ones(n)])
        residual = transformed - design @ np.linalg.pinv(design, rcond=1e-6) @ transformed
        expected = np.corrcoef(residual, target - target.mean())[0, 1]
        actual = neutral_correlation(pd.DataFrame({"p": prediction}), pd.DataFrame(neutralizers),
                                     pd.Series(target, name="target"), max_filtered_index_ratio=0)["p"]
        self.assertAlmostEqual(expected, actual, places=12)

    def test_official_scores_preserve_permuted_key_alignment(self):
        idx = pd.Index(["z", "a", "d", "b", "c"])
        p = pd.DataFrame({"p": [.2, .9, .1, .7, .3]}, index=idx)
        y = pd.Series([.25, 1., 0., .75, .5], index=idx, name="target")
        a = numerai_corr(p, y, max_filtered_index_ratio=0)["p"]
        b = numerai_corr(p.iloc[::-1], y.sample(frac=1, random_state=2), max_filtered_index_ratio=0)["p"]
        self.assertAlmostEqual(a, b, places=12)

    def test_population_std_and_initial_loss_drawdown(self):
        result = summarize_scores(pd.Series([-.2, .1, -.15, .05]))
        self.assertAlmostEqual(result["std_ddof0"], np.std([-.2, .1, -.15, .05]), places=12)
        self.assertAlmostEqual(result["max_drawdown"], .25, places=12)
        self.assertEqual(result["positive_fraction"], .5)
        self.assertEqual(result["n_scored_dates"], 4)

    def test_zero_std_sharpe_is_undefined(self):
        self.assertTrue(np.isnan(summarize_scores(pd.Series([.1, .1]))["sharpe"]))

    def test_rank_ensemble_is_within_date_and_keeps_ties(self):
        frame = pd.DataFrame({"date": ["a", "a", "a", "b", "b"],
                              "p": [0, 0, 2, 100, 200], "q": [3, 1, 2, 200, 100]})
        actual = rank_ensemble(frame, ["p", "q"])
        np.testing.assert_allclose(actual, [.75, 5/12, 5/6, .75, .75])
        permuted = frame.sample(frac=1, random_state=8)
        np.testing.assert_allclose(actual, rank_ensemble(permuted, ["p", "q"]).sort_index())


if __name__ == "__main__":
    unittest.main()
