"""Checks for corrupted keys and the label-overlap boundary."""
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_data import normalize_keys, prepare


class CleaningTests(unittest.TestCase):
    def test_rejects_nat_string(self):
        with self.assertRaises(ValueError):
            normalize_keys(pd.DataFrame({"numerai_ticker": ["ABC US"], "date": ["NaT"]}))

    def test_rejects_empty_ticker(self):
        with self.assertRaises(ValueError):
            normalize_keys(pd.DataFrame({"numerai_ticker": ["   "], "date": ["2020-01-01"]}))

    def test_rejects_insufficient_embargo_before_io(self):
        with self.assertRaises(ValueError):
            prepare(embargo_bdays=59)

    def test_weekday_embargo_across_weekend(self):
        train_end = pd.Timestamp("2018-12-28")
        cutoff = train_end + pd.offsets.BDay(65)
        self.assertEqual(cutoff, pd.Timestamp("2019-03-29"))
        # Boundary is strictly greater, never the date on which the interval ends.
        dates = pd.Series(pd.to_datetime(["2019-03-29", "2019-04-05"]))
        self.assertEqual((dates > cutoff).tolist(), [False, True])


if __name__ == "__main__":
    unittest.main()
