from __future__ import annotations

import unittest

from src.data.splits import expanding_window_splits, walkforward_expanding_splits


class SplitUtilityTest(unittest.TestCase):
    def test_expanding_split_single_holdout(self) -> None:
        split = expanding_window_splits(
            n_samples=100,
            train_size=0.6,
            val_size=0.2,
            test_size=0.2,
            n_splits=1,
        )[0]
        self.assertEqual(len(split.train_idx), 60)
        self.assertEqual(len(split.val_idx), 20)
        self.assertEqual(len(split.test_idx), 20)

    def test_walkforward_expanding_splits_are_non_overlapping_on_test(self) -> None:
        splits = walkforward_expanding_splits(
            n_samples=300,
            initial_train_size=120,
            val_size=40,
            test_size=40,
            step_size=40,
        )
        self.assertEqual(len(splits), 3)
        self.assertEqual(splits[0].test_idx[0], 160)
        self.assertEqual(splits[0].test_idx[-1], 199)
        self.assertEqual(splits[1].test_idx[0], 200)
        self.assertEqual(splits[1].test_idx[-1], 239)
        self.assertEqual(splits[2].test_idx[0], 240)
        self.assertEqual(splits[2].test_idx[-1], 279)


if __name__ == "__main__":
    unittest.main()
