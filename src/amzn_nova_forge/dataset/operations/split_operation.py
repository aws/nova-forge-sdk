# Copyright 2025 Amazon Inc

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Operation for splitting datasets into train/validation/test sets."""

import random
from typing import Any, Dict, Iterator, List, Optional, Tuple

from ...util.logging import logger
from .base import DataPrepError, NovaForgeSplitOperation


class SplitOperation(NovaForgeSplitOperation):
    """Split the dataset into train, validation, and test DatasetLoader objects."""

    def execute(self, loader: Any, **kwargs) -> Tuple[Any, Any, Any]:
        """
        Split the current dataset into train/val/test loaders.

        Args:
            loader: The DatasetLoader instance.
            train_ratio: Fraction for training.
            val_ratio: Fraction for validation.
            test_ratio: Fraction for testing.
            seed: Random seed.

        Returns:
            Tuple of (train_loader, val_loader, test_loader).
        """
        train_ratio, val_ratio, test_ratio = self._resolve_ratios(
            kwargs.get("train_ratio"),
            kwargs.get("val_ratio"),
            kwargs.get("test_ratio"),
        )
        seed: int = kwargs.get("seed", 42)

        dataset = self._materialize_dataset(loader)
        indices = self._shuffled_indices(len(dataset), seed)
        n_train, n_val, n_test = self._compute_split_sizes(
            len(dataset), train_ratio, val_ratio, test_ratio
        )

        train_indices = indices[:n_train]
        val_indices = indices[n_train : n_train + n_val]
        test_indices = indices[n_train + n_val :]

        train_loader = self._create_split_loader(loader, dataset, train_indices)
        val_loader = self._create_split_loader(loader, dataset, val_indices)
        test_loader = self._create_split_loader(loader, dataset, test_indices)

        logger.info(f"Data split: {n_train} train, {n_val} val, {n_test} test")
        return train_loader, val_loader, test_loader

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_ratios(
        train_ratio: Optional[float],
        val_ratio: Optional[float],
        test_ratio: Optional[float],
    ) -> Tuple[float, float, float]:
        """Return validated (train, val, test) ratios, applying defaults when all are None."""
        if (train_ratio, val_ratio, test_ratio) == (None, None, None):
            return 0.8, 0.1, 0.1

        if train_ratio is None or val_ratio is None or test_ratio is None:
            raise DataPrepError(
                f"Please provide three values for split_data: train_ratio, val_ratio, and test_ratio."
                f"You provided: (Train: {train_ratio}, Val: {val_ratio}, Test: {test_ratio})"
            )

        if any(r < 0 for r in [train_ratio, val_ratio, test_ratio]):
            raise DataPrepError(
                "Calculated ratio is negative. Provided ratios sum to > 1.0"
            )

        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise DataPrepError(
                f"Split ratios must sum to 1.0. Current Ratios: "
                f"(Train: {train_ratio}, Val: {val_ratio}, Test: {test_ratio} "
                f"-> Total: {abs(train_ratio + val_ratio + test_ratio)})"
            )

        return train_ratio, val_ratio, test_ratio

    @staticmethod
    def _materialize_dataset(loader: Any) -> List[Dict]:
        """Load the full dataset into memory, raising on empty."""
        dataset = list(loader.dataset())
        if not dataset:
            raise DataPrepError("Dataset is empty. Call load() method first")

        if len(dataset) < 10:
            logger.info(
                "The provided dataset is small. Data will be split, "
                "but consider adding more data for better results."
            )
        return dataset

    @staticmethod
    def _shuffled_indices(n: int, seed: int) -> List[int]:
        """Return a shuffled list of indices [0, n)."""
        random.seed(seed)
        indices = list(range(n))
        random.shuffle(indices)
        return indices

    @staticmethod
    def _compute_split_sizes(
        n_total: int,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
    ) -> Tuple[int, int, int]:
        """Convert ratios to concrete counts that sum to n_total."""
        n_train = max(1, round(n_total * train_ratio)) if train_ratio > 0 else 0
        remaining = n_total - n_train

        if val_ratio == 0:
            n_val = 0
            n_test = remaining
        elif test_ratio == 0:
            n_val = remaining
            n_test = 0
        else:
            n_val = max(1, round(n_total * val_ratio)) if val_ratio > 0 else 0
            n_test = remaining - n_val

        # Ensure we haven't exceeded the total length
        if n_train + n_val + n_test > n_total:
            if n_test > 0:
                n_test -= 1
            elif n_val > 0:
                n_val -= 1
            else:
                n_train -= 1

        return n_train, n_val, n_test

    @staticmethod
    def _create_split_loader(
        loader: Any, dataset: List[Dict], split_indices: List[int]
    ) -> Any:
        """Create a new loader of the same type backed by the given indices."""

        def generator() -> Iterator[Dict]:
            for idx in split_indices:
                yield dataset[idx]

        new_loader = loader.__class__(**loader.column_mappings)
        new_loader.dataset = generator
        return new_loader
