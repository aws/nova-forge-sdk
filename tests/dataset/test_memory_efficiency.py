"""
Memory efficiency tests to verify that lazy loading keeps memory usage bounded
regardless of dataset size.

These tests ensure that the lazy loading implementation doesn't load entire
datasets into memory, which is critical for processing large datasets.
"""

import json
import tracemalloc
import unittest
from io import StringIO
from typing import Dict, Iterator
from unittest.mock import patch

import pytest

from amzn_nova_customization_sdk.dataset.dataset_loader import (
    CSVDatasetLoader,
    JSONLDatasetLoader,
)
from amzn_nova_customization_sdk.model.model_enums import Model, TrainingMethod


class TestMemoryEfficiency(unittest.TestCase):
    """Test that dataset loaders maintain bounded memory usage with large datasets."""

    def _generate_jsonl_lines(self, size: int) -> Iterator[str]:
        """Generate JSONL lines for testing without storing in memory."""
        for i in range(size):
            yield json.dumps(
                {
                    "id": i,
                    "text": f"This is sample text number {i} "
                    * 10,  # ~300 bytes per record
                    "metadata": {"index": i, "category": f"cat_{i % 10}"},
                }
            )

    def _generate_csv_lines(self, size: int) -> Iterator[str]:
        """Generate CSV lines for testing without storing in memory."""
        yield "id,text,category"  # Header
        for i in range(size):
            text = f"Sample text {i} " * 10  # ~150 bytes per record
            yield f'{i},"{text}",cat_{i % 10}'

    def _measure_peak_memory_mb(self, func) -> float:
        """
        Measure peak memory usage of a function in MB.

        Args:
            func: Callable to measure

        Returns:
            Peak memory usage in MB
        """
        tracemalloc.start()
        try:
            func()
            current, peak = tracemalloc.get_traced_memory()
            return peak / 1024 / 1024  # Convert to MB
        finally:
            tracemalloc.stop()

    @pytest.mark.memory
    def test_jsonl_memory_scales_with_batch_not_dataset(self):
        """
        Verify JSONL loader memory usage is bounded by batch size, not total dataset size.

        Memory should remain relatively constant when processing 100 vs 10,000 records
        since we're streaming one record at a time.
        """

        def process_dataset(size: int) -> float:
            """Process dataset and return peak memory in MB."""

            def mock_generator():
                return self._generate_jsonl_lines(size)

            with patch(
                "amzn_nova_customization_sdk.dataset.dataset_loader.load_file_content",
                side_effect=lambda *args, **kwargs: mock_generator(),
            ):
                loader = JSONLDatasetLoader()
                loader.load("test.jsonl")

                # Process all records
                count = 0
                for record in loader.raw_dataset():
                    count += 1
                    # Simulate some processing
                    _ = record.get("text", "")

                return count

        # Measure memory for different dataset sizes
        small_memory = self._measure_peak_memory_mb(lambda: process_dataset(100))
        medium_memory = self._measure_peak_memory_mb(lambda: process_dataset(1000))
        large_memory = self._measure_peak_memory_mb(lambda: process_dataset(10000))

        # Calculate growth factors
        medium_growth = medium_memory / small_memory if small_memory > 0 else 0
        large_growth = large_memory / small_memory if small_memory > 0 else 0

        # Memory should not grow linearly with dataset size
        # Allow up to 3x growth for 100x data increase (accounts for overhead)
        self.assertLess(
            large_growth,
            3.0,
            f"Memory grew {large_growth:.2f}x for 100x data increase. "
            f"Small: {small_memory:.2f}MB, Large: {large_memory:.2f}MB. "
            f"This suggests data is being loaded into memory instead of streamed.",
        )

        # Log the actual memory usage for debugging
        print(
            f"\nMemory usage: Small={small_memory:.2f}MB, "
            f"Medium={medium_memory:.2f}MB, Large={large_memory:.2f}MB"
        )
        print(
            f"Growth factors: 10x data={medium_growth:.2f}x, 100x data={large_growth:.2f}x"
        )

    @pytest.mark.memory
    def test_csv_memory_scales_with_batch_not_dataset(self):
        """
        Verify CSV loader memory usage is bounded by batch size, not total dataset size.

        CSV requires reading the header, but data rows should stream lazily.
        """

        def process_dataset(size: int) -> float:
            """Process dataset and return peak memory in MB."""

            def mock_generator():
                return self._generate_csv_lines(size)

            with patch(
                "amzn_nova_customization_sdk.dataset.dataset_loader.load_file_content",
                side_effect=lambda *args, **kwargs: mock_generator(),
            ):
                loader = CSVDatasetLoader()
                loader.load("test.csv")

                # Process all records
                count = 0
                for record in loader.raw_dataset():
                    count += 1
                    _ = record.get("text", "")

                return count

        small_memory = self._measure_peak_memory_mb(lambda: process_dataset(100))
        large_memory = self._measure_peak_memory_mb(lambda: process_dataset(10000))

        growth_factor = large_memory / small_memory if small_memory > 0 else 0

        self.assertLess(
            growth_factor,
            3.0,
            f"Memory grew {growth_factor:.2f}x for 100x data increase. "
            f"Small: {small_memory:.2f}MB, Large: {large_memory:.2f}MB",
        )

        print(
            f"\nCSV Memory usage: Small={small_memory:.2f}MB, Large={large_memory:.2f}MB"
        )
        print(f"Growth factor: {growth_factor:.2f}x")

    @pytest.mark.memory
    def test_transformation_memory_efficiency(self):
        """
        Verify that transformation pipeline maintains lazy evaluation.

        Transforming data should not load the entire dataset into memory.
        """

        def process_with_transform(size: int) -> int:
            """Process dataset with transformation and return count."""

            def mock_generator():
                for i in range(size):
                    yield json.dumps({"text": f"Sample text {i} " * 10})

            with patch(
                "amzn_nova_customization_sdk.dataset.dataset_loader.load_file_content",
                side_effect=lambda *args, **kwargs: mock_generator(),
            ):
                loader = JSONLDatasetLoader()
                loader.load("test.jsonl")
                loader.transform(TrainingMethod.CPT, Model.NOVA_MICRO)

                # Process transformed data
                count = 0
                for record in loader.transformed_dataset():
                    count += 1
                    _ = record.get("text", "")

                return count

        small_memory = self._measure_peak_memory_mb(lambda: process_with_transform(100))
        large_memory = self._measure_peak_memory_mb(
            lambda: process_with_transform(10000)
        )

        growth_factor = large_memory / small_memory if small_memory > 0 else 0

        self.assertLess(
            growth_factor,
            3.0,
            f"Transformation memory grew {growth_factor:.2f}x for 100x data increase. "
            f"Small: {small_memory:.2f}MB, Large: {large_memory:.2f}MB",
        )

        print(
            f"\nTransformation Memory: Small={small_memory:.2f}MB, Large={large_memory:.2f}MB"
        )

    @pytest.mark.memory
    def test_iteration_does_not_accumulate_memory(self):
        """
        Verify that iterating through a dataset doesn't accumulate memory.

        Each iteration should release the previous record's memory.
        """

        def iterate_dataset(size: int):
            """Iterate through dataset without storing records."""

            def mock_generator():
                return self._generate_jsonl_lines(size)

            with patch(
                "amzn_nova_customization_sdk.dataset.dataset_loader.load_file_content",
                side_effect=lambda *args, **kwargs: mock_generator(),
            ):
                loader = JSONLDatasetLoader()
                loader.load("test.jsonl")

                # Iterate without storing
                for record in loader.raw_dataset():
                    # Just access the data, don't store it
                    _ = record.get("text")

        # Start memory tracking
        tracemalloc.start()

        # Get baseline memory
        baseline_current, baseline_peak = tracemalloc.get_traced_memory()

        # Iterate through dataset
        iterate_dataset(5000)

        # Get memory after iteration
        final_current, final_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Current memory should be close to baseline (records released)
        # Peak will be higher, but current should return to baseline
        memory_retained_mb = (final_current - baseline_current) / 1024 / 1024

        # Allow up to 5MB of retained memory for overhead
        self.assertLess(
            memory_retained_mb,
            5.0,
            f"Retained {memory_retained_mb:.2f}MB after iteration. "
            f"Records may not be getting garbage collected properly.",
        )

        print(
            f"\nIteration memory: Baseline={baseline_current / 1024 / 1024:.2f}MB, "
            f"Final={final_current / 1024 / 1024:.2f}MB, "
            f"Retained={memory_retained_mb:.2f}MB"
        )

    @pytest.mark.memory
    def test_show_method_memory_bounded(self):
        """
        Verify that show() method only loads the requested number of records.

        show(n=10) should not load the entire dataset into memory.
        """

        def show_dataset(size: int, show_n: int = 10):
            """Call show() on a dataset."""

            def mock_generator():
                return self._generate_jsonl_lines(size)

            with patch(
                "amzn_nova_customization_sdk.dataset.dataset_loader.load_file_content",
                side_effect=lambda *args, **kwargs: mock_generator(),
            ):
                # Suppress logger output
                with patch("amzn_nova_customization_sdk.dataset.dataset_loader.logger"):
                    loader = JSONLDatasetLoader()
                    loader.load("test.jsonl")
                    loader.show(n=show_n)

        # Memory for show(10) should be similar regardless of dataset size
        small_memory = self._measure_peak_memory_mb(lambda: show_dataset(100, 10))
        large_memory = self._measure_peak_memory_mb(lambda: show_dataset(10000, 10))

        growth_factor = large_memory / small_memory if small_memory > 0 else 0

        # show() should only load n records, so memory should be nearly constant
        self.assertLess(
            growth_factor,
            2.0,
            f"show() memory grew {growth_factor:.2f}x when dataset grew 100x. "
            f"Small: {small_memory:.2f}MB, Large: {large_memory:.2f}MB. "
            f"show() may be loading the entire dataset.",
        )

        print(
            f"\nshow() Memory: Small dataset={small_memory:.2f}MB, "
            f"Large dataset={large_memory:.2f}MB"
        )

    @pytest.mark.memory
    def test_multiple_iterations_do_not_accumulate(self):
        """
        Verify that multiple iterations over the same dataset don't accumulate memory.

        Each call to raw_dataset() should create a fresh iterator.
        """

        def mock_generator():
            return self._generate_jsonl_lines(1000)

        with patch(
            "amzn_nova_customization_sdk.dataset.dataset_loader.load_file_content",
            side_effect=lambda *args, **kwargs: mock_generator(),
        ):
            loader = JSONLDatasetLoader()
            loader.load("test.jsonl")

            tracemalloc.start()
            baseline_current, _ = tracemalloc.get_traced_memory()

            # Iterate multiple times
            for iteration in range(3):
                for record in loader.raw_dataset():
                    _ = record.get("text")

            final_current, _ = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            memory_retained_mb = (final_current - baseline_current) / 1024 / 1024

            # Should not accumulate memory across iterations
            self.assertLess(
                memory_retained_mb,
                5.0,
                f"Retained {memory_retained_mb:.2f}MB after 3 iterations. "
                f"Multiple iterations may be accumulating memory.",
            )

            print(f"\nMultiple iterations retained: {memory_retained_mb:.2f}MB")
