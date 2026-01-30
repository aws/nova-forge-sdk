"""
Tests to verify that dataset loaders actually load data lazily without
pulling the entire file into memory at once.
"""

import unittest
from io import BytesIO
from unittest.mock import MagicMock, mock_open, patch

from amzn_nova_customization_sdk.dataset.dataset_loader import (
    CSVDatasetLoader,
    JSONLDatasetLoader,
)


class TestLazyLoading(unittest.TestCase):
    """Test that loaders actually stream data lazily."""

    def test_jsonl_loader_streams_lazily(self):
        """Verify JSONL loader doesn't load entire file into memory."""
        # Create a mock that tracks when lines are accessed
        lines_accessed = []

        def mock_line_generator():
            """Generator that tracks which lines are accessed."""
            for i in range(5):
                lines_accessed.append(i)
                yield f'{{"id": {i}, "value": "line_{i}"}}'

        with patch(
            "amzn_nova_customization_sdk.dataset.dataset_loader.load_file_content",
            side_effect=lambda *args, **kwargs: mock_line_generator(),
        ):
            loader = JSONLDatasetLoader()
            loader.load("test.jsonl")

            # At this point, no lines should have been accessed yet
            self.assertEqual(
                len(lines_accessed), 0, "Lines accessed before iteration started"
            )

            # Get the generator
            dataset_iter = loader.raw_dataset()

            # Still no lines accessed
            self.assertEqual(
                len(lines_accessed), 0, "Lines accessed when getting iterator"
            )

            # Access first item
            first_item = next(dataset_iter)
            self.assertEqual(first_item, {"id": 0, "value": "line_0"})
            self.assertEqual(len(lines_accessed), 1, "Should only access first line")

            # Access second item
            second_item = next(dataset_iter)
            self.assertEqual(second_item, {"id": 1, "value": "line_1"})
            self.assertEqual(
                len(lines_accessed), 2, "Should only access first two lines"
            )

            # Verify we haven't accessed all lines yet
            self.assertLess(
                len(lines_accessed), 5, "Should not have accessed all lines yet"
            )

    def test_csv_loader_streams_lazily(self):
        """Verify CSV loader doesn't load entire file into memory (except header)."""
        lines_accessed = []

        def mock_line_generator():
            """Generator that tracks which lines are accessed."""
            # CSV needs header first
            lines_accessed.append("header")
            yield "id,name,value"

            for i in range(5):
                lines_accessed.append(i)
                yield f"{i},name_{i},value_{i}"

        with patch(
            "amzn_nova_customization_sdk.dataset.dataset_loader.load_file_content",
            side_effect=lambda *args, **kwargs: mock_line_generator(),
        ):
            loader = CSVDatasetLoader()
            loader.load("test.csv")

            # Get the generator
            dataset_iter = loader.raw_dataset()

            # CSV reader needs to read header, so that's expected
            # But data rows should not be accessed yet

            # Access first data row
            first_item = next(dataset_iter)
            self.assertEqual(first_item["id"], "0")
            self.assertEqual(first_item["name"], "name_0")

            # Should have accessed header + first row only
            self.assertIn("header", lines_accessed)
            self.assertIn(0, lines_accessed)
            self.assertEqual(
                len(lines_accessed), 2, "Should only access header and first row"
            )

            # Access second row
            second_item = next(dataset_iter)
            self.assertEqual(second_item["id"], "1")
            self.assertEqual(
                len(lines_accessed), 3, "Should only access header and first two rows"
            )

    def test_jsonl_loader_with_transform_streams_lazily(self):
        """Verify that transformation also happens lazily after initial validation."""
        lines_accessed = []

        def mock_line_generator():
            for i in range(5):
                lines_accessed.append(i)
                # Use a format that will require transformation
                yield f'{{"my_text": "line_{i}"}}'

        with patch(
            "amzn_nova_customization_sdk.dataset.dataset_loader.load_file_content",
            side_effect=lambda *args, **kwargs: mock_line_generator(),
        ):
            loader = JSONLDatasetLoader(text="my_text")
            loader.load("test.jsonl")

            from amzn_nova_customization_sdk.model.model_enums import (
                Model,
                TrainingMethod,
            )

            # Reset counters before transform
            lines_accessed.clear()

            loader.transform(TrainingMethod.CPT, Model.NOVA_MICRO)

            # Transform will validate the schema which reads all records once
            # This is expected - we need to check if transformation is needed
            # But the important part is that transformed_dataset is still lazy

            # Reset counters to test lazy iteration of transformed data
            lines_accessed.clear()

            # Get transformed iterator
            transformed_iter = loader.transformed_dataset()

            # Nothing should be accessed yet
            self.assertEqual(
                len(lines_accessed),
                0,
                "No lines should be accessed when getting iterator",
            )

            # Access first item
            first_item = next(transformed_iter)

            # Now we should have accessed exactly one item
            self.assertEqual(
                len(lines_accessed), 1, "Should only access one line for first item"
            )
            self.assertIn("text", first_item)
            self.assertEqual(first_item["text"], "line_0")

            # Access second item
            second_item = next(transformed_iter)

            # Should have accessed exactly two items
            self.assertEqual(
                len(lines_accessed), 2, "Should only access two lines for two items"
            )
            self.assertEqual(second_item["text"], "line_1")

    def test_s3_streaming_uses_iter_lines(self):
        """Verify S3 downloads use streaming iter_lines, not read()."""
        with patch("amzn_nova_customization_sdk.util.recipe.boto3.client") as mock_boto:
            mock_s3 = MagicMock()
            mock_boto.return_value = mock_s3

            # Mock the response body with iter_lines
            mock_body = MagicMock()
            lines = [b'{"id": 1}', b'{"id": 2}', b'{"id": 3}']
            mock_body.iter_lines.return_value = iter(lines)

            mock_s3.get_object.return_value = {"Body": mock_body}

            with patch(
                "amzn_nova_customization_sdk.util.recipe._parse_s3_uri",
                return_value=("bucket", "key.jsonl"),
            ):
                loader = JSONLDatasetLoader()
                loader.load("s3://bucket/key.jsonl")

                # Consume first item
                dataset_iter = loader.raw_dataset()
                first = next(dataset_iter)

                # Verify iter_lines was called (streaming), not read() (full load)
                mock_body.iter_lines.assert_called_once()
                self.assertFalse(
                    hasattr(mock_body, "read") and mock_body.read.called,
                    "Should use iter_lines, not read()",
                )

                self.assertEqual(first, {"id": 1})
