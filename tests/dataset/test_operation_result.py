"""Tests for OperationResult.to_dict()."""

import unittest

from amzn_nova_forge.dataset.operations.base import (
    FilterOperationResult,
    OperationResult,
)


class TestOperationResultToDict(unittest.TestCase):
    def test_to_dict_includes_all_fields(self):
        result = OperationResult(status="SUCCEEDED")
        self.assertEqual(result.to_dict(), {"status": "SUCCEEDED"})

    def test_to_dict_excludes_none_values(self):
        result = FilterOperationResult(status="SKIPPED")
        d = result.to_dict()
        # filtered_count and total_count are 0 (not None), so they should be present
        self.assertIn("filtered_count", d)
        self.assertIn("total_count", d)
        self.assertIn("filters_applied", d)

    def test_to_dict_with_populated_filter_result(self):
        result = FilterOperationResult(
            status="SUCCEEDED",
            filtered_count=5,
            total_count=100,
            filters_applied=["converse_format_reserved_keywords"],
        )
        self.assertEqual(
            result.to_dict(),
            {
                "status": "SUCCEEDED",
                "filtered_count": 5,
                "total_count": 100,
                "filters_applied": ["converse_format_reserved_keywords"],
            },
        )


if __name__ == "__main__":
    unittest.main()
