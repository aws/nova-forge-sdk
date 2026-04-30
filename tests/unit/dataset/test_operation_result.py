# Copyright Amazon.com, Inc. or its affiliates

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
"""Tests for OperationResult.to_dict()."""

import unittest

from amzn_nova_forge.dataset.operations.base import (
    FilterOperationResult,
    OperationResult,
)


class TestOperationResultToDict(unittest.TestCase):
    def test_to_dict_includes_all_fields(self):
        result = OperationResult(status="SUCCEEDED", output_state=None)
        self.assertEqual(result.to_dict(), {"status": "SUCCEEDED"})

    def test_to_dict_excludes_none_values(self):
        result = FilterOperationResult(
            status="SKIPPED", output_state=None, filtered_count=0, total_count=0
        )
        d = result.to_dict()
        self.assertIn("filtered_count", d)
        self.assertIn("total_count", d)
        self.assertIn("filters_applied", d)

    def test_to_dict_with_populated_filter_result(self):
        result = FilterOperationResult(
            status="SUCCEEDED",
            output_state=None,
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
