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
"""Unit tests for amzn_nova_forge.util.dataset_writer."""

import unittest
from unittest.mock import MagicMock, patch

from amzn_nova_forge.util.dataset_writer import DatasetWriter


class TestDatasetWriterSaveToS3(unittest.TestCase):
    @patch("amzn_nova_forge.util.dataset_writer.boto3")
    def test_save_to_s3_propagates_region(self, mock_boto3):
        mock_s3 = MagicMock()
        mock_boto3.client.return_value = mock_s3

        DatasetWriter.save_to_s3(
            "s3://bucket/key.jsonl", iter([{"a": 1}]), is_jsonl=True, region="eu-west-1"
        )

        mock_boto3.client.assert_called_once_with("s3", region_name="eu-west-1")
