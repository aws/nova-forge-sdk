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
import json
import unittest
from unittest.mock import MagicMock, patch

from botocore.exceptions import ClientError

from amzn_nova_forge.util.hub_util import get_hub_content


class TestGetHubContent(unittest.TestCase):
    @patch("amzn_nova_forge.util.hub_util.boto3.client")
    def test_returns_response_with_parsed_json_document(self, mock_boto_client):
        mock_sagemaker = MagicMock()
        mock_boto_client.return_value = mock_sagemaker
        doc = {"key": "value", "nested": {"a": 1}}
        mock_sagemaker.describe_hub_content.return_value = {
            "HubContentName": "my-content",
            "HubContentDocument": json.dumps(doc),
        }

        result = get_hub_content("my-hub", "my-content", "Model", "us-west-2")

        mock_boto_client.assert_called_once_with("sagemaker", region_name="us-west-2")
        mock_sagemaker.describe_hub_content.assert_called_once_with(
            HubName="my-hub",
            HubContentType="Model",
            HubContentName="my-content",
        )
        self.assertEqual(result["HubContentDocument"], doc)
        self.assertEqual(result["HubContentName"], "my-content")

    @patch("amzn_nova_forge.util.hub_util.boto3.client")
    def test_leaves_non_string_document_unchanged(self, mock_boto_client):
        mock_sagemaker = MagicMock()
        mock_boto_client.return_value = mock_sagemaker
        doc = {"already": "parsed"}
        mock_sagemaker.describe_hub_content.return_value = {
            "HubContentDocument": doc,
        }

        result = get_hub_content("hub", "content", "Model", "us-east-1")

        self.assertEqual(result["HubContentDocument"], doc)

    @patch("amzn_nova_forge.util.hub_util.boto3.client")
    def test_leaves_invalid_json_string_as_is(self, mock_boto_client):
        mock_sagemaker = MagicMock()
        mock_boto_client.return_value = mock_sagemaker
        invalid_json = "not valid json {{"
        mock_sagemaker.describe_hub_content.return_value = {
            "HubContentDocument": invalid_json,
        }

        result = get_hub_content("hub", "content", "Model", "us-east-1")

        self.assertEqual(result["HubContentDocument"], invalid_json)

    @patch("amzn_nova_forge.util.hub_util.boto3.client")
    def test_response_without_hub_content_document(self, mock_boto_client):
        mock_sagemaker = MagicMock()
        mock_boto_client.return_value = mock_sagemaker
        mock_sagemaker.describe_hub_content.return_value = {
            "HubContentName": "my-content",
        }

        result = get_hub_content("hub", "content", "Model", "us-east-1")

        self.assertNotIn("HubContentDocument", result)
        self.assertEqual(result["HubContentName"], "my-content")

    @patch("amzn_nova_forge.util.hub_util.boto3.client")
    def test_raises_runtime_error_on_api_failure(self, mock_boto_client):
        mock_sagemaker = MagicMock()
        mock_boto_client.return_value = mock_sagemaker
        mock_sagemaker.describe_hub_content.side_effect = Exception("Access denied")

        with self.assertRaises(RuntimeError) as ctx:
            get_hub_content("hub", "my-content", "Model", "us-west-2")

        self.assertIn("my-content", str(ctx.exception))
        self.assertIn("Access denied", str(ctx.exception))

    @patch("amzn_nova_forge.util.hub_util.boto3.client")
    def test_raises_runtime_error_on_client_error(self, mock_boto_client):
        mock_sagemaker = MagicMock()
        mock_boto_client.return_value = mock_sagemaker
        mock_sagemaker.describe_hub_content.side_effect = ClientError(
            error_response={
                "Error": {
                    "Code": "ResourceNotFound",
                    "Message": "Hub content not found",
                }
            },
            operation_name="DescribeHubContent",
        )

        with self.assertRaises(RuntimeError) as ctx:
            get_hub_content("hub", "my-content", "Model", "us-east-1")

        self.assertIn("Failed to get SageMaker hub content", str(ctx.exception))
        self.assertIn("my-content", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
