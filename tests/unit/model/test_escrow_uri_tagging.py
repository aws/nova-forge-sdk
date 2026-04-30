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
"""Tests for escrow URI tagging feature."""

import unittest
from unittest.mock import MagicMock, Mock, patch

from botocore.exceptions import ClientError

from amzn_nova_forge.model.model_config import (
    ESCROW_URI_TAG_KEY,
    ModelDeployResult,
    _escrow_tag_value,
)
from amzn_nova_forge.util.bedrock import find_bedrock_model_by_tag
from amzn_nova_forge.util.sagemaker import find_sagemaker_model_by_tag


class TestEscrowTagValue(unittest.TestCase):
    """Tests for _escrow_tag_value helper."""

    def test_short_uri_unchanged(self):
        uri = "s3://bucket/path/to/model/"
        self.assertEqual(_escrow_tag_value(uri), uri)

    def test_256_char_uri_unchanged(self):
        uri = "s3://" + "a" * 251
        self.assertEqual(len(uri), 256)
        self.assertEqual(_escrow_tag_value(uri), uri)

    def test_long_uri_truncated_with_hash(self):
        uri = "s3://" + "a" * 300
        result = _escrow_tag_value(uri)
        self.assertLessEqual(len(result), 256)
        self.assertIn("-", result[224:])  # dash separator

    def test_different_long_uris_produce_different_values(self):
        uri1 = "s3://bucket-a/" + "x" * 300
        uri2 = "s3://bucket-b/" + "x" * 300
        self.assertNotEqual(_escrow_tag_value(uri1), _escrow_tag_value(uri2))


class TestFindSageMakerModelByTag(unittest.TestCase):
    """Tests for find_sagemaker_model_by_tag using ResourceGroupsTaggingAPI."""

    @patch("amzn_nova_forge.util.sagemaker.boto3.client")
    def test_found(self, mock_boto_client):
        mock_tagging = MagicMock()
        mock_boto_client.return_value = mock_tagging
        mock_tagging.get_resources.return_value = {
            "ResourceTagMappingList": [
                {"ResourceARN": "arn:aws:sagemaker:us-east-1:123456789012:model/my-model"}
            ]
        }
        mock_sm_client = MagicMock()
        mock_sm_client.meta.region_name = "us-east-1"

        result = find_sagemaker_model_by_tag("s3://bucket/path/", mock_sm_client)
        self.assertEqual(result, "arn:aws:sagemaker:us-east-1:123456789012:model/my-model")
        mock_tagging.get_resources.assert_called_once()
        call_kwargs = mock_tagging.get_resources.call_args[1]
        self.assertEqual(call_kwargs["ResourceTypeFilters"], ["sagemaker:model"])

    @patch("amzn_nova_forge.util.sagemaker.boto3.client")
    def test_not_found(self, mock_boto_client):
        mock_tagging = MagicMock()
        mock_boto_client.return_value = mock_tagging
        mock_tagging.get_resources.return_value = {"ResourceTagMappingList": []}
        mock_sm_client = MagicMock()
        mock_sm_client.meta.region_name = "us-east-1"

        result = find_sagemaker_model_by_tag("s3://bucket/path/", mock_sm_client)
        self.assertIsNone(result)

    @patch("amzn_nova_forge.util.sagemaker.boto3.client")
    def test_permission_denied_returns_none(self, mock_boto_client):
        mock_tagging = MagicMock()
        mock_boto_client.return_value = mock_tagging
        mock_tagging.get_resources.side_effect = ClientError(
            {"Error": {"Code": "AccessDeniedException", "Message": "denied"}},
            "GetResources",
        )
        mock_sm_client = MagicMock()
        mock_sm_client.meta.region_name = "us-east-1"

        result = find_sagemaker_model_by_tag("s3://bucket/path/", mock_sm_client)
        self.assertIsNone(result)

    @patch("amzn_nova_forge.util.sagemaker.boto3.client")
    def test_unexpected_error_returns_none(self, mock_boto_client):
        mock_tagging = MagicMock()
        mock_boto_client.return_value = mock_tagging
        mock_tagging.get_resources.side_effect = RuntimeError("boom")
        mock_sm_client = MagicMock()
        mock_sm_client.meta.region_name = "us-east-1"

        result = find_sagemaker_model_by_tag("s3://bucket/path/", mock_sm_client)
        self.assertIsNone(result)


class TestFindBedrockModelByTag(unittest.TestCase):
    """Tests for find_bedrock_model_by_tag using ResourceGroupsTaggingAPI."""

    @patch("amzn_nova_forge.util.bedrock.boto3.client")
    def test_found(self, mock_boto_client):
        mock_tagging = MagicMock()
        mock_boto_client.return_value = mock_tagging
        mock_tagging.get_resources.return_value = {
            "ResourceTagMappingList": [
                {"ResourceARN": "arn:aws:bedrock:us-east-1:123456789012:custom-model/m1"}
            ]
        }
        mock_bedrock_client = MagicMock()
        mock_bedrock_client.meta.region_name = "us-east-1"

        result = find_bedrock_model_by_tag("s3://bucket/path/", mock_bedrock_client)
        self.assertEqual(result, "arn:aws:bedrock:us-east-1:123456789012:custom-model/m1")
        call_kwargs = mock_tagging.get_resources.call_args[1]
        self.assertEqual(call_kwargs["ResourceTypeFilters"], ["bedrock:custom-model"])
        self.assertEqual(call_kwargs["TagFilters"][0]["Key"], ESCROW_URI_TAG_KEY)

    @patch("amzn_nova_forge.util.bedrock.boto3.client")
    def test_not_found(self, mock_boto_client):
        mock_tagging = MagicMock()
        mock_boto_client.return_value = mock_tagging
        mock_tagging.get_resources.return_value = {"ResourceTagMappingList": []}
        mock_bedrock_client = MagicMock()
        mock_bedrock_client.meta.region_name = "us-east-1"

        result = find_bedrock_model_by_tag("s3://bucket/path/", mock_bedrock_client)
        self.assertIsNone(result)

    @patch("amzn_nova_forge.util.bedrock.boto3.client")
    def test_permission_denied_returns_none(self, mock_boto_client):
        mock_tagging = MagicMock()
        mock_boto_client.return_value = mock_tagging
        mock_tagging.get_resources.side_effect = ClientError(
            {"Error": {"Code": "AccessDeniedException", "Message": "denied"}},
            "GetResources",
        )
        mock_bedrock_client = MagicMock()
        mock_bedrock_client.meta.region_name = "us-east-1"

        result = find_bedrock_model_by_tag("s3://bucket/path/", mock_bedrock_client)
        self.assertIsNone(result)


class TestModelDeployResultFromArnWithTags(unittest.TestCase):
    """Tests for enriched from_arn() that reads escrow_uri from tags."""

    def test_from_arn_populates_escrow_uri_from_tag(self):
        mock_client = Mock()
        mock_client.get_custom_model.return_value = {
            "modelArn": "arn:aws:bedrock:us-east-1:123456789012:custom-model/my-model",
            "modelName": "my-model",
            "creationTime": "2026-04-01T12:00:00+00:00",
        }
        mock_client.list_tags_for_resource.return_value = {
            "tags": [{"key": ESCROW_URI_TAG_KEY, "value": "s3://escrow/path/"}]
        }
        result = ModelDeployResult.from_arn(
            "arn:aws:bedrock:us-east-1:123456789012:custom-model/my-model",
            bedrock_client=mock_client,
        )
        self.assertEqual(result.escrow_uri, "s3://escrow/path/")

    def test_from_arn_empty_escrow_uri_when_no_tag(self):
        mock_client = Mock()
        mock_client.get_custom_model.return_value = {
            "modelArn": "arn:aws:bedrock:us-east-1:123456789012:custom-model/test",
            "modelName": "test",
            "creationTime": "2026-04-01T12:00:00+00:00",
        }
        mock_client.list_tags_for_resource.return_value = {"tags": []}
        result = ModelDeployResult.from_arn(
            "arn:aws:bedrock:us-east-1:123456789012:custom-model/test",
            bedrock_client=mock_client,
        )
        self.assertEqual(result.escrow_uri, "")

    def test_from_arn_empty_escrow_uri_when_tag_read_fails(self):
        mock_client = Mock()
        mock_client.get_custom_model.return_value = {
            "modelArn": "arn:aws:bedrock:us-east-1:123456789012:custom-model/test",
            "modelName": "test",
            "creationTime": "2026-04-01T12:00:00+00:00",
        }
        mock_client.list_tags_for_resource.side_effect = Exception("no perms")
        result = ModelDeployResult.from_arn(
            "arn:aws:bedrock:us-east-1:123456789012:custom-model/test",
            bedrock_client=mock_client,
        )
        self.assertEqual(result.escrow_uri, "")


class TestEscrowTagKey(unittest.TestCase):
    """Tests for the tag key constant."""

    def test_tag_key_is_namespaced(self):
        self.assertTrue(ESCROW_URI_TAG_KEY.startswith("sagemaker.amazonaws.com/"))
        self.assertIn("escrow-uri", ESCROW_URI_TAG_KEY)


if __name__ == "__main__":
    unittest.main()
