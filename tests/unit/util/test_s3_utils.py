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
"""Unit tests for amzn_nova_forge.util.s3_utils."""

import unittest
from unittest.mock import MagicMock, patch

from botocore.exceptions import ClientError

from amzn_nova_forge.util.s3_utils import (
    ensure_bucket_exists,
    get_dataprep_bucket_name,
)


class TestGetDataprepBucketName(unittest.TestCase):
    def test_explicit_args(self):
        name = get_dataprep_bucket_name(account_id="123456789012", region="us-west-2")
        self.assertEqual(name, "sagemaker-forge-dataprep-123456789012-us-west-2")

    @patch("amzn_nova_forge.util.s3_utils.boto3")
    def test_resolves_from_session(self, mock_boto3):
        mock_session = MagicMock()
        mock_session.region_name = "eu-west-1"
        mock_boto3.session.Session.return_value = mock_session
        mock_sts = MagicMock()
        mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
        mock_boto3.client.return_value = mock_sts

        name = get_dataprep_bucket_name()
        self.assertEqual(name, "sagemaker-forge-dataprep-123456789012-eu-west-1")


class TestEnsureBucketExists(unittest.TestCase):
    @patch("amzn_nova_forge.util.s3_utils.boto3")
    def test_bucket_already_exists(self, mock_boto3):
        mock_s3 = MagicMock()
        mock_boto3.client.return_value = mock_s3
        ensure_bucket_exists("my-bucket", region="us-east-1")
        mock_s3.head_bucket.assert_called_once_with(Bucket="my-bucket")
        mock_s3.create_bucket.assert_not_called()

    @patch("amzn_nova_forge.util.s3_utils.boto3")
    def test_bucket_created_when_missing(self, mock_boto3):
        mock_s3 = MagicMock()
        mock_boto3.client.return_value = mock_s3
        mock_s3.head_bucket.side_effect = ClientError({"Error": {"Code": "404"}}, "HeadBucket")
        ensure_bucket_exists("new-bucket", region="us-east-1")
        mock_s3.create_bucket.assert_called_once()

    @patch("amzn_nova_forge.util.s3_utils.boto3")
    def test_bucket_created_with_location_constraint(self, mock_boto3):
        mock_s3 = MagicMock()
        mock_boto3.client.return_value = mock_s3
        mock_s3.head_bucket.side_effect = ClientError({"Error": {"Code": "404"}}, "HeadBucket")
        ensure_bucket_exists("new-bucket", region="us-west-2")
        mock_s3.create_bucket.assert_called_once_with(
            Bucket="new-bucket",
            CreateBucketConfiguration={"LocationConstraint": "us-west-2"},
        )

    @patch("amzn_nova_forge.util.s3_utils.boto3")
    def test_bucket_created_with_kms_encryption(self, mock_boto3):
        mock_s3 = MagicMock()
        mock_boto3.client.return_value = mock_s3
        mock_s3.head_bucket.side_effect = ClientError({"Error": {"Code": "404"}}, "HeadBucket")
        ensure_bucket_exists(
            "new-bucket",
            region="us-east-1",
            kms_key_arn="arn:aws:kms:us-east-1:123:key/abc",
        )
        mock_s3.create_bucket.assert_called_once()
        mock_s3.put_bucket_encryption.assert_called_once()
        enc_config = mock_s3.put_bucket_encryption.call_args.kwargs[
            "ServerSideEncryptionConfiguration"
        ]
        rule = enc_config["Rules"][0]["ApplyServerSideEncryptionByDefault"]
        self.assertEqual(rule["SSEAlgorithm"], "aws:kms")
        self.assertEqual(rule["KMSMasterKeyID"], "arn:aws:kms:us-east-1:123:key/abc")

    @patch("amzn_nova_forge.util.s3_utils.boto3")
    def test_permission_error_on_forbidden(self, mock_boto3):
        mock_s3 = MagicMock()
        mock_boto3.client.return_value = mock_s3
        mock_s3.head_bucket.side_effect = ClientError({"Error": {"Code": "403"}}, "HeadBucket")
        with self.assertRaises(PermissionError):
            ensure_bucket_exists("someone-elses-bucket", region="us-east-1")


if __name__ == "__main__":
    unittest.main()
