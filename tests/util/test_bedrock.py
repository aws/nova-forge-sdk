import json
import unittest
from unittest.mock import MagicMock, patch

from amzn_nova_customization_sdk.util.bedrock import create_bedrock_execution_role


class TestBedrock(unittest.TestCase):
    @patch("boto3.client")
    def test_create_bedrock_execution_role_with_wildcards(self, mock_boto_client):
        mock_sts = MagicMock()
        mock_sts.get_caller_identity.return_value = {"Account": "123456789"}
        mock_boto_client.return_value = mock_sts

        role_name = "role_name"
        mock_iam_client = MagicMock()
        mock_iam_client.exceptions.NoSuchEntityException = type(
            "NoSuchEntityException", (Exception,), {}
        )
        mock_iam_client.exceptions.EntityAlreadyExistsException = type(
            "EntityAlreadyExistsException", (Exception,), {}
        )
        mock_iam_client.get_role.side_effect = (
            mock_iam_client.exceptions.NoSuchEntityException("Role not found")
        )
        mock_iam_client.create_policy.return_value = {
            "Policy": {"Arn": "arn:aws:iam::123456789:policy/foo"}
        }

        create_bedrock_execution_role(mock_iam_client, role_name)

        mock_iam_client.create_policy.assert_any_call(
            PolicyName=f"{role_name}Bedrock_Policy",
            PolicyDocument=json.dumps(
                {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Action": [
                                "bedrock:CreateCustomModelDeployment",
                                "bedrock:CreateCustomModel",
                                "bedrock:CreateProvisionedModelThroughput",
                                "bedrock:GetCustomModel",
                                "bedrock:GetCustomModelDeployment",
                            ],
                            "Resource": "*",
                        }
                    ],
                }
            ),
        )

        mock_iam_client.create_policy.assert_any_call(
            PolicyName=f"{role_name}S3_Read_Policy",
            PolicyDocument=json.dumps(
                {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Action": ["s3:GetObject", "s3:ListBucket"],
                            "Resource": "*",
                        }
                    ],
                }
            ),
        )

    @patch("boto3.client")
    def test_create_bedrock_execution_role_with_scoped_resources(
        self, mock_boto_client
    ):
        account_id = "123456789"
        mock_sts = MagicMock()
        mock_sts.get_caller_identity.return_value = {"Account": account_id}
        mock_boto_client.return_value = mock_sts

        role_name = "role_name"
        scoped_resource = "resource"
        mock_iam_client = MagicMock()
        mock_iam_client.exceptions.NoSuchEntityException = type(
            "NoSuchEntityException", (Exception,), {}
        )
        mock_iam_client.exceptions.EntityAlreadyExistsException = type(
            "EntityAlreadyExistsException", (Exception,), {}
        )
        mock_iam_client.get_role.side_effect = (
            mock_iam_client.exceptions.NoSuchEntityException("Role not found")
        )
        mock_iam_client.create_policy.return_value = {
            "Policy": {"Arn": f"arn:aws:iam::{account_id}:policy/foo"}
        }

        create_bedrock_execution_role(
            mock_iam_client, role_name, scoped_resource, scoped_resource
        )

        mock_iam_client.create_policy.assert_any_call(
            PolicyName=f"{role_name}Bedrock_Policy",
            PolicyDocument=json.dumps(
                {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Action": [
                                "bedrock:CreateCustomModelDeployment",
                                "bedrock:CreateCustomModel",
                                "bedrock:CreateProvisionedModelThroughput",
                                "bedrock:GetCustomModel",
                                "bedrock:GetCustomModelDeployment",
                            ],
                            "Resource": f"arn:aws:bedrock:*:*:custom-model/{scoped_resource}*",
                        }
                    ],
                }
            ),
        )

        mock_iam_client.create_policy.assert_any_call(
            PolicyName=f"{role_name}S3_Read_Policy",
            PolicyDocument=json.dumps(
                {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Action": ["s3:GetObject", "s3:ListBucket"],
                            "Resource": [
                                f"arn:aws:s3:::{scoped_resource}*",
                                f"arn:aws:s3:::{scoped_resource}*/*",
                                f"arn:aws:s3:::customer-escrow-{account_id}*",
                                f"arn:aws:s3:::customer-escrow-{account_id}*/*",
                            ],
                        }
                    ],
                }
            ),
        )


if __name__ == "__main__":
    unittest.main()
