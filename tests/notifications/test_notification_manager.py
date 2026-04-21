import time
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, mock_open, patch

from botocore.exceptions import ClientError

from amzn_nova_forge.core.enums import Platform
from amzn_nova_forge.notifications.notification_manager import (
    NotificationManagerInfraError,
)
from amzn_nova_forge.notifications.smhp_notification_manager import (
    SMHPNotificationManager,
)
from amzn_nova_forge.notifications.smtj_notification_manager import (
    SMTJNotificationManager,
)


class TestSMTJNotificationManager(unittest.TestCase):
    """Test suite for SMTJNotificationManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.region = "us-east-1"

        # Create manager with mocked AWS clients
        with patch("amzn_nova_forge.notifications.notification_manager.boto3"):
            self.manager = SMTJNotificationManager(region=self.region)
            self.manager.cfn = MagicMock()
            self.manager.dynamodb = MagicMock()
            self.manager.sns = MagicMock()

    def test_initialization(self):
        """Test SMTJNotificationManager initialization."""
        self.assertEqual(self.manager.platform, Platform.SMTJ)
        self.assertEqual(self.manager.region, "us-east-1")

    def test_get_stack_name(self):
        """Test get_stack_name returns correct name for SMTJ."""
        stack_name = self.manager.get_stack_name()
        self.assertEqual(stack_name, "NovaForgeSDK-SMTJ-JobNotifications")

    def test_get_template_path(self):
        """Test get_template_path returns correct path for SMTJ."""
        template_path = self.manager.get_template_path()
        self.assertTrue(str(template_path).endswith("smtj_notification_cf_stack.yaml"))

    def test_get_platform_name(self):
        """Test get_platform_name returns correct platform name."""
        platform_name = self.manager.get_platform_name()
        self.assertEqual(platform_name, "SMTJ")

    def test_get_stack_parameters_no_kms(self):
        """Test _get_stack_parameters returns empty list when no KMS key provided."""
        parameters = self.manager._get_stack_parameters()
        self.assertEqual(parameters, [])

    def test_get_stack_parameters_with_kms(self):
        """Test _get_stack_parameters returns KMS parameter when provided."""
        kms_key_id = "arn:aws:kms:us-east-1:123456789012:key/abc-123"
        parameters = self.manager._get_stack_parameters(kms_key_id=kms_key_id)

        self.assertEqual(len(parameters), 1)
        self.assertEqual(parameters[0]["ParameterKey"], "KmsKeyId")
        self.assertEqual(parameters[0]["ParameterValue"], kms_key_id)

    def test_validate_email_valid(self):
        """Test _validate_email with valid email addresses."""
        valid_emails = [
            "user@example.com",
            "test.user@example.co.uk",
            "user+tag@example.com",
            "user123@test-domain.com",
        ]
        for email in valid_emails:
            self.assertTrue(
                SMTJNotificationManager._validate_email(email),
                f"Email {email} should be valid",
            )

    def test_validate_email_invalid(self):
        """Test _validate_email with invalid email addresses."""
        invalid_emails = [
            "invalid",
            "@example.com",
            "user@",
            "user @example.com",
            "user@example",
        ]
        for email in invalid_emails:
            self.assertFalse(
                SMTJNotificationManager._validate_email(email),
                f"Email {email} should be invalid",
            )

    def test_parse_stack_outputs(self):
        """Test _parse_stack_outputs correctly parses CloudFormation outputs."""
        outputs = [
            {"OutputKey": "TableName", "OutputValue": "test-table"},
            {
                "OutputKey": "TopicArn",
                "OutputValue": "arn:aws:sns:us-east-1:123456789012:topic",
            },
        ]
        parsed = SMTJNotificationManager._parse_stack_outputs(outputs)
        self.assertEqual(parsed["TableName"], "test-table")
        self.assertEqual(parsed["TopicArn"], "arn:aws:sns:us-east-1:123456789012:topic")

    @patch("amzn_nova_forge.notifications.notification_manager.logger")
    def test_ensure_infrastructure_exists_already_created(self, mock_logger):
        """Test _ensure_infrastructure_exists when stack already exists."""
        # Mock stack already exists
        self.manager.cfn.describe_stacks.return_value = {
            "Stacks": [
                {
                    "StackStatus": "CREATE_COMPLETE",
                    "Outputs": [
                        {"OutputKey": "DynamoDBTableName", "OutputValue": "test-table"},
                        {
                            "OutputKey": "SNSTopicArn",
                            "OutputValue": "arn:aws:sns:us-east-1:123456789012:topic",
                        },
                    ],
                }
            ]
        }

        outputs = self.manager._ensure_infrastructure_exists()

        self.assertEqual(outputs["DynamoDBTableName"], "test-table")
        self.assertEqual(outputs["SNSTopicArn"], "arn:aws:sns:us-east-1:123456789012:topic")
        mock_logger.info.assert_called()

    @patch("amzn_nova_forge.notifications.notification_manager.logger")
    @patch("builtins.open", new_callable=mock_open, read_data="template: content")
    def test_ensure_infrastructure_exists_creates_stack(self, mock_file, mock_logger):
        """Test _ensure_infrastructure_exists creates stack when it doesn't exist."""

        # Mock stack doesn't exist initially, then exists after creation
        def describe_stacks_side_effect(*args, **kwargs):
            if self.manager.cfn.create_stack.called:
                # After creation, return the new stack
                return {
                    "Stacks": [
                        {
                            "StackStatus": "CREATE_COMPLETE",
                            "Outputs": [
                                {
                                    "OutputKey": "DynamoDBTableName",
                                    "OutputValue": "new-table",
                                },
                                {
                                    "OutputKey": "SNSTopicArn",
                                    "OutputValue": "arn:aws:sns:us-east-1:123456789012:new-topic",
                                },
                            ],
                        }
                    ]
                }
            else:
                # Before creation, stack doesn't exist
                raise ClientError(
                    {"Error": {"Code": "ValidationError", "Message": "does not exist"}},
                    "DescribeStacks",
                )

        self.manager.cfn.describe_stacks.side_effect = describe_stacks_side_effect

        # Mock successful stack creation
        self.manager.cfn.create_stack.return_value = {"StackId": "stack-123"}
        self.manager.cfn.get_waiter.return_value.wait = MagicMock()

        with patch.object(Path, "exists", return_value=True):
            outputs = self.manager._ensure_infrastructure_exists()

        self.assertEqual(outputs["DynamoDBTableName"], "new-table")
        self.manager.cfn.create_stack.assert_called_once()

    def test_ensure_infrastructure_exists_template_not_found(self):
        """Test _ensure_infrastructure_exists raises error when template not found."""
        # Mock stack doesn't exist
        self.manager.cfn.describe_stacks.side_effect = ClientError(
            {"Error": {"Code": "ValidationError", "Message": "does not exist"}},
            "DescribeStacks",
        )

        with patch.object(Path, "exists", return_value=False):
            with self.assertRaises(NotificationManagerInfraError) as context:
                self.manager._ensure_infrastructure_exists()
            self.assertIn("template not found", str(context.exception))

    @patch("amzn_nova_forge.notifications.notification_manager.logger")
    @patch("amzn_nova_forge.notifications.notification_manager.time")
    def test_enable_notifications_success(self, mock_time, mock_logger):
        """Test enable_notifications successfully enables notifications."""
        mock_time.time.return_value = 1000000000

        # Mock infrastructure exists
        self.manager.cfn.describe_stacks.return_value = {
            "Stacks": [
                {
                    "StackStatus": "CREATE_COMPLETE",
                    "Outputs": [
                        {"OutputKey": "DynamoDBTableName", "OutputValue": "test-table"},
                        {
                            "OutputKey": "SNSTopicArn",
                            "OutputValue": "arn:aws:sns:us-east-1:123456789012:topic",
                        },
                    ],
                }
            ]
        }

        # Mock SNS list subscriptions (no existing subscriptions)
        self.manager.sns.list_subscriptions_by_topic.return_value = {"Subscriptions": []}

        job_name = "test-job-123"
        emails = ["user@example.com"]
        output_s3_path = "s3://bucket/path"

        self.manager.enable_notifications(job_name, emails, output_s3_path)

        # Verify DynamoDB put_item was called
        self.manager.dynamodb.put_item.assert_called_once()
        call_args = self.manager.dynamodb.put_item.call_args
        self.assertEqual(call_args[1]["TableName"], "test-table")
        self.assertEqual(call_args[1]["Item"]["job_id"]["S"], job_name)

        # Verify SNS subscribe was called
        self.manager.sns.subscribe.assert_called_once_with(
            TopicArn="arn:aws:sns:us-east-1:123456789012:topic",
            Protocol="email",
            Endpoint="user@example.com",
            ReturnSubscriptionArn=True,
        )

    @patch("amzn_nova_forge.notifications.notification_manager.logger")
    @patch("amzn_nova_forge.notifications.notification_manager.time")
    def test_enable_notifications_with_kms(self, mock_time, mock_logger):
        """Test enable_notifications with KMS key parameter."""
        mock_time.time.return_value = 1000000000

        # Mock infrastructure exists
        self.manager.cfn.describe_stacks.return_value = {
            "Stacks": [
                {
                    "StackStatus": "CREATE_COMPLETE",
                    "Outputs": [
                        {"OutputKey": "DynamoDBTableName", "OutputValue": "test-table"},
                        {
                            "OutputKey": "SNSTopicArn",
                            "OutputValue": "arn:aws:sns:us-east-1:123456789012:topic",
                        },
                    ],
                }
            ]
        }

        self.manager.sns.list_subscriptions_by_topic.return_value = {"Subscriptions": []}

        kms_key_id = "arn:aws:kms:us-east-1:123456789012:key/abc-123"
        self.manager.enable_notifications(
            "test-job-123",
            ["user@example.com"],
            "s3://bucket/path",
            kms_key_id=kms_key_id,
        )

        # Verify DynamoDB put_item was called
        self.manager.dynamodb.put_item.assert_called_once()

    def test_enable_notifications_invalid_job_name(self):
        """Test enable_notifications raises error for empty job name."""
        with self.assertRaises(ValueError) as context:
            self.manager.enable_notifications("", ["user@example.com"], "s3://bucket/path")
        self.assertIn("job_name cannot be empty", str(context.exception))

    def test_enable_notifications_invalid_emails(self):
        """Test enable_notifications raises error for invalid emails."""
        with self.assertRaises(ValueError) as context:
            self.manager.enable_notifications("job-123", [], "s3://bucket/path")
        self.assertIn("emails must be a non-empty list", str(context.exception))

    def test_enable_notifications_missing_output_s3_path(self):
        """Test enable_notifications raises error when output_s3_path is missing."""
        with self.assertRaises(ValueError) as context:
            self.manager.enable_notifications("job-123", ["user@example.com"], "")
        self.assertIn("output_s3_path is required", str(context.exception))

    @patch("amzn_nova_forge.notifications.notification_manager.logger")
    def test_enable_notifications_skips_existing_subscriptions(self, mock_logger):
        """Test enable_notifications skips emails that are already subscribed."""
        # Mock infrastructure exists
        self.manager.cfn.describe_stacks.return_value = {
            "Stacks": [
                {
                    "StackStatus": "CREATE_COMPLETE",
                    "Outputs": [
                        {"OutputKey": "DynamoDBTableName", "OutputValue": "test-table"},
                        {
                            "OutputKey": "SNSTopicArn",
                            "OutputValue": "arn:aws:sns:us-east-1:123456789012:topic",
                        },
                    ],
                }
            ]
        }

        # Mock existing subscription
        self.manager.sns.list_subscriptions_by_topic.return_value = {
            "Subscriptions": [{"Protocol": "email", "Endpoint": "user@example.com"}]
        }

        self.manager.enable_notifications("job-123", ["user@example.com"], "s3://bucket/path")

        # Verify SNS subscribe was NOT called (already subscribed)
        self.manager.sns.subscribe.assert_not_called()
        mock_logger.info.assert_any_call(
            "Email user@example.com is already subscribed to the topic"
        )

    @patch("amzn_nova_forge.notifications.notification_manager.logger")
    def test_delete_notification_stack_success(self, mock_logger):
        """Test delete_notification_stack successfully initiates deletion."""
        # Mock stack exists
        self.manager.cfn.describe_stacks.return_value = {
            "Stacks": [{"StackStatus": "CREATE_COMPLETE"}]
        }

        self.manager.delete_notification_stack()

        # Verify delete_stack was called
        self.manager.cfn.delete_stack.assert_called_once_with(
            StackName="NovaForgeSDK-SMTJ-JobNotifications"
        )

        # Verify waiter was NOT called (no polling)
        self.manager.cfn.get_waiter.assert_not_called()

        # Verify user-friendly message was logged with console URL
        logged_messages = [call[0][0] for call in mock_logger.info.call_args_list]
        self.assertTrue(
            any("cloudformation/home" in msg for msg in logged_messages),
            "Expected console URL in log messages",
        )

    @patch("amzn_nova_forge.notifications.notification_manager.logger")
    def test_delete_notification_stack_already_deleted(self, mock_logger):
        """Test delete_notification_stack handles already deleted stack."""
        # Mock stack doesn't exist
        self.manager.cfn.describe_stacks.side_effect = ClientError(
            {"Error": {"Code": "ValidationError", "Message": "does not exist"}},
            "DescribeStacks",
        )

        self.manager.delete_notification_stack()

        # Should log info and return without error
        mock_logger.info.assert_called_with(
            "Stack NovaForgeSDK-SMTJ-JobNotifications does not exist, nothing to delete"
        )
        self.manager.cfn.delete_stack.assert_not_called()

    @patch("amzn_nova_forge.notifications.notification_manager.logger")
    def test_delete_notification_stack_deletion_in_progress(self, mock_logger):
        """Test delete_notification_stack handles deletion already in progress."""
        # Mock stack is being deleted
        self.manager.cfn.describe_stacks.return_value = {
            "Stacks": [{"StackStatus": "DELETE_IN_PROGRESS"}]
        }

        self.manager.delete_notification_stack()

        # Should log info and return without calling delete again
        mock_logger.info.assert_called_with(
            "Stack NovaForgeSDK-SMTJ-JobNotifications is already being deleted or has been deleted"
        )
        self.manager.cfn.delete_stack.assert_not_called()

    def test_delete_notification_stack_failure(self):
        """Test delete_notification_stack raises error on failure."""
        # Mock stack exists
        self.manager.cfn.describe_stacks.return_value = {
            "Stacks": [{"StackStatus": "CREATE_COMPLETE"}]
        }

        # Mock delete failure
        self.manager.cfn.delete_stack.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "Access denied"}},
            "DeleteStack",
        )

        with self.assertRaises(NotificationManagerInfraError) as context:
            self.manager.delete_notification_stack()
        self.assertIn("Failed to delete stack", str(context.exception))

    def test_stack_name_length_within_limit(self):
        """Test SMTJ stack name is within CloudFormation's 128 character limit."""
        stack_name = self.manager.get_stack_name()
        self.assertLessEqual(
            len(stack_name),
            self.manager.CLOUDFORMATION_STACK_NAME_MAX_LENGTH,
            f"SMTJ stack name '{stack_name}' exceeds 128 character limit",
        )


class TestSMHPNotificationManager(unittest.TestCase):
    """Test suite for SMHPNotificationManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.region = "us-west-2"
        self.cluster_name = "test-cluster"

        # Create manager with mocked AWS clients
        with patch("amzn_nova_forge.notifications.notification_manager.boto3"):
            self.manager = SMHPNotificationManager(
                cluster_name=self.cluster_name, region=self.region
            )
            self.manager.cfn = MagicMock()
            self.manager.dynamodb = MagicMock()
            self.manager.sns = MagicMock()

    def test_initialization(self):
        """Test SMHPNotificationManager initialization."""
        self.assertEqual(self.manager.platform, Platform.SMHP)
        self.assertEqual(self.manager.region, "us-west-2")
        self.assertEqual(self.manager.cluster_name, "test-cluster")

    def test_get_stack_name(self):
        """Test get_stack_name returns correct name for SMHP."""
        stack_name = self.manager.get_stack_name()
        self.assertEqual(stack_name, "NovaForgeSDK-SMHP-JobNotifications-test-cluster")

    def test_get_template_path(self):
        """Test get_template_path returns correct path for SMHP."""
        template_path = self.manager.get_template_path()
        self.assertTrue(str(template_path).endswith("smhp_notification_cf_stack.yaml"))

    def test_get_platform_name(self):
        """Test get_platform_name returns correct platform name."""
        platform_name = self.manager.get_platform_name()
        self.assertEqual(platform_name, "SMHP")

    def test_get_stack_parameters(self):
        """Test _get_stack_parameters with SMHP-specific parameters."""
        parameters = self.manager._get_stack_parameters(
            eks_cluster_arn="arn:aws:eks:us-west-2:123456789012:cluster/test",
            vpc_id="vpc-12345",
            subnet_ids=["subnet-1", "subnet-2"],
            security_group_id="sg-12345",
            kubectl_layer_arn="arn:aws:lambda:us-west-2:123456789012:layer:kubectl:1",
        )

        # Verify required parameters are present
        param_dict = {p["ParameterKey"]: p["ParameterValue"] for p in parameters}
        self.assertEqual(param_dict["ClusterName"], "test-cluster")
        self.assertEqual(
            param_dict["EksClusterArn"],
            "arn:aws:eks:us-west-2:123456789012:cluster/test",
        )
        self.assertEqual(param_dict["VpcId"], "vpc-12345")
        self.assertEqual(param_dict["SubnetIds"], "subnet-1,subnet-2")
        self.assertEqual(param_dict["SecurityGroupId"], "sg-12345")
        self.assertEqual(
            param_dict["KubectlLayerArn"],
            "arn:aws:lambda:us-west-2:123456789012:layer:kubectl:1",
        )

    @patch("amzn_nova_forge.notifications.smhp_notification_manager.logger")
    def test_enable_notifications_requires_namespace(self, mock_logger):
        """Test enable_notifications raises error when namespace is not provided."""
        with self.assertRaises(ValueError) as context:
            self.manager.enable_notifications(
                job_name="test-job",
                emails=["user@example.com"],
                output_s3_path="s3://bucket/path",
                namespace=None,  # Missing namespace
                kubectl_layer_arn="arn:aws:lambda:us-west-2:123456789012:layer:kubectl:1",
            )
        self.assertIn("namespace is required", str(context.exception))

    @patch("amzn_nova_forge.notifications.smhp_notification_manager.logger")
    def test_enable_notifications_requires_kubectl_layer(self, mock_logger):
        """Test enable_notifications raises error when kubectl_layer_arn is not provided."""
        # Mock infrastructure exists
        self.manager.cfn.describe_stacks.return_value = {
            "Stacks": [
                {
                    "StackStatus": "CREATE_COMPLETE",
                    "Outputs": [
                        {"OutputKey": "DynamoDBTableName", "OutputValue": "test-table"},
                        {
                            "OutputKey": "SNSTopicArn",
                            "OutputValue": "arn:aws:sns:us-west-2:123456789012:topic",
                        },
                    ],
                }
            ]
        }

        # Mock cluster info for auto-detection
        self.manager._sagemaker_client = MagicMock()
        self.manager._sagemaker_client.describe_cluster.return_value = {
            "Orchestrator": {
                "Eks": {"ClusterArn": "arn:aws:eks:us-west-2:123456789012:cluster/test"}
            },
            "VpcConfig": {
                "Subnets": ["subnet-1", "subnet-2"],
                "SecurityGroupIds": ["sg-12345"],
            },
        }
        self.manager._ec2_client = MagicMock()
        self.manager._ec2_client.describe_subnets.return_value = {
            "Subnets": [{"VpcId": "vpc-12345"}]
        }

        with self.assertRaises(ValueError) as context:
            self.manager.enable_notifications(
                job_name="test-job",
                emails=["user@example.com"],
                output_s3_path="s3://bucket/path",
                namespace="kubeflow",
                # Missing kubectl_layer_arn
            )
        self.assertIn("kubectl_layer_arn is required", str(context.exception))

    @patch("amzn_nova_forge.notifications.smhp_notification_manager.logger")
    @patch("amzn_nova_forge.notifications.notification_manager.time")
    def test_enable_notifications_stores_namespace_in_dynamodb(self, mock_time, mock_logger):
        """Test enable_notifications stores namespace in DynamoDB."""
        mock_time.time.return_value = 1000000000

        # Mock infrastructure exists
        self.manager.cfn.describe_stacks.return_value = {
            "Stacks": [
                {
                    "StackStatus": "CREATE_COMPLETE",
                    "Outputs": [
                        {"OutputKey": "DynamoDBTableName", "OutputValue": "test-table"},
                        {
                            "OutputKey": "SNSTopicArn",
                            "OutputValue": "arn:aws:sns:us-west-2:123456789012:topic",
                        },
                    ],
                }
            ]
        }

        # Mock cluster info for auto-detection
        self.manager._sagemaker_client = MagicMock()
        self.manager._sagemaker_client.describe_cluster.return_value = {
            "Orchestrator": {
                "Eks": {"ClusterArn": "arn:aws:eks:us-west-2:123456789012:cluster/test"}
            },
            "VpcConfig": {
                "Subnets": ["subnet-1", "subnet-2"],
                "SecurityGroupIds": ["sg-12345"],
            },
        }
        self.manager._ec2_client = MagicMock()
        self.manager._ec2_client.describe_subnets.return_value = {
            "Subnets": [{"VpcId": "vpc-12345"}]
        }
        self.manager._ec2_client.describe_route_tables.return_value = {
            "RouteTables": [{"RouteTableId": "rtb-12345"}]
        }
        self.manager._ec2_client.describe_vpc_endpoints.return_value = {"VpcEndpoints": []}

        self.manager.sns.list_subscriptions_by_topic.return_value = {"Subscriptions": []}

        self.manager.enable_notifications(
            job_name="test-job",
            emails=["user@example.com"],
            output_s3_path="s3://bucket/path",
            namespace="kubeflow",
            kubectl_layer_arn="arn:aws:lambda:us-west-2:123456789012:layer:kubectl:1",
        )

        # Verify DynamoDB put_item was called with namespace
        self.manager.dynamodb.put_item.assert_called_once()
        call_args = self.manager.dynamodb.put_item.call_args
        item = call_args[1]["Item"]
        self.assertEqual(item["namespace"]["S"], "kubeflow")

    @patch("amzn_nova_forge.notifications.smhp_notification_manager.logger")
    def test_get_cluster_info_failure(self, mock_logger):
        """Test _get_cluster_info raises error when cluster info cannot be retrieved."""
        self.manager._sagemaker_client = MagicMock()
        self.manager._sagemaker_client.describe_cluster.side_effect = ClientError(
            {"Error": {"Code": "ResourceNotFound", "Message": "Cluster not found"}},
            "DescribeCluster",
        )

        with self.assertRaises(NotificationManagerInfraError) as context:
            self.manager._get_cluster_info()
        self.assertIn("Failed to get cluster info", str(context.exception))

    @patch("amzn_nova_forge.notifications.smhp_notification_manager.logger")
    def test_get_vpc_id_from_subnet(self, mock_logger):
        """Test _get_vpc_id_from_subnet retrieves VPC ID from subnet."""
        self.manager._ec2_client = MagicMock()
        self.manager._ec2_client.describe_subnets.return_value = {
            "Subnets": [{"VpcId": "vpc-12345"}]
        }

        vpc_id = self.manager._get_vpc_id_from_subnet("subnet-1")

        self.assertEqual(vpc_id, "vpc-12345")
        self.manager._ec2_client.describe_subnets.assert_called_once_with(SubnetIds=["subnet-1"])

    @patch("amzn_nova_forge.notifications.smhp_notification_manager.logger")
    def test_check_existing_vpc_endpoints(self, mock_logger):
        """Test _check_existing_vpc_endpoints detects existing endpoints."""
        self.manager._ec2_client = MagicMock()
        self.manager._ec2_client.describe_vpc_endpoints.return_value = {
            "VpcEndpoints": [
                {"ServiceName": "com.amazonaws.us-west-2.dynamodb"},
                {"ServiceName": "com.amazonaws.us-west-2.s3"},
            ]
        }

        result = self.manager._check_existing_vpc_endpoints("vpc-12345")

        self.assertTrue(result["dynamodb"])
        self.assertTrue(result["s3"])

    @patch("amzn_nova_forge.notifications.smhp_notification_manager.logger")
    def test_get_route_table_ids_from_subnets(self, mock_logger):
        """Test _get_route_table_ids_from_subnets retrieves route table IDs."""
        self.manager._ec2_client = MagicMock()
        self.manager._ec2_client.describe_route_tables.return_value = {
            "RouteTables": [
                {
                    "RouteTableId": "rtb-1",
                    "Associations": [{"SubnetId": "subnet-1"}],
                },
                {
                    "RouteTableId": "rtb-2",
                    "Associations": [{"SubnetId": "subnet-2"}],
                },
            ]
        }

        route_table_ids = self.manager._get_route_table_ids_from_subnets(["subnet-1", "subnet-2"])

        self.assertEqual(set(route_table_ids), {"rtb-1", "rtb-2"})

    @patch("amzn_nova_forge.notifications.notification_manager.logger")
    def test_delete_notification_stack(self, mock_logger):
        """Test delete_notification_stack returns immediately without polling."""
        # Mock stack exists
        self.manager.cfn.describe_stacks.return_value = {
            "Stacks": [{"StackStatus": "CREATE_COMPLETE"}]
        }

        self.manager.delete_notification_stack()

        # Verify delete_stack was called
        self.manager.cfn.delete_stack.assert_called_once_with(
            StackName="NovaForgeSDK-SMHP-JobNotifications-test-cluster"
        )

        logged_messages = [call[0][0] for call in mock_logger.info.call_args_list]
        console_url_logged = any("cloudformation/home" in msg for msg in logged_messages)
        self.assertTrue(
            console_url_logged,
            "Expected CloudFormation console URL in log messages to guide users",
        )

    @patch("amzn_nova_forge.notifications.smhp_notification_manager.logger")
    def test_enable_notifications_missing_eks_cluster_arn_after_auto_detection(self, mock_logger):
        """Test enable_notifications raises ValueError when eks_cluster_arn is missing after auto-detection."""
        # Mock auto-detection that returns partial cluster info (missing EKS ARN)
        self.manager._sagemaker_client = MagicMock()
        self.manager._sagemaker_client.describe_cluster.return_value = {
            "Orchestrator": {
                "Eks": {}  # Missing ClusterArn
            },
            "VpcConfig": {
                "Subnets": ["subnet-1", "subnet-2"],
                "SecurityGroupIds": ["sg-12345"],
            },
        }
        self.manager._ec2_client = MagicMock()
        self.manager._ec2_client.describe_subnets.return_value = {
            "Subnets": [{"VpcId": "vpc-12345"}]
        }

        with self.assertRaises(ValueError) as context:
            self.manager.enable_notifications(
                job_name="test-job",
                emails=["user@example.com"],
                output_s3_path="s3://bucket/path",
                namespace="kubeflow",
                kubectl_layer_arn="arn:aws:lambda:us-west-2:123456789012:layer:kubectl:1",
            )
        self.assertIn("eks_cluster_arn is required", str(context.exception))

    @patch("amzn_nova_forge.notifications.smhp_notification_manager.logger")
    def test_enable_notifications_missing_vpc_id_after_auto_detection(self, mock_logger):
        """Test enable_notifications raises ValueError when vpc_id is missing after auto-detection."""
        # Mock auto-detection that returns partial cluster info (missing VPC ID)
        self.manager._sagemaker_client = MagicMock()
        self.manager._sagemaker_client.describe_cluster.return_value = {
            "Orchestrator": {
                "Eks": {"ClusterArn": "arn:aws:eks:us-west-2:123456789012:cluster/test"}
            },
            "VpcConfig": {
                "Subnets": ["subnet-1", "subnet-2"],
                "SecurityGroupIds": ["sg-12345"],
            },
        }
        # Mock EC2 client that fails to return VPC ID
        self.manager._ec2_client = MagicMock()
        self.manager._ec2_client.describe_subnets.return_value = {"Subnets": []}

        with self.assertRaises(ValueError) as context:
            self.manager.enable_notifications(
                job_name="test-job",
                emails=["user@example.com"],
                output_s3_path="s3://bucket/path",
                namespace="kubeflow",
                kubectl_layer_arn="arn:aws:lambda:us-west-2:123456789012:layer:kubectl:1",
            )
        self.assertIn("vpc_id is required", str(context.exception))

    @patch("amzn_nova_forge.notifications.smhp_notification_manager.logger")
    def test_enable_notifications_missing_subnet_ids_after_auto_detection(self, mock_logger):
        """Test enable_notifications raises ValueError when subnet_ids is missing after auto-detection."""
        # Mock auto-detection that returns partial cluster info (missing subnets)
        # We need to provide vpc_id explicitly since it can't be derived from empty subnets
        self.manager._sagemaker_client = MagicMock()
        self.manager._sagemaker_client.describe_cluster.return_value = {
            "Orchestrator": {
                "Eks": {"ClusterArn": "arn:aws:eks:us-west-2:123456789012:cluster/test"}
            },
            "VpcConfig": {
                "Subnets": [],  # Empty subnets list
                "SecurityGroupIds": ["sg-12345"],
            },
        }
        self.manager._ec2_client = MagicMock()

        with self.assertRaises(ValueError) as context:
            self.manager.enable_notifications(
                job_name="test-job",
                emails=["user@example.com"],
                output_s3_path="s3://bucket/path",
                namespace="kubeflow",
                kubectl_layer_arn="arn:aws:lambda:us-west-2:123456789012:layer:kubectl:1",
                vpc_id="vpc-12345",  # Provide vpc_id explicitly to isolate subnet_ids validation
            )
        self.assertIn("subnet_ids must be a non-empty list", str(context.exception))

    @patch("amzn_nova_forge.notifications.smhp_notification_manager.logger")
    def test_enable_notifications_invalid_subnet_ids_type_after_auto_detection(self, mock_logger):
        """Test enable_notifications raises ValueError when subnet_ids is not a list after auto-detection."""
        # Mock auto-detection that returns partial cluster info (subnet_ids wrong type)
        self.manager._sagemaker_client = MagicMock()
        self.manager._sagemaker_client.describe_cluster.return_value = {
            "Orchestrator": {
                "Eks": {"ClusterArn": "arn:aws:eks:us-west-2:123456789012:cluster/test"}
            },
            "VpcConfig": {
                "Subnets": ["subnet-1"],
                "SecurityGroupIds": ["sg-12345"],
            },
        }
        self.manager._ec2_client = MagicMock()
        self.manager._ec2_client.describe_subnets.return_value = {
            "Subnets": [{"VpcId": "vpc-12345"}]
        }

        # Explicitly provide subnet_ids as a string instead of list
        with self.assertRaises(ValueError) as context:
            self.manager.enable_notifications(
                job_name="test-job",
                emails=["user@example.com"],
                output_s3_path="s3://bucket/path",
                namespace="kubeflow",
                kubectl_layer_arn="arn:aws:lambda:us-west-2:123456789012:layer:kubectl:1",
                subnet_ids="subnet-1",  # Wrong type: string instead of list
            )
        self.assertIn("subnet_ids must be a non-empty list", str(context.exception))

    @patch("amzn_nova_forge.notifications.smhp_notification_manager.logger")
    def test_enable_notifications_missing_security_group_id_after_auto_detection(self, mock_logger):
        """Test enable_notifications raises ValueError when security_group_id is missing after auto-detection."""
        # Mock auto-detection that returns partial cluster info (missing security group)
        self.manager._sagemaker_client = MagicMock()
        self.manager._sagemaker_client.describe_cluster.return_value = {
            "Orchestrator": {
                "Eks": {"ClusterArn": "arn:aws:eks:us-west-2:123456789012:cluster/test"}
            },
            "VpcConfig": {
                "Subnets": ["subnet-1", "subnet-2"],
                "SecurityGroupIds": [],  # Empty security groups list
            },
        }
        self.manager._ec2_client = MagicMock()
        self.manager._ec2_client.describe_subnets.return_value = {
            "Subnets": [{"VpcId": "vpc-12345"}]
        }

        with self.assertRaises(ValueError) as context:
            self.manager.enable_notifications(
                job_name="test-job",
                emails=["user@example.com"],
                output_s3_path="s3://bucket/path",
                namespace="kubeflow",
                kubectl_layer_arn="arn:aws:lambda:us-west-2:123456789012:layer:kubectl:1",
            )
        self.assertIn("security_group_id is required", str(context.exception))

    def test_cluster_name_length_valid_short(self):
        """Test SMHP accepts short cluster names."""
        with patch("amzn_nova_forge.notifications.notification_manager.boto3"):
            # Should not raise
            manager = SMHPNotificationManager(cluster_name="test", region="us-west-2")
            self.assertEqual(manager.cluster_name, "test")

    def test_cluster_name_length_valid_at_limit(self):
        """Test SMHP accepts cluster name at 30 character limit."""
        cluster_name = "a" * 30
        with patch("amzn_nova_forge.notifications.notification_manager.boto3"):
            # Should not raise
            manager = SMHPNotificationManager(cluster_name=cluster_name, region="us-west-2")
            self.assertEqual(manager.cluster_name, cluster_name)

    def test_cluster_name_length_exceeds_limit_by_one(self):
        """Test SMHP rejects cluster name that exceeds 30 character limit by 1."""
        cluster_name = "a" * 31
        with patch("amzn_nova_forge.notifications.notification_manager.boto3"):
            with self.assertRaises(ValueError) as context:
                SMHPNotificationManager(cluster_name=cluster_name, region="us-west-2")

            error_msg = str(context.exception)
            self.assertIn(f"Cluster name '{cluster_name}' is too long", error_msg)
            self.assertIn("31 characters", error_msg)
            self.assertIn("Maximum length is 30 characters", error_msg)

    def test_cluster_name_length_far_exceeds_limit(self):
        """Test SMHP rejects very long cluster names."""
        cluster_name = "very-long-cluster-name-" * 10  # 230 characters
        with patch("amzn_nova_forge.notifications.notification_manager.boto3"):
            with self.assertRaises(ValueError) as context:
                SMHPNotificationManager(cluster_name=cluster_name, region="us-west-2")

            error_msg = str(context.exception)
            self.assertIn("is too long", error_msg)
            self.assertIn("230 characters", error_msg)
            self.assertIn("Maximum length is 30 characters", error_msg)


if __name__ == "__main__":
    unittest.main()
