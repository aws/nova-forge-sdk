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
import re
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import boto3
from botocore.exceptions import ClientError

from amzn_nova_forge.core.enums import Platform
from amzn_nova_forge.telemetry import Feature, _telemetry_emitter
from amzn_nova_forge.util.logging import logger


class NotificationManagerInfraError(Exception):
    """Raised when there's an issue with notification manager infrastructure."""

    pass


class NotificationManager(ABC):
    """
    Base class for notification management. Handles infrastructure setup
    and notification configuration for training jobs.
    """

    CLOUDFORMATION_STACK_NAME_MAX_LENGTH = 128

    def __init__(self, platform: Platform, region: str = "us-east-1"):
        """
        Initialize the notification manager.

        Args:
            platform: Platform (Platform.SMTJ or Platform.SMHP)
            region: AWS region for the infrastructure
        """
        self.platform = platform
        self.region = region
        self.cfn = boto3.client("cloudformation", region_name=region)
        self.dynamodb = boto3.client("dynamodb", region_name=region)
        self.sns = boto3.client("sns", region_name=region)

    @abstractmethod
    def get_stack_name(self) -> str:
        """Get the CloudFormation stack name for this platform."""
        pass

    @abstractmethod
    def get_template_path(self) -> Path:
        """Get the path to the CloudFormation template for this platform."""
        pass

    def get_platform_name(self) -> str:
        """Get the platform name (e.g., 'SMTJ', 'SMHP')."""
        return self.platform.value

    def _get_stack_parameters(self, **kwargs) -> List[dict]:
        """
        Get CloudFormation stack parameters.
        Override in child classes to provide platform-specific parameters.

        Returns:
            List of CloudFormation parameter dicts in format:
            [{"ParameterKey": "KeyName", "ParameterValue": "value"}, ...]
        """
        return []

    @staticmethod
    def _validate_email(email: str) -> bool:
        """Validate email format."""
        pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        return re.match(pattern, email) is not None

    @staticmethod
    def _parse_stack_outputs(outputs: List[dict]) -> dict:
        """Parse CloudFormation stack outputs into a dict."""
        return {output["OutputKey"]: output["OutputValue"] for output in outputs}

    def _ensure_infrastructure_exists(self, **kwargs) -> dict:
        """
        Ensure the necessary notification infrastructure exists.
        If it doesn't, create the infra with platform-specific parameters.

        Args:
            **kwargs: Platform-specific parameters passed to _get_stack_parameters

        Returns:
            dict: Stack outputs containing resource ARNs/names

        Raises:
            NotificationManagerInfraError: If stack creation fails
            ValueError: If stack name exceeds CF's 128 character limit
        """
        stack_name = self.get_stack_name()

        # Validate stack name length (128 max)
        if len(stack_name) > self.CLOUDFORMATION_STACK_NAME_MAX_LENGTH:
            raise ValueError(
                f"CloudFormation stack name exceeds {self.CLOUDFORMATION_STACK_NAME_MAX_LENGTH} "
                f"character limit: '{stack_name}' ({len(stack_name)} characters). "
                f"Please use a shorter cluster name."
            )

        parameters = self._get_stack_parameters(**kwargs)

        try:
            # Check if stack exists
            response = self.cfn.describe_stacks(StackName=stack_name)
            stack = response["Stacks"][0]
            status = stack["StackStatus"]

            # Verify state of stack, polls for completion if creation is in progress.
            if status in ["CREATE_COMPLETE", "UPDATE_COMPLETE"]:
                logger.info(f"Notification infrastructure already exists: {stack_name}")
                return self._parse_stack_outputs(stack.get("Outputs", []))
            elif status == "CREATE_IN_PROGRESS":
                # Wait for stack creation
                logger.info(f"Stack {stack_name} is being created. Waiting...")
                waiter = self.cfn.get_waiter("stack_create_complete")
                waiter.wait(StackName=stack_name)
                # Creation complete
                response = self.cfn.describe_stacks(StackName=stack_name)
                return self._parse_stack_outputs(response["Stacks"][0].get("Outputs", []))
            elif status == "UPDATE_IN_PROGRESS":
                # Wait for stack update
                logger.info(f"Stack {stack_name} is being updated. Waiting...")
                waiter = self.cfn.get_waiter("stack_update_complete")
                waiter.wait(StackName=stack_name)
                # Update complete
                response = self.cfn.describe_stacks(StackName=stack_name)
                return self._parse_stack_outputs(response["Stacks"][0].get("Outputs", []))
            else:
                raise NotificationManagerInfraError(
                    f"Stack {stack_name} is in unexpected state: {status}"
                )

        except ClientError as e:
            if "does not exist" not in str(e):
                raise NotificationManagerInfraError(f"Error checking stack status: {e}")

            # Stack doesn't exist, create it
            logger.info(f"Creating notification infrastructure: {stack_name}")

            template_path = self.get_template_path()
            if not template_path.exists():
                raise NotificationManagerInfraError(
                    f"CloudFormation template not found: {template_path}"
                )

            with open(template_path, "r") as f:
                template_body = f.read()

            try:
                create_stack_params: dict = {
                    "StackName": stack_name,
                    "TemplateBody": template_body,
                    "Capabilities": ["CAPABILITY_NAMED_IAM"],
                }

                # Add parameters if provided
                if parameters:
                    create_stack_params["Parameters"] = parameters

                self.cfn.create_stack(**create_stack_params)

                # Wait for stack creation
                waiter = self.cfn.get_waiter("stack_create_complete")
                waiter.wait(StackName=stack_name)

                logger.info(f"Successfully created stack: {stack_name}")

                # Get stack outputs
                response = self.cfn.describe_stacks(StackName=stack_name)
                return self._parse_stack_outputs(response["Stacks"][0].get("Outputs", []))

            except ClientError as e:
                raise NotificationManagerInfraError(f"Failed to create stack: {e}")

    @_telemetry_emitter(
        Feature.INFRA,
        "enable_notifications",
        extra_info_fn=lambda self, *args, **kwargs: {
            "platform": self.platform,
        },
    )
    def enable_notifications(
        self,
        job_name: str,
        emails: List[str],
        output_s3_path: str,
        namespace: Optional[str] = None,
        **platform_kwargs,
    ) -> None:
        """
        Enable email notifications for a specific job name.
        Notifications will be sent when the job reaches a terminal state.

        Args:
            job_name: Job name/ID
            emails: List of email addresses to notify
            output_s3_path: S3 path where job outputs are stored (required for manifest validation)
            namespace: Kubernetes namespace (for SMHP only, not needed for SMTJ)
            **platform_kwargs: Platform-specific parameters (e.g., kms_key_id for SMTJ)

        Raises:
            ValueError: If inputs are invalid or output_s3_path is missing
            NotificationManagerInfraError: If infrastructure setup fails
        """
        # Validate inputs
        if not job_name:
            raise ValueError("job_name cannot be empty")

        if not emails or not isinstance(emails, list):
            raise ValueError("emails must be a non-empty list")

        for email in emails:
            if not self._validate_email(email):
                raise ValueError(f"Invalid email format: {email}")

        if not output_s3_path:
            raise ValueError(
                "output_s3_path is required for job notifications. "
                "This is needed to validate the manifest file when the job completes."
            )

        # Ensure infrastructure exists -- attempts to set up if it doesn't exist.
        try:
            outputs = self._ensure_infrastructure_exists(**platform_kwargs)
            table_name = outputs["DynamoDBTableName"]
            topic_arn = outputs["SNSTopicArn"]
        except Exception as e:
            raise NotificationManagerInfraError(f"Failed to set up infrastructure: {e}")

        # Store job configuration in DynamoDB
        try:
            item = {
                "job_id": {"S": job_name},
                "emails": {"SS": emails},
                "output_s3_path": {"S": output_s3_path},
                "created_at": {"S": datetime.now(timezone.utc).isoformat()},
                "ttl": {"N": str(int(time.time()) + 30 * 24 * 60 * 60)},  # 30 days
            }

            # Add namespace for SMHP jobs
            if namespace:
                item["namespace"] = {"S": namespace}

            self.dynamodb.put_item(TableName=table_name, Item=item)
            logger.info(f"Stored notification config for job {job_name}")
        except ClientError as e:
            raise NotificationManagerInfraError(f"Failed to store job configuration: {e}")

        # Subscribe emails to SNS topic (check for existing subscriptions first)
        subscribed_emails = []

        # Get existing subscriptions to avoid duplicates
        try:
            existing_subs = self.sns.list_subscriptions_by_topic(TopicArn=topic_arn)
            existing_emails = {
                sub["Endpoint"]
                for sub in existing_subs.get("Subscriptions", [])
                if sub["Protocol"] == "email"
            }
        except ClientError as e:
            logger.warning(f"Could not list existing subscriptions: {e}")
            existing_emails = set()

        for email in emails:
            if email in existing_emails:
                logger.info(f"Email {email} is already subscribed to the topic")
                subscribed_emails.append(email)
            else:
                try:
                    self.sns.subscribe(
                        TopicArn=topic_arn,
                        Protocol="email",
                        Endpoint=email,
                        ReturnSubscriptionArn=True,
                    )
                    subscribed_emails.append(email)
                    logger.info(f"Subscription request sent to {email}")
                except ClientError as e:
                    logger.warning(f"Failed to subscribe {email}: {e}")

        logger.info(f"Notifications enabled for {self.get_platform_name()} job: {job_name}")
        logger.info(f"Emails: {', '.join(emails)}")
        logger.info(
            "Note: Users must confirm their email subscriptions by clicking the link in the confirmation email from AWS."
        )

    @_telemetry_emitter(
        Feature.INFRA,
        "delete_notification_stack",
        extra_info_fn=lambda self, *args, **kwargs: {
            "platform": self.platform,
        },
    )
    def delete_notification_stack(self) -> None:
        """
        Delete the CloudFormation stack and all associated resources for this platform.

        This will remove:
        - DynamoDB table (all job notification configurations)
        - SNS topic (all email subscriptions)
        - Lambda function (notification handler)
        - EventBridge rule (job state change monitoring)
        - IAM roles and policies

        Raises:
            NotificationManagerInfraError: If stack deletion fails
        """
        stack_name = self.get_stack_name()

        try:
            # Check if stack exists
            try:
                response = self.cfn.describe_stacks(StackName=stack_name)
                stack_status = response["Stacks"][0]["StackStatus"]

                # Check if stack is already being deleted or doesn't exist
                if stack_status in ["DELETE_IN_PROGRESS", "DELETE_COMPLETE"]:
                    logger.info(f"Stack {stack_name} is already being deleted or has been deleted")
                    return

            except ClientError as e:
                if "does not exist" in str(e):
                    logger.info(f"Stack {stack_name} does not exist, nothing to delete")
                    return
                raise

            # Delete the stack
            logger.info(f"Deleting notification infrastructure stack: {stack_name}")
            self.cfn.delete_stack(StackName=stack_name)

            logger.info(
                f"Stack deletion initiated for {stack_name}. "
                f"This may take several minutes (especially for SMHP stacks with VPC resources). "
                f"Check the CloudFormation console to monitor deletion progress: "
                f"https://console.aws.amazon.com/cloudformation/home?region={self.region}#/stacks"
            )

        except ClientError as e:
            error_msg = f"Failed to delete stack {stack_name}: {e}"
            logger.error(error_msg)
            raise NotificationManagerInfraError(error_msg)
