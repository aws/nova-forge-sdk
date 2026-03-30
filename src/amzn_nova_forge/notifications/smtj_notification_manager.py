# Copyright 2025 Amazon Inc

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
from pathlib import Path
from typing import List, Optional

from amzn_nova_forge.model.model_enums import Platform
from amzn_nova_forge.notifications.notification_manager import NotificationManager


class SMTJNotificationManager(NotificationManager):
    """
    Notification manager for SMTJ (SageMaker Training Jobs).
    Supports optional KMS encryption for SNS topic.
    """

    def __init__(self, region: str = "us-east-1"):
        """
        Initialize SMTJ notification manager.

        Args:
            region: AWS region for the infrastructure
        """
        super().__init__(platform=Platform.SMTJ, region=region)

    def get_stack_name(self) -> str:
        """Get the CloudFormation stack name for SMTJ."""
        return "NovaForgeSDK-SMTJ-JobNotifications"

    def get_template_path(self) -> Path:
        """Get the path to the SMTJ CloudFormation template."""
        return Path(__file__).parent / "templates" / "smtj_notification_cf_stack.yaml"

    def _get_stack_parameters(
        self, kms_key_id: Optional[str] = None, **kwargs
    ) -> List[dict]:
        """
        Get SMTJ-specific CloudFormation parameters.

        Args:
            kms_key_id: Optional KMS key ARN for SNS topic encryption

        Returns:
            List of CloudFormation parameter dicts
        """
        parameters = []

        if kms_key_id:
            parameters.append(
                {"ParameterKey": "KmsKeyId", "ParameterValue": kms_key_id}
            )

        return parameters

    def enable_notifications(
        self,
        job_name: str,
        emails: List[str],
        output_s3_path: str,
        namespace: Optional[str] = None,  # Not used for SMTJ.
        kms_key_id: Optional[str] = None,
        **platform_kwargs,
    ) -> None:
        """
        Enable SMTJ job notifications.

        Args:
            job_name: SMTJ job name/ID
            emails: List of email addresses to notify
            output_s3_path: S3 path where job outputs are stored
            namespace: Not used for SMTJ (included for signature compatibility)
            kms_key_id: Optional KMS key ID for SNS topic encryption

        Raises:
            ValueError: If inputs are invalid
            NotificationManagerInfraError: If infrastructure setup fails
        """
        super().enable_notifications(
            job_name=job_name,
            emails=emails,
            output_s3_path=output_s3_path,
            namespace=namespace,
            kms_key_id=kms_key_id,
        )
