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
from typing import Any, Dict, List, Optional

import boto3

from amzn_nova_forge.model.model_enums import Platform
from amzn_nova_forge.notifications.notification_manager import (
    NotificationManager,
    NotificationManagerInfraError,
)
from amzn_nova_forge.util.logging import logger


class SMHPNotificationManager(NotificationManager):
    """
    Notification manager for SMHP (SageMaker HyperPod).

    Supports EKS/VPC configuration for Lambda function that monitors job status.
    Creates one CloudFormation stack per HyperPod cluster.
    """

    # IAM role name has the most restrictive requirement of max 64 characters.
    # So, the max cluster name length = 64 - 34 = 30 chars
    MAX_CLUSTER_NAME_LENGTH = 30

    def __init__(self, cluster_name: str, region: str = "us-east-1"):
        """
        Initialize SMHP notification manager.

        Args:
            cluster_name: HyperPod cluster name (used for stack naming)
            region: AWS region for the infrastructure

        Raises:
            ValueError: If cluster name exceeds maximum length
        """
        # Verify that the HP cluster name is short enough to fit naming restrictions
        if len(cluster_name) > self.MAX_CLUSTER_NAME_LENGTH:
            raise ValueError(
                f"Cluster name '{cluster_name}' is too long ({len(cluster_name)} characters). "
                f"Maximum length is {self.MAX_CLUSTER_NAME_LENGTH} characters."
            )
        super().__init__(platform=Platform.SMHP, region=region)
        self.cluster_name = cluster_name
        self._sagemaker_client = boto3.client("sagemaker", region_name=region)
        self._ec2_client = boto3.client("ec2", region_name=region)

    def _get_cluster_info(self) -> Dict[str, Any]:
        """
        Get HyperPod cluster information from SageMaker API.

        Returns:
            Dict containing cluster configuration including EKS ARN, VPC, subnets, etc.

        Raises:
            NotificationManagerInfraError: If cluster info cannot be retrieved
        """
        try:
            response = self._sagemaker_client.describe_cluster(
                ClusterName=self.cluster_name
            )
            return response
        except Exception as e:
            raise NotificationManagerInfraError(
                f"Failed to get cluster info for {self.cluster_name}: {str(e)}"
            )

    def _get_vpc_id_from_subnet(self, subnet_id: str) -> Optional[str]:
        """
        Get VPC ID from a subnet ID using EC2 API.

        Args:
            subnet_id: Subnet ID to query

        Returns:
            VPC ID or None if not found
        """
        try:
            response = self._ec2_client.describe_subnets(SubnetIds=[subnet_id])
            if response.get("Subnets"):
                return response["Subnets"][0].get("VpcId")
            return None
        except Exception as e:
            logger.warning(f"Could not get VPC ID from subnet {subnet_id}: {e}")
            return None

    def _check_existing_vpc_endpoints(self, vpc_id: str) -> Dict[str, bool]:
        """
        Check if VPC endpoints already exist for DynamoDB and S3 in the VPC.

        Args:
            vpc_id: VPC ID to check

        Returns:
            Dict with keys 'dynamodb' and 's3' indicating if endpoints exist
        """
        existing = {"dynamodb": False, "s3": False}

        try:
            response = self._ec2_client.describe_vpc_endpoints(
                Filters=[
                    {"Name": "vpc-id", "Values": [vpc_id]},
                    {"Name": "vpc-endpoint-type", "Values": ["Gateway"]},
                ]
            )

            for endpoint in response.get("VpcEndpoints", []):
                service_name = endpoint.get("ServiceName", "")
                if "dynamodb" in service_name:
                    existing["dynamodb"] = True
                    logger.info(
                        f"Found existing DynamoDB VPC endpoint: {endpoint.get('VpcEndpointId')}"
                    )
                elif "s3" in service_name:
                    existing["s3"] = True
                    logger.info(
                        f"Found existing S3 VPC endpoint: {endpoint.get('VpcEndpointId')}"
                    )

            return existing

        except Exception as e:
            logger.warning(f"Could not check for existing VPC endpoints: {e}")
            return existing

    def _get_route_table_ids_from_subnets(self, subnet_ids: List[str]) -> List[str]:
        """
        Get route table IDs associated with the given subnets.

        Args:
            subnet_ids: List of subnet IDs to query

        Returns:
            List of unique route table IDs
        """
        try:
            response = self._ec2_client.describe_route_tables(
                Filters=[
                    {
                        "Name": "association.subnet-id",
                        "Values": subnet_ids,
                    }
                ]
            )

            route_table_ids = []
            for rt in response.get("RouteTables", []):
                rt_id = rt.get("RouteTableId")
                if rt_id and rt_id not in route_table_ids:
                    route_table_ids.append(rt_id)

            # If no explicit associations found, check for main route table
            if not route_table_ids:
                # Get VPC ID from first subnet
                vpc_id = self._get_vpc_id_from_subnet(subnet_ids[0])
                if vpc_id:
                    response = self._ec2_client.describe_route_tables(
                        Filters=[
                            {"Name": "vpc-id", "Values": [vpc_id]},
                            {"Name": "association.main", "Values": ["true"]},
                        ]
                    )
                    for rt in response.get("RouteTables", []):
                        rt_id = rt.get("RouteTableId")
                        if rt_id:
                            route_table_ids.append(rt_id)

            return route_table_ids
        except Exception as e:
            logger.warning(f"Could not get route table IDs from subnets: {e}")
            return []

    def get_stack_name(self) -> str:
        """Get the CloudFormation stack name for SMHP (includes cluster name)."""
        return f"NovaForgeSDK-SMHP-JobNotifications-{self.cluster_name}"

    def get_template_path(self) -> Path:
        """Get the path to the SMHP CloudFormation template."""
        return Path(__file__).parent / "templates" / "smhp_notification_cf_stack.yaml"

    def _get_stack_parameters(self, **kwargs: Any) -> List[Dict[Any, Any]]:
        """
        Get SMHP-specific CloudFormation parameters.

        Args:
            **kwargs: Platform-specific parameters including:
                - eks_cluster_arn: EKS cluster ARN for the HyperPod cluster
                - vpc_id: VPC ID where the HyperPod cluster is running
                - subnet_ids: List of subnet IDs for Lambda function (private subnets with NAT)
                - security_group_id: Security group ID for Lambda function
                - route_table_ids: List of route table IDs for Gateway VPC endpoints (auto-detected if not provided)
                - create_dynamodb_endpoint: Whether to create DynamoDB endpoint (default: True)
                - create_s3_endpoint: Whether to create S3 endpoint (default: True)
                - kubectl_layer_arn: ARN of lambda-kubectl layer
                - polling_interval_minutes: How often to check job status in minutes (default: 5)
                - kms_key_id: Optional KMS key ID for SNS topic encryption

        Returns:
            List of CloudFormation parameter dicts
        """
        eks_cluster_arn: str = kwargs.get("eks_cluster_arn", "")
        vpc_id: str = kwargs.get("vpc_id", "")
        subnet_ids: List[str] = kwargs.get("subnet_ids", [])
        security_group_id: str = kwargs.get("security_group_id", "")
        route_table_ids: List[str] = kwargs.get("route_table_ids", [])
        create_dynamodb_endpoint: bool = kwargs.get("create_dynamodb_endpoint", True)
        create_s3_endpoint: bool = kwargs.get("create_s3_endpoint", True)
        kubectl_layer_arn: str = kwargs.get("kubectl_layer_arn", "")
        polling_interval_minutes: int = kwargs.get("polling_interval_minutes", 5)
        kms_key_id: Optional[str] = kwargs.get("kms_key_id")

        parameters: List[Dict[Any, Any]] = [
            {"ParameterKey": "ClusterName", "ParameterValue": self.cluster_name},
            {"ParameterKey": "EksClusterArn", "ParameterValue": eks_cluster_arn},
            {"ParameterKey": "VpcId", "ParameterValue": vpc_id},
            {
                "ParameterKey": "SubnetIds",
                "ParameterValue": ",".join(subnet_ids),
            },
            {"ParameterKey": "SecurityGroupId", "ParameterValue": security_group_id},
            {
                "ParameterKey": "RouteTableIds",
                "ParameterValue": ",".join(route_table_ids),
            },
            {
                "ParameterKey": "CreateDynamoDBEndpoint",
                "ParameterValue": str(create_dynamodb_endpoint),
            },
            {
                "ParameterKey": "CreateS3Endpoint",
                "ParameterValue": str(create_s3_endpoint),
            },
            {"ParameterKey": "KubectlLayerArn", "ParameterValue": kubectl_layer_arn},
            {
                "ParameterKey": "PollingIntervalMinutes",
                "ParameterValue": str(polling_interval_minutes),
            },
        ]

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
        namespace: Optional[str] = None,
        **platform_kwargs: Any,
    ) -> None:
        """
        Enable SMHP job notifications.

        Args:
            job_name: SMHP job name/ID (PyTorchJob name)
            emails: List of email addresses to notify
            output_s3_path: S3 path where job outputs are stored
            namespace: Kubernetes namespace where the job runs (e.g., 'kubeflow', 'default') - REQUIRED for SMHP
            **platform_kwargs: Platform-specific parameters (all optional if cluster is accessible):
                - eks_cluster_arn: EKS cluster ARN (auto-detected if not provided)
                - vpc_id: VPC ID (auto-detected if not provided)
                - subnet_ids: List of subnet IDs for Lambda (auto-detected if not provided)
                - security_group_id: Security group ID for Lambda (auto-detected if not provided)
                - kubectl_layer_arn: ARN of lambda-kubectl layer (required)
                - kms_key_id: Optional KMS key ID for SNS topic encryption

        Raises:
            ValueError: If inputs are invalid
            NotificationManagerInfraError: If infrastructure setup fails or cluster info cannot be retrieved
        """
        # Validate namespace (required for SMHP)
        if not namespace:
            raise ValueError(
                "namespace is required for SMHP notifications. "
                "Specify the Kubernetes namespace where your PyTorchJob runs (e.g., 'kubeflow', 'default')."
            )

        # Try to auto-detect cluster configuration if not provided
        eks_cluster_arn = platform_kwargs.get("eks_cluster_arn")
        vpc_id = platform_kwargs.get("vpc_id")
        subnet_ids = platform_kwargs.get("subnet_ids")
        security_group_id = platform_kwargs.get("security_group_id")
        route_table_ids = platform_kwargs.get("route_table_ids")

        # If any required parameter is missing, try to auto-detect from cluster
        if not all([eks_cluster_arn, vpc_id, subnet_ids, security_group_id]):
            logger.info(
                f"Auto-detecting cluster configuration for {self.cluster_name}..."
            )
            try:
                cluster_info = self._get_cluster_info()

                # Get EKS cluster ARN from Orchestrator config
                if not eks_cluster_arn:
                    orchestrator = cluster_info.get("Orchestrator", {})
                    eks_config = orchestrator.get("Eks", {})
                    eks_cluster_arn = eks_config.get("ClusterArn")
                    if eks_cluster_arn:
                        logger.info(f"Auto-detected EKS cluster ARN: {eks_cluster_arn}")
                        platform_kwargs["eks_cluster_arn"] = eks_cluster_arn

                # Get VPC configuration
                vpc_config = cluster_info.get("VpcConfig", {})

                if not subnet_ids:
                    subnet_ids = vpc_config.get("Subnets", [])
                    if subnet_ids:
                        logger.info(f"Auto-detected {len(subnet_ids)} subnets")
                        platform_kwargs["subnet_ids"] = subnet_ids

                # Get VPC ID from subnet (not directly in cluster response)
                if not vpc_id and subnet_ids:
                    vpc_id = self._get_vpc_id_from_subnet(subnet_ids[0])
                    if vpc_id:
                        logger.info(f"Auto-detected VPC ID from subnet: {vpc_id}")
                        platform_kwargs["vpc_id"] = vpc_id

                if not security_group_id:
                    security_groups = vpc_config.get("SecurityGroupIds", [])
                    if security_groups:
                        # Use the first security group
                        security_group_id = security_groups[0]
                        logger.info(
                            f"Auto-detected security group: {security_group_id}"
                        )
                        platform_kwargs["security_group_id"] = security_group_id

            except Exception as e:
                logger.warning(
                    f"Could not auto-detect cluster configuration: {e}. "
                    "You may need to provide eks_cluster_arn, vpc_id, subnet_ids, and security_group_id explicitly."
                )

        # Auto-detect route table IDs if not provided and we have subnet_ids
        if not route_table_ids and platform_kwargs.get("subnet_ids"):
            logger.info("Auto-detecting route table IDs from subnets...")
            route_table_ids = self._get_route_table_ids_from_subnets(
                platform_kwargs["subnet_ids"]
            )
            if route_table_ids:
                logger.info(
                    f"Auto-detected {len(route_table_ids)} route table(s): {', '.join(route_table_ids)}"
                )
                platform_kwargs["route_table_ids"] = route_table_ids
            else:
                logger.warning(
                    "Could not auto-detect route table IDs. Gateway VPC endpoints may not work correctly."
                )

        # Check for existing Gateway VPC endpoints
        if platform_kwargs.get("vpc_id"):
            logger.info("Checking for existing Gateway VPC endpoints...")
            existing_endpoints = self._check_existing_vpc_endpoints(
                platform_kwargs["vpc_id"]
            )

            if existing_endpoints["dynamodb"]:
                logger.info(
                    "DynamoDB Gateway endpoint already exists, will skip creation"
                )
                platform_kwargs["create_dynamodb_endpoint"] = False
            else:
                platform_kwargs["create_dynamodb_endpoint"] = True

            if existing_endpoints["s3"]:
                logger.info("S3 Gateway endpoint already exists, will skip creation")
                platform_kwargs["create_s3_endpoint"] = False
            else:
                platform_kwargs["create_s3_endpoint"] = True

        # Validate required parameters (after auto-detection attempt)
        if not platform_kwargs.get("eks_cluster_arn"):
            raise ValueError(
                "eks_cluster_arn is required for SMHP notifications. "
                "Either provide it explicitly or ensure the cluster is accessible for auto-detection."
            )
        if not platform_kwargs.get("vpc_id"):
            raise ValueError(
                "vpc_id is required for SMHP notifications. "
                "Either provide it explicitly or ensure the cluster is accessible for auto-detection."
            )
        if not platform_kwargs.get("subnet_ids") or not isinstance(
            platform_kwargs.get("subnet_ids"), list
        ):
            raise ValueError(
                "subnet_ids must be a non-empty list. "
                "Either provide it explicitly or ensure the cluster is accessible for auto-detection."
            )
        if not platform_kwargs.get("security_group_id"):
            raise ValueError(
                "security_group_id is required for SMHP notifications. "
                "Either provide it explicitly or ensure the cluster is accessible for auto-detection."
            )
        if not platform_kwargs.get("kubectl_layer_arn"):
            raise ValueError(
                "kubectl_layer_arn is required for SMHP notifications. "
                "Please create a layer here: https://serverlessrepo.aws.amazon.com/applications/arn:aws:serverlessrepo:us-east-1:903779448426:applications~lambda-layer-kubectl and provide the ARN."
            )

        super().enable_notifications(
            job_name=job_name,
            emails=emails,
            output_s3_path=output_s3_path,
            namespace=namespace,
            **platform_kwargs,
        )
