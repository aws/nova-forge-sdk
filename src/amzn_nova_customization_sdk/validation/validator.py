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
import builtins
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import boto3

from amzn_nova_customization_sdk.manager.runtime_manager import (
    RuntimeManager,
    SMHPRuntimeManager,
    SMTJRuntimeManager,
)
from amzn_nova_customization_sdk.model.model_enums import (
    Platform,
    TrainingMethod,
)
from amzn_nova_customization_sdk.recipe.recipe_config import (
    BYOD_AVAILABLE_EVAL_TASKS,
    EVAL_TASK_STRATEGY_MAP,
    EvaluationTask,
    get_available_subtasks,
)
from amzn_nova_customization_sdk.util.logging import logger
from amzn_nova_customization_sdk.util.sagemaker import get_cluster_instance_info

LAMBDA_ARN_REGEX = re.compile(
    r"^arn:aws:lambda:[a-z0-9-]+:\d{12}:function:[A-Za-z0-9-_]+$"
)

# ECR image URI pattern: account.dkr.ecr.region.amazonaws.com/repository:tag
ECR_IMAGE_URI_REGEX = re.compile(
    r"^\d{12}\.dkr\.ecr\.[a-z0-9-]+\.amazonaws\.com/[a-zA-Z0-9][a-zA-Z0-9._/-]*:[a-zA-Z0-9._-]+$"
)

# https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_TrainingJob.html
JOB_NAME_REGEX = re.compile(r"^[a-zA-Z0-9\-]{1,63}$")

# https://kubernetes.io/docs/concepts/overview/working-with-objects/namespaces/
NAMESPACE_REGEX = re.compile(r"^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$")

# https://docs.aws.amazon.com/eks/latest/APIReference/API_CreateCluster.html#API_CreateCluster_RequestParameters
CLUSTER_NAME_REGEX = re.compile(r"^[0-9A-Za-z][A-Za-z0-9\-_]{1,100}$")

TYPE_ALIASES = {
    "string": "str",
    "str": "str",
    "integer": "int",
    "int": "int",
    "boolean": "bool",
    "bool": "bool",
    "float": "float",
}


class Validator:
    """
    Validator class providing validation functionality.
    """

    @staticmethod
    def _get_default_validation_config() -> Dict[str, bool]:
        """Get default validation configuration."""
        return {"iam": True, "infra": True}.copy()

    @staticmethod
    def _resolve_execution_role(infra: Optional[RuntimeManager]) -> str:
        """Resolve the role used to execute the job being validated."""
        execution_role = None

        if (
            infra
            and hasattr(infra, "execution_role")
            and getattr(infra, "execution_role", None)
        ):
            execution_role = infra.execution_role
        else:
            raise ValueError(
                f"RuntimeManager {infra} is invalid or does not use execution roles!"
            )

        return execution_role

    @staticmethod
    def _is_cross_account_role(execution_role_arn: str) -> bool:
        """Check if execution role is in a different account."""
        try:
            # Extract account from role ARN: arn:aws:iam::ACCOUNT:role/RoleName
            role_account = execution_role_arn.split(":")[4]

            # Get current account
            sts_client = boto3.client("sts")
            current_account = sts_client.get_caller_identity()["Account"]

            return role_account != current_account
        except Exception:
            return False  # Assume same-account if parsing fails

    @staticmethod
    def _access_cross_account_role(
        execution_role_arn: str, region_name: str
    ) -> Tuple[Optional[Any], Optional[Dict], bool]:
        """
        Retrieve validation data from a cross-account role.

        Returns:
            Tuple of (iam_client, trust_policy, can_read_policies)
            - iam_client: IAM client with assumed role credentials, or None on failure
            - trust_policy: Role trust policy document, or None on failure
            - can_read_policies: True if we can read the role's policies, False otherwise
        """
        try:
            # Assume the cross-account role
            sts_client = boto3.client("sts")
            assumed_role = sts_client.assume_role(
                RoleArn=execution_role_arn, RoleSessionName="SageMakerValidation"
            )

            # Create IAM client with assumed role credentials
            credentials = assumed_role["Credentials"]
            iam_client = boto3.client(
                "iam",
                region_name=region_name,
                aws_access_key_id=credentials["AccessKeyId"],
                aws_secret_access_key=credentials["SecretAccessKey"],
                aws_session_token=credentials["SessionToken"],
            )

            # Get role and trust policy using assumed credentials
            role_name = execution_role_arn.split("/")[-1]
            role_response = iam_client.get_role(RoleName=role_name)
            trust_policy = role_response["Role"]["AssumeRolePolicyDocument"]

            # Test if we can read policies by attempting the policy operations
            can_read_policies = True
            try:
                # Test inline policies access
                iam_client.list_role_policies(RoleName=role_name)

                # Test managed policies access
                attached_policies = iam_client.list_attached_role_policies(
                    RoleName=role_name
                )

                # If there are managed policies, test if we can read them
                if attached_policies["AttachedPolicies"]:
                    first_policy = attached_policies["AttachedPolicies"][0]
                    iam_client.get_policy(PolicyArn=first_policy["PolicyArn"])

            except Exception:
                can_read_policies = False

            return iam_client, trust_policy, can_read_policies

        except Exception:
            return None, None, False

    @staticmethod
    def _check_policy_json_permissions(
        policies: List[Dict], required_permissions: List[str]
    ) -> List[str]:
        """
        Check if required permissions are granted in policy JSON documents.

        Args:
            policies: List of IAM policy documents
            required_permissions: List of required permissions to check

        Returns:
            List of missing permissions
        """
        missing_permissions = []
        for permission in required_permissions:
            service, action = permission.split(":", 1)
            found = False

            for policy in policies:
                for statement in policy.get("Statement", []):
                    if statement.get("Effect") == "Allow":
                        actions = statement.get("Action", [])
                        if isinstance(actions, str):
                            actions = [actions]

                        # Check for exact match or wildcard
                        for action in actions:
                            if (
                                permission == action
                                or action == f"{service}:*"
                                or action == "*"
                                or Validator._matches_iam_wildcard_pattern(
                                    action, permission
                                )
                            ):
                                found = True
                                break
                        if found:
                            break
                if found:
                    break

            if not found:
                missing_permissions.append(permission)

        return missing_permissions

    @staticmethod
    def _matches_iam_wildcard_pattern(pattern: str, permission: str) -> bool:
        """
        Check if an API permission matches an IAM policy pattern.
        Supports both exact matches and `*` wildcards
        """
        if "*" not in pattern:
            return False

        # Convert wildcard pattern to regex
        import re

        # Escape special regex characters except *
        escaped = re.escape(pattern).replace(r"\*", ".*")

        # Ensure full match
        regex_pattern = f"^{escaped}$"

        return bool(re.match(regex_pattern, permission))

    @staticmethod
    def _validate_calling_role_permissions(
        errors: List[str],
        required_permissions: Union[
            List[str],
            List[Tuple[str, str]],
            List[Tuple[str, Callable[[RuntimeManager], str]]],
            List[
                Union[str, Tuple[str, str], Tuple[str, Callable[[RuntimeManager], str]]]
            ],
        ],
        infra: Optional[RuntimeManager] = None,
        region_name: str = "us-east-1",
    ):
        """
        Validate that the current calling role has the required permissions.

        Args:
            errors: List to append validation errors to
            required_permissions: List of required IAM permissions in format:
                - str: API permission to check via JSON parsing
                - (api_string, resource_string): Use SimulatePrincipalPolicy with resource string
                - (api_string, resource_lambda): Use SimulatePrincipalPolicy with lambda result
            infra: Infrastructure manager for resource lambda evaluation
            region_name: AWS region name for clients
        """
        try:
            iam_client = boto3.client("iam", region_name=region_name)
            sts_client = boto3.client("sts", region_name=region_name)

            # Get current caller identity
            caller_identity = sts_client.get_caller_identity()
            caller_arn = caller_identity["Arn"]

            # Convert assumed role ARN to role ARN for simulation
            if ":assumed-role/" in caller_arn:
                # Extract role name from assumed role ARN
                role_name = caller_arn.split("/")[1]
                account_id = caller_identity["Account"]
                role_arn = f"arn:aws:iam::{account_id}:role/{role_name}"
            else:
                role_arn = caller_arn

            # Test ability to run IAM simulation at all
            try:
                iam_client.simulate_principal_policy(
                    PolicySourceArn=role_arn,
                    ActionNames=[],
                    ResourceArns=["*"],
                )
            except Exception as e:
                errors.append(
                    f"Cannot run iam:SimulatePrincipalPolicy to validate calling role permissions: {e}"
                )
                return

            # Process each required permission based on its format
            for permission in required_permissions:
                try:
                    if isinstance(permission, str):
                        # Simple API permission - use JSON parsing helper
                        Validator._validate_permission_via_json_parsing(
                            errors, permission, role_arn, iam_client
                        )
                    elif isinstance(permission, tuple) and len(permission) == 2:
                        api_string, resource_spec = permission

                        if isinstance(resource_spec, str):
                            # (api_string, resource_string) - use SimulatePrincipalPolicy
                            Validator._validate_permission_via_simulation(
                                errors, api_string, resource_spec, role_arn, iam_client
                            )
                        elif callable(resource_spec):
                            # (api_string, resource_lambda) - call lambda with infra
                            if infra is None:
                                errors.append(
                                    f"Cannot evaluate resource lambda for {api_string}: infra is None"
                                )
                                continue

                            try:
                                resource_arn = resource_spec(infra)
                                Validator._validate_permission_via_simulation(
                                    errors,
                                    api_string,
                                    resource_arn,
                                    role_arn,
                                    iam_client,
                                )
                            except Exception as e:
                                errors.append(
                                    f"Failed to evaluate resource lambda for {api_string}: {e}"
                                )
                        else:
                            errors.append(f"Invalid permission format: {permission}")
                    else:
                        errors.append(f"Invalid permission format: {permission}")

                except Exception as e:
                    errors.append(f"Could not verify permission {permission}: {str(e)}")

        except Exception as e:
            errors.append(f"Failed to validate calling role permissions: {str(e)}")

    @staticmethod
    def _validate_permission_via_simulation(
        errors: List[str], api_string: str, resource_arn: str, role_arn: str, iam_client
    ):
        """Validate permission using IAM SimulatePrincipalPolicy."""
        response = iam_client.simulate_principal_policy(
            PolicySourceArn=role_arn,
            ActionNames=[api_string],
            ResourceArns=[resource_arn],
        )

        if response["EvaluationResults"]:
            result = response["EvaluationResults"][0]
            if result["EvalDecision"] != "allowed":
                errors.append(
                    f"Missing required calling role permission: {api_string} on {resource_arn}"
                )

    @staticmethod
    def _validate_permission_via_json_parsing(
        errors: List[str], api_string: str, role_arn: str, iam_client
    ):
        """Validate permission using JSON policy parsing."""
        try:
            role_name = role_arn.split("/")[-1]

            # Get all policies attached to the role
            policies = []

            # Get inline policies
            inline_policies = iam_client.list_role_policies(RoleName=role_name)
            for policy_name in inline_policies["PolicyNames"]:
                policy_doc = iam_client.get_role_policy(
                    RoleName=role_name, PolicyName=policy_name
                )
                policies.append(policy_doc["PolicyDocument"])

            # Get attached managed policies
            attached_policies = iam_client.list_attached_role_policies(
                RoleName=role_name
            )
            for policy in attached_policies["AttachedPolicies"]:
                policy_version = iam_client.get_policy(PolicyArn=policy["PolicyArn"])
                policy_doc = iam_client.get_policy_version(
                    PolicyArn=policy["PolicyArn"],
                    VersionId=policy_version["Policy"]["DefaultVersionId"],
                )
                policies.append(policy_doc["PolicyVersion"]["Document"])

            # Check if permission is granted using the refactored helper
            missing_permissions = Validator._check_policy_json_permissions(
                policies, [api_string]
            )

            if missing_permissions:
                errors.append(f"Missing required calling role permission: {api_string}")

        except Exception as e:
            errors.append(
                f"Failed to validate permission {api_string} via JSON parsing: {e}"
            )

    @staticmethod
    def _validate_iam_permissions(
        errors: List[str],
        infra: Optional[RuntimeManager] = None,
        data_s3_path: Optional[str] = None,
        output_s3_path: Optional[str] = None,
    ) -> None:
        """
        Validate required IAM permissions for training jobs.

        Args:
            errors: List to append validation errors to
            infra: Optional infrastructure manager to check for overridden execution role
            data_s3_path: Optional S3 path for training data
            output_s3_path: Optional S3 path for output
        """
        if isinstance(infra, SMHPRuntimeManager):
            # SMHP validations - validate calling role permissions
            try:
                region_name = getattr(infra, "region", "us-east-1")

                # Required permissions for HyperPod operations
                required_calling_role_permissions = type(
                    infra
                ).required_calling_role_permissions(data_s3_path, output_s3_path)

                # Validate calling role permissions
                Validator._validate_calling_role_permissions(
                    errors, required_calling_role_permissions, infra, region_name
                )

                # Additional cluster-specific validation
                cluster_name = getattr(infra, "cluster_name", None)
                if cluster_name:
                    try:
                        sagemaker_client = boto3.client(
                            "sagemaker", region_name=region_name
                        )
                        sagemaker_client.describe_cluster(ClusterName=cluster_name)
                    except Exception as e:
                        if "ResourceNotFound" in str(e):
                            errors.append(
                                f"HyperPod cluster '{cluster_name}' not found"
                            )
                        elif "AccessDenied" in str(e):
                            errors.append(
                                f"Access denied when checking cluster '{cluster_name}' - verify sagemaker:DescribeCluster and eks:DescribeCluster permissions"
                            )

            except Exception as e:
                errors.append(f"Failed to validate SMHP IAM permissions: {str(e)}")
            return

        elif isinstance(infra, SMTJRuntimeManager):
            # SMTJ validations
            try:
                region_name = getattr(infra, "region", "us-east-1")

                # Required permissions for SMTJ calling role operations,
                # as well as execution role validation
                required_calling_role_permissions = type(
                    infra
                ).required_calling_role_permissions(data_s3_path, output_s3_path)

                # Validate calling role permissions
                Validator._validate_calling_role_permissions(
                    errors, required_calling_role_permissions, infra, region_name
                )

            except Exception as e:
                errors.append(
                    f"Failed to validate SMTJ calling role permissions: {str(e)}\n"
                    "Note that this may result in failure to validate the execution role as well."
                )

            # Validate SageMaker execution role (which only applies to SMTJ)
            execution_role = None
            try:
                try:
                    execution_role = Validator._resolve_execution_role(infra)
                except Exception as e:
                    errors.append(
                        f"Could not resolve intended execution role for job: {str(e)}"
                    )
                    return

                region_name = getattr(infra, "region", "us-east-1")

                # Extract role name from ARN, handle non-string execution roles
                if not isinstance(execution_role, str):
                    errors.append(
                        f"Invalid execution role format: {type(execution_role).__name__}"
                    )
                    return

                role_name = execution_role.split("/")[-1]

                is_cross_account_role = Validator._is_cross_account_role(execution_role)

                # Check if this is a cross-account role
                if is_cross_account_role:
                    # Attempt cross-account data collection
                    iam_client, trust_policy, can_read_policies = (
                        Validator._access_cross_account_role(
                            execution_role, region_name
                        )
                    )
                    if iam_client is None:
                        # Cross-account data collection failed, skip validation logic
                        logger.info(
                            f"Could not access cross-account execution role {execution_role} for validation, will assume correctness."
                        )
                        return
                else:
                    # Same-account role validation
                    can_read_policies = (
                        True  # Assume we can read policies in same account
                    )
                    try:
                        iam_client = boto3.client("iam", region_name=region_name)
                        role_response = iam_client.get_role(RoleName=role_name)
                        trust_policy = role_response["Role"]["AssumeRolePolicyDocument"]
                    except Exception as e:
                        if "NoSuchEntity" in str(e):
                            errors.append(
                                f"SageMaker execution role {role_name} does not exist"
                            )
                        elif "AccessDenied" in str(e):
                            errors.append(
                                "Missing IAM permissions in current role: iam:GetRole required to validate execution role"
                            )
                        else:
                            errors.append(
                                f"Failed to retrieve execution role from IAM: {str(e)}"
                            )
                        return

                    if trust_policy is None:
                        errors.append(
                            f"Failed to parse trust policy from IAM role. GetRole output: {role_response}"
                        )

                # Check if SageMaker service can assume this role
                sagemaker_trusted = False
                try:
                    if trust_policy:
                        for statement in trust_policy.get("Statement", []):
                            if statement.get("Effect") == "Allow":
                                principal = statement.get("Principal", {})
                                if isinstance(principal, dict):
                                    service = principal.get("Service", [])
                                    if isinstance(service, str):
                                        service = [service]

                                    if "sagemaker.amazonaws.com" in service:
                                        sagemaker_trusted = True
                                        break
                    # For cross-account roles, if we can't check trust policy assume it's valid
                    elif is_cross_account_role:
                        sagemaker_trusted = True

                    if not sagemaker_trusted:
                        errors.append(
                            f"SageMaker execution role {role_name} does not trust sagemaker.amazonaws.com service"
                        )
                except Exception as e:
                    errors.append(f"Failed to parse trust policy: {str(e)}")

                # Check required permissions for CreateTrainingJob (only if we can read policies)
                if can_read_policies:
                    # Taken from https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html#sagemaker-roles-createtrainingjob-perms
                    # NOTE: Removed the CloudWatch permissions as likely not availability impacting
                    required_execution_role_permissions = [
                        "s3:GetObject",
                        "s3:PutObject",
                        "s3:ListBucket",
                    ]

                    try:
                        # Get all policies attached to the role
                        policies = []

                        # Get inline policies
                        inline_policies = iam_client.list_role_policies(
                            RoleName=role_name
                        )
                        for policy_name in inline_policies["PolicyNames"]:
                            policy_doc = iam_client.get_role_policy(
                                RoleName=role_name, PolicyName=policy_name
                            )
                            policies.append(policy_doc["PolicyDocument"])

                        # Get attached managed policies
                        attached_policies = iam_client.list_attached_role_policies(
                            RoleName=role_name
                        )
                        for policy in attached_policies["AttachedPolicies"]:
                            policy_version = iam_client.get_policy(
                                PolicyArn=policy["PolicyArn"]
                            )
                            policy_doc = iam_client.get_policy_version(
                                PolicyArn=policy["PolicyArn"],
                                VersionId=policy_version["Policy"]["DefaultVersionId"],
                            )
                            policies.append(policy_doc["PolicyVersion"]["Document"])

                        # Check if required permissions are granted using refactored helper
                        missing_permissions = Validator._check_policy_json_permissions(
                            policies, required_execution_role_permissions
                        )

                        if missing_permissions:
                            errors.append(
                                f"Execution role missing required permissions: {', '.join(missing_permissions)}"
                            )
                    except Exception as e:
                        if "AccessDenied" in str(e):
                            errors.append(
                                f"Missing IAM permissions to validate execution role permissions: {str(e)}"
                            )
                        else:
                            errors.append(
                                f"Failed to validate execution role permissions: {str(e)}"
                            )
            except Exception as e:
                # For cross-account roles, silently skip IAM validation on unexpected errors
                if execution_role and Validator._is_cross_account_role(execution_role):
                    return

                if not execution_role:
                    errors.append(f"Unknown issue resolving execution role: {str(e)}")
                elif "Could not find credentials" in str(e):
                    errors.append("AWS credentials not configured")
                elif "Unable to locate credentials" in str(e):
                    errors.append("AWS credentials not found")
                else:
                    errors.append(f"Failed to validate execution role: {str(e)}")

    @staticmethod
    def _validate_infrastructure(infra: Any, errors: List[str]) -> None:
        """
        Validate SMHP infrastructure requirements.

        Args:
            infra: SMHPRuntimeManager instance
            errors: List to append validation errors to
        """
        try:
            region_name = getattr(infra, "region", "us-east-1")

            cluster_name = getattr(infra, "cluster_name", None)
            if not cluster_name:
                errors.append(
                    "SMHP cluster name not found in infrastructure configuration"
                )
                return

            # Test permission to describe Sagemaker clusters
            sagemaker_client = boto3.client("sagemaker", region_name=region_name)
            try:
                sagemaker_client.describe_cluster(ClusterName=cluster_name)
            except Exception as e:
                if "AccessDenied" in str(e) or "UnauthorizedOperation" in str(e):
                    errors.append(
                        "Missing SageMaker permissions: sagemaker:DescribeCluster required"
                    )
                    return
                else:
                    # Re-raise if it's not a permission error
                    raise e

            # Get cluster instance information by describing the specified cluster
            cluster_info = get_cluster_instance_info(cluster_name, region_name)
            restricted_instance_groups = cluster_info["restricted_instance_groups"]

            # Check if required instance type exists in any restricted instance group
            compatible_groups = []
            for group in restricted_instance_groups:
                if group["instance_type"] == infra.instance_type:
                    compatible_groups.append(group)

            if not compatible_groups:
                available_types = [
                    group["instance_type"] for group in restricted_instance_groups
                ]
                errors.append(
                    f"Instance type '{infra.instance_type}' not available in restricted instance groups in cluster '{cluster_name}'. "
                    f"Available types: {sorted(set(available_types))}"
                )
                return

            # Check if any compatible group has sufficient capacity
            sufficient_capacity = False
            for group in compatible_groups:
                if group["current_count"] >= infra.instance_count:
                    sufficient_capacity = True
                    break

            if not sufficient_capacity:
                max_available = max(
                    group["current_count"] for group in compatible_groups
                )
                errors.append(
                    f"Insufficient capacity for instance type '{infra.instance_type}' in cluster '{cluster_name}'. "
                    f"Required: {infra.instance_count}, Maximum available: {max_available}"
                )

        except Exception as e:
            errors.append(f"Failed to validate cluster infrastructure: {str(e)}")

    @staticmethod
    def validate_data_mixing_config(
        config: dict,
        nova_prefix: str,
        percent_suffix: str,
        customer_data_field: str,
        dataset_catalog_fields: str,
        expected_nova_fields: set,
    ) -> None:
        """
        Validate the data mixing configuration. The validation rules are as follows:
        - The datamix config should have valid fields.
        - Customer data can be between 0-100.
        - If customer data is 100, then no nova data is used
        - If customer data < 100, then sum of nova data percent fields should be 100.

        Raises:
            ValueError: If configuration is invalid
        """

        nova_fields = {}
        customer_percent = 0
        total = 0

        if expected_nova_fields:
            for key in config.keys():
                # Skip customer_data_percent
                if key == dataset_catalog_fields:
                    continue
                # Check if key is a nova field that's not in the known defaults
                if key not in expected_nova_fields:
                    raise ValueError(
                        f"Invalid nova field '{key}'. Valid fields are: {sorted(expected_nova_fields)}"
                    )

        for key, value in config.items():
            if key == customer_data_field:
                customer_percent = value
                if value is not None and not 0 <= value <= 100:
                    raise ValueError(
                        f"{customer_data_field} must be between 0 and 100, got {value}"
                    )
            elif key.startswith(nova_prefix) and key.endswith(percent_suffix):
                nova_fields[key] = value
                # Each nova field must be between 0 and 100
                if value is not None and not 0 <= value <= 100:
                    raise ValueError(f"{key} must be between 0 and 100, got {value}")

        if nova_fields:
            non_none_values = [v for v in nova_fields.values() if v is not None]
            if non_none_values:
                total = sum(non_none_values)
                if abs(total - 100.0) > 0.01:  # Allow small floating point errors
                    raise ValueError(
                        f"Nova data percentages must sum to 100, got {total}. "
                        f"Fields: {nova_fields}"
                    )

        if customer_percent == 100 and total > 0:
            raise ValueError(
                f"Since {customer_data_field} is 100 %, all nova data should sum to 0 %"
            )

        if customer_percent < 100 and total == 0:
            raise ValueError(
                f"Since {customer_data_field} is less than 100 % {customer_percent}%, all nova data cannot be 0"
                f"Fields: {nova_fields} should sum to 100 %"
            )

    @staticmethod
    def _validate_recipe(
        recipe: Dict[str, Any],
        overrides_template: Dict[str, Any],
        instance_type: str,
        errors: List[str],
        method: TrainingMethod,
        rft_lambda_arn: Optional[str] = None,
        eval_task: Optional[EvaluationTask] = None,
        data_s3_path: Optional[str] = None,
        subtask: Optional[str] = None,
        processor_config: Optional[Dict[str, Any]] = None,
        rl_env_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Validate the generated recipe against constraints stored in Jump Start's S3 bucket

        Args:
            recipe: The recipe dict to validate
            overrides_template: Dict containing recipe constraints
            instance_type: The instance type used for training
            errors: List to append validation errors to
            method: The training method being performed

            rft_lambda_arn: The Lambda ARN to validate for RFT training
            eval_task: Evaluation task
            data_s3_path: Input data s3 Path
            subtask: Evaluation subtask
            processor_config: BYOM processor configuration
            rl_env_config: BYO RFT evaluation configuration
        """

        def validate_rft(rft_lambda_arn: Optional[str] = None):
            """
            Validate RFT-related parameters

            Args:
                rft_lambda_arn: Optional RFT Lambda ARN
            """
            if rft_lambda_arn is None:
                errors.append(
                    "'rft_lambda_arn' is a required parameter when calling train() for RFT"
                )
            elif not LAMBDA_ARN_REGEX.match(rft_lambda_arn):
                errors.append("'rft_lambda_arn' must be a valid Lambda function ARN")

        def validate_eval(
            eval_task: EvaluationTask,
            data_s3_path: Optional[str] = None,
            subtask: Optional[str] = None,
            processor_config: Optional[Dict[str, Any]] = None,
            rl_env_config: Optional[Dict[str, Any]] = None,
        ):
            """
            Validate evaluation-related parameters

            Args:
                eval_task: Evaluation task
                data_s3_path: Optional input data S3 path
                subtask: Optional subtask
                processor_config: Optional processor_config
                rl_env_config: Optional rl_env_config
            """
            # Validate eval task strategy
            if eval_task not in EVAL_TASK_STRATEGY_MAP:
                errors.append(
                    f"Evaluation task '{eval_task.value}' is not currently supported"
                )

            # Validate BYOD task
            if data_s3_path:
                if eval_task.value not in BYOD_AVAILABLE_EVAL_TASKS:
                    errors.append(
                        f"BYOD evaluation must use one of the following eval tasks: {BYOD_AVAILABLE_EVAL_TASKS}. If you wish to use '{eval_task.value}' for eval, remove 'data_s3_path' from your input"
                    )

            # Validate subtask
            if subtask:
                valid_subtasks = get_available_subtasks(eval_task)
                if not valid_subtasks:
                    errors.append(f"Task {eval_task.value} does not support subtasks")
                if subtask not in valid_subtasks:
                    errors.append(
                        f'Invalid subtask "{subtask}" for task {eval_task.value}. Valid subtasks: {valid_subtasks}'
                    )

            # Check processor_config
            if processor_config:
                if eval_task != EvaluationTask.GEN_QA:
                    errors.append(
                        f"processor_config is only supported for gen_qa task, but you provided {eval_task.value}"
                    )
                else:
                    if not processor_config.get("lambda_arn"):
                        errors.append("processor_config must contain a lambda_arn")
                    else:
                        lambda_arn = processor_config.get("lambda_arn")
                        if not isinstance(
                            lambda_arn, str
                        ) or not LAMBDA_ARN_REGEX.match(lambda_arn):
                            errors.append(
                                "'lambda_arn' must be a valid Lambda function ARN"
                            )

            # Check rl_env_config
            if rl_env_config:
                if eval_task != EvaluationTask.RFT_EVAL:
                    errors.append(
                        f"rl_env_config is only supported for rft_eval task, but you provided {eval_task.value}"
                    )
                if not rl_env_config.get("reward_lambda_arn"):
                    errors.append(f"rl_env must contain a reward_lambda_arn")
                else:
                    reward_lambda_arn = rl_env_config.get("reward_lambda_arn")
                    if not isinstance(
                        reward_lambda_arn, str
                    ) or not LAMBDA_ARN_REGEX.match(reward_lambda_arn):
                        errors.append(
                            "'reward_lambda_arn' must be a valid Lambda function ARN"
                        )

        def get_recipe_value(data: Dict[str, Any], key_to_find: str) -> Any:
            """
            Get value from a dict for a given key

            Args:
                data: The dictionary to read from
                key_to_find: The key to search for

            Returns:
                The value from the dict
            """
            for key, value in data.items():
                if key == key_to_find:
                    return value
                if isinstance(value, dict):
                    try:
                        return get_recipe_value(value, key_to_find)
                    except Exception:
                        continue
            raise Exception("Unable to find override key in recipe.")

        if method in [TrainingMethod.RFT_LORA, TrainingMethod.RFT_FULL]:
            validate_rft(rft_lambda_arn=rft_lambda_arn)
        elif method in [TrainingMethod.EVALUATION]:
            assert eval_task is not None
            validate_eval(
                eval_task=eval_task,
                data_s3_path=data_s3_path,
                subtask=subtask,
                processor_config=processor_config,
                rl_env_config=rl_env_config,
            )

        for key, override_metadata in overrides_template.items():
            # Skip HyperPod specific key since it's not actually present within recipes
            if key == "namespace":
                continue
            # TODO: Need to figure out what this actually refers to within the recipe. Until then, we won't validate it.
            elif key == "max_context_length":
                continue
            # Validate instance type manually since it's not actually present within recipes
            elif key == "instance_type":
                allowed_values = override_metadata.get("enum", None)
                if allowed_values and instance_type not in allowed_values:
                    errors.append(
                        f"Instance type '{instance_type}' is not supported. "
                        f"Allowed types: {sorted(allowed_values)}"
                    )
                continue

            try:
                recipe_value = get_recipe_value(recipe, key)
            except Exception:
                if override_metadata.get("required", False):
                    errors.append(
                        f"'{key}' is required, but was not found in your recipe"
                    )
                continue

            # Validate proper types are used
            if "type" in override_metadata:
                python_type_name = TYPE_ALIASES.get(override_metadata["type"])
                if python_type_name is None:
                    continue
                expected_type = getattr(builtins, python_type_name, None)
                if expected_type is None:
                    errors.append(f"Unknown type '{expected_type}' for '{key}'")
                elif not isinstance(recipe_value, expected_type):
                    errors.append(
                        f"'{key}' expects {override_metadata['type']}. You provided {type(recipe_value).__name__}."
                    )
                    continue  # If wrong type is used, continue to next key to prevent type exceptions
            # Validate enum constraints are met
            if "enum" in override_metadata:
                if recipe_value not in override_metadata["enum"] and recipe_value != "":
                    errors.append(
                        f"'{key}' must be one of {override_metadata['enum']}. You provided {recipe_value}."
                    )
            # Validate minimum value constraints are met
            if "min" in override_metadata:
                if recipe_value < override_metadata["min"]:
                    errors.append(
                        f"'{key}' must be at least {override_metadata['min']}. You provided {recipe_value}."
                    )
            # Validate maximum value constraints are met
            if "max" in override_metadata:
                if recipe_value > override_metadata["max"]:
                    errors.append(
                        f"'{key}' must be no greater than {override_metadata['max']}. You provided {recipe_value}."
                    )

    @staticmethod
    def validate_job_name(job_name: str) -> None:
        """
        Validation method that checks job name

        Args:
            job_name: User provided job name

        Raises:
            ValueError: If validation fails
        """
        if not JOB_NAME_REGEX.match(job_name):
            raise ValueError(f"Job name must fit pattern ${JOB_NAME_REGEX.pattern}")

    @staticmethod
    def validate_namespace(namespace: str) -> None:
        """
        Validation method that checks namespace

        Args:
            namespace: User provided job name

        Raises:
            ValueError: If validation fails
        """
        if not NAMESPACE_REGEX.match(namespace):
            raise ValueError(f"Namespace must fit pattern ${NAMESPACE_REGEX.pattern}")

    @staticmethod
    def validate_cluster_name(cluster_name: str) -> None:
        """
        Validation method that checks cluster name

        Args:
            cluster_name: User provided job name

        Raises:
            ValueError: If validation fails
        """
        if not CLUSTER_NAME_REGEX.match(cluster_name):
            raise ValueError(
                f"Cluster name must fit pattern ${CLUSTER_NAME_REGEX.pattern}"
            )

    @staticmethod
    def validate_ecr_image_uri(image_uri: str) -> None:
        """
        Validation method that checks ECR image URI format

        Args:
            image_uri: User provided ECR image URI

        Raises:
            ValueError: If validation fails
        """
        if not ECR_IMAGE_URI_REGEX.match(image_uri):
            raise ValueError(
                f"Image URI must be a valid ECR image URI in format: "
                f"<account>.dkr.ecr.<region>.amazonaws.com/<repository>:<tag>. "
                f"Provided: {image_uri}"
            )

    @classmethod
    def validate(
        cls,
        platform: Platform,
        method: TrainingMethod,
        infra: RuntimeManager,
        recipe: Dict[str, Any],
        overrides_template: Dict[str, Any],
        output_s3_path: Optional[str] = None,
        data_s3_path: Optional[str] = None,
        validation_config: Optional[Dict[str, bool]] = None,
        rft_lambda_arn: Optional[str] = None,
        eval_task: Optional[EvaluationTask] = None,
        subtask: Optional[str] = None,
        processor_config: Optional[Dict[str, Any]] = None,
        rl_env_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Master validation method that orchestrates all validation checks.

        Args:
            platform: Training platform (SMTJ / SMHP)
            method: The training method
            infra: RuntimeManager object to validate
            recipe: Recipe dict to validate
            overrides_template: Dict containing recipe constraints
            output_s3_path: Output S3 data path
            data_s3_path: Input S3 data path
            validation_config: Optional configuration to determine which resource validation checks to perform
            rft_lambda_arn: Optional Lambda ARN for RFT
            eval_task: Optional evaluation task
            subtask: Optional subtask for evaluation
            processor_config: Optional BYOM processor configuration
            rl_env_config: Optional BYO RFT evaluation configuration

        Raises:
            ValueError: If validation fails
        """
        errors: List[str] = []

        # Get validation configuration
        default_config = cls._get_default_validation_config()
        if validation_config:
            default_config.update(validation_config)

        validation_config = default_config

        # Infrastructure validation
        if validation_config.get("iam", True):
            cls._validate_iam_permissions(errors, infra, data_s3_path, output_s3_path)
        if validation_config.get("infra", True) and platform == Platform.SMHP:
            cls._validate_infrastructure(infra, errors)

        # Recipe validation
        cls._validate_recipe(
            recipe=recipe,
            overrides_template=overrides_template,
            instance_type=infra.instance_type,
            errors=errors,
            method=method,
            rft_lambda_arn=rft_lambda_arn,
            eval_task=eval_task,
            data_s3_path=data_s3_path,
            subtask=subtask,
            processor_config=processor_config,
            rl_env_config=rl_env_config,
        )

        if errors:
            raise ValueError("\n".join(errors))
