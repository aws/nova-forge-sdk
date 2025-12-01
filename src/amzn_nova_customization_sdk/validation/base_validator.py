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
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import boto3

from amzn_nova_customization_sdk.manager.runtime_manager import (
    RuntimeManager,
    SMHPRuntimeManager,
)
from amzn_nova_customization_sdk.model.model_enums import (
    Model,
    Platform,
    TrainingMethod,
)
from amzn_nova_customization_sdk.util.sagemaker import get_cluster_instance_info


@dataclass
class InstanceTypeConstraints:
    """
    Defines constraints for a specific instance type.

    Each instance type may have different valid instance counts and configuration limits.
    """

    allowed_counts: Set[int]
    max_length_range: Optional[Tuple[int, int]] = None  # (min, max)
    global_batch_size_options: Optional[Set[int]] = None


@dataclass
class Constraints:
    """
    Defines valid configurations for a specific model/method combination.

    Maps instance types to their specific constraints.
    """

    instance_type_constraints: Dict[str, InstanceTypeConstraints]

    def get_all_instance_types(self) -> Set[str]:
        return set(self.instance_type_constraints.keys())

    def get_allowed_counts_for_type(self, instance_type: str) -> Optional[Set[int]]:
        if instance_type in self.instance_type_constraints:
            return self.instance_type_constraints[instance_type].allowed_counts
        return None


class BaseValidator(ABC):
    """
    Base validator class providing common validation functionality.
    """

    @classmethod
    @abstractmethod
    def get_constraints_registry(
        cls,
    ) -> Dict[Platform, Dict[Model, Dict[TrainingMethod, Constraints]]]:
        """
        Get the constraints registry for this validator.

        Must be implemented by subclasses to return their CONSTRAINTS dictionary.
        """
        pass

    @classmethod
    def get_constraints(
        cls, platform: Platform, model: Model, method: TrainingMethod
    ) -> Optional[Constraints]:
        """
        Get the constraints for a specific training combination.

        Args:
            platform: SMTJ or SMHP
            model: The Nova model
            method: The training method

        Returns:
            Constraints if defined, None otherwise
        """
        constraints_registry = cls.get_constraints_registry()
        if platform in constraints_registry:
            if model in constraints_registry[platform]:
                return constraints_registry[platform][model].get(method)
        return None

    @classmethod
    def validate(
        cls,
        platform: Platform,
        model: Model,
        method: TrainingMethod,
        instance_type: str,
        instance_count: int,
        infra: Optional[RuntimeManager] = None,
        overrides: Optional[Dict[str, Any]] = None,
        validation_config: Optional[Dict[str, bool]] = None,
        **kwargs,
    ) -> None:
        """
        Master validation method that orchestrates all validation checks.

        Args:
            platform: SMTJ or SMHP
            model: The Nova model to train
            method: The training method to use
            instance_type: EC2 instance type
            instance_count: Number of instances
            overrides: Optional dictionary of parameter overrides
            **kwargs: Additional arguments for specific training methods

        Raises:
            ValueError: If validation fails
        """
        # Get validation configuration
        default_config = BaseValidator._get_default_validation_config()
        if validation_config:
            default_config.update(validation_config)

        validation_config = default_config

        errors: List[str] = []

        # Infrastructure validation
        if infra is not None:
            if validation_config.get("iam", True):
                cls._validate_iam_permissions(errors, infra)

            if validation_config.get("infra", True) and platform == Platform.SMHP:
                cls._validate_infrastructure(
                    infra, instance_type, instance_count, errors
                )

        # Training-specific validation
        cls.validate_training_config(
            platform=platform,
            model=model,
            method=method,
            instance_type=instance_type,
            instance_count=instance_count,
            overrides=overrides,
            **kwargs,
        )

        if errors:
            raise ValueError("\n".join(errors))

    @classmethod
    @abstractmethod
    def validate_training_config(
        cls,
        platform: Platform,
        model: Model,
        method: TrainingMethod,
        instance_type: str,
        instance_count: int,
        overrides: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """
        Validate training-specific configuration.

        Must be implemented by subclasses.
        """
        pass

    @staticmethod
    def _get_default_validation_config() -> Dict[str, bool]:
        """Get default validation configuration."""
        return {"iam": True, "infra": True}.copy()

    @staticmethod
    def _validate_iam_permissions(
        errors: List[str], infra: Optional[RuntimeManager] = None
    ) -> None:
        """
        Validate required IAM permissions for training jobs.

        Args:
            errors: List to append validation errors to
            infra: Optional infrastructure manager to check for overridden execution role
        """
        # Validate SageMaker execution role (which only applies to SMTJ)
        # TODO: Validate which permissions are needed by HyperPod CLI
        try:
            if isinstance(infra, SMHPRuntimeManager):
                return

            # Check if infra has an overridden execution role
            execution_role = None

            if (
                infra
                and hasattr(infra, "execution_role")
                and getattr(infra, "execution_role", None)
            ):
                execution_role = infra.execution_role
            else:
                import sagemaker

                execution_role = sagemaker.get_execution_role(use_default=True)

            region_name = getattr(infra, "region", "us-east-1")

            # Test if role exists and has proper trust policy
            iam_client = boto3.client("iam", region_name=region_name)

            # Extract role name from ARN, handle non-string execution roles
            if not isinstance(execution_role, str):
                errors.append(
                    f"Invalid execution role format: {type(execution_role).__name__}"
                )
                return

            role_name = execution_role.split("/")[-1]

            try:
                role_response = iam_client.get_role(RoleName=role_name)
                trust_policy = role_response["Role"]["AssumeRolePolicyDocument"]

                # Check if SageMaker service can assume this role
                sagemaker_trusted = False
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

                if not sagemaker_trusted:
                    errors.append(
                        f"SageMaker execution role {role_name} does not trust sagemaker.amazonaws.com service"
                    )

            except Exception as e:
                if "NoSuchEntity" in str(e):
                    errors.append(
                        f"SageMaker execution role {role_name} does not exist"
                    )
                elif "AccessDenied" in str(e):
                    errors.append(
                        "Missing IAM permissions: iam:GetRole required to validate execution role"
                    )
                else:
                    errors.append(f"Failed to validate execution role: {str(e)}")

        except Exception as e:
            if "Could not find credentials" in str(e):
                errors.append("AWS credentials not configured")
            elif "Unable to locate credentials" in str(e):
                errors.append("AWS credentials not found")
            else:
                errors.append(f"Failed to get SageMaker execution role: {str(e)}")

    @staticmethod
    def _validate_infrastructure(
        infra: Any, instance_type: str, instance_count: int, errors: List[str]
    ) -> None:
        """
        Validate SMHP cluster infrastructure requirements.

        Args:
            infra: SMHPRuntimeManager instance
            instance_type: Required instance type
            instance_count: Required instance count
            errors: List to append validation errors to
        """
        if infra is None:
            return

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
                if group["instance_type"] == instance_type:
                    compatible_groups.append(group)

            if not compatible_groups:
                available_types = [
                    group["instance_type"] for group in restricted_instance_groups
                ]
                errors.append(
                    f"Instance type '{instance_type}' not available in restricted instance groups in cluster '{cluster_name}'. "
                    f"Available types: {sorted(set(available_types))}"
                )
                return

            # Check if any compatible group has sufficient capacity
            sufficient_capacity = False
            for group in compatible_groups:
                if group["current_count"] >= instance_count:
                    sufficient_capacity = True
                    break

            if not sufficient_capacity:
                max_available = max(
                    group["current_count"] for group in compatible_groups
                )
                errors.append(
                    f"Insufficient capacity for instance type '{instance_type}' in cluster '{cluster_name}'. "
                    f"Required: {instance_count}, Maximum available: {max_available}"
                )

        except Exception as e:
            errors.append(f"Failed to validate cluster infrastructure: {str(e)}")
