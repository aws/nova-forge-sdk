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
"""
RFT constraints and validation.

This module defines valid configurations for RFT and provides validation functionality.
"""

import re
from enum import Enum
from typing import Any, Dict, List, Optional, get_origin

import amzn_nova_customization_sdk.recipe_config.v_two.rft_config_smhp as smhp
import amzn_nova_customization_sdk.recipe_config.v_two.rft_config_smtj as smtj
from amzn_nova_customization_sdk.model.model_enums import (
    Model,
    Platform,
    TrainingMethod,
)
from amzn_nova_customization_sdk.recipe_config.base_recipe_config import (
    BaseRecipeConfig,
)
from amzn_nova_customization_sdk.util.logging import logger
from amzn_nova_customization_sdk.util.recipe import (
    get_all_key_names,
    get_all_type_hints,
)
from amzn_nova_customization_sdk.validation.base_validator import (
    BaseValidator,
    Constraints,
    InstanceTypeConstraints,
)

LAMBDA_ARN_REGEX = re.compile(
    r"^arn:aws:lambda:[a-z0-9-]+:\d{12}:function:[A-Za-z0-9-_]+$"
)

# RFT constraints for each training combination. Based on actual recipe configurations.
CONSTRAINTS: Dict[Platform, Dict[Model, Dict[TrainingMethod, Constraints]]] = {
    Platform.SMTJ: {
        Model.NOVA_LITE_2: {
            TrainingMethod.RFT: Constraints(
                instance_type_constraints={
                    "ml.p5.48xlarge": InstanceTypeConstraints(
                        allowed_counts={4},
                        global_batch_size_options={16, 32, 64, 128, 256},
                    ),
                    "ml.p5en.48xlarge": InstanceTypeConstraints(
                        allowed_counts={4},
                        global_batch_size_options={16, 32, 64, 128, 256},
                    ),
                }
            ),
            TrainingMethod.RFT_LORA: Constraints(
                instance_type_constraints={
                    "ml.p5en.48xlarge": InstanceTypeConstraints(
                        allowed_counts={4},
                        global_batch_size_options={16, 32, 64, 128, 256},
                    ),
                    "ml.p5.48xlarge": InstanceTypeConstraints(
                        allowed_counts={4},
                        global_batch_size_options={16, 32, 64, 128, 256},
                    ),
                }
            ),
        },
    },
    Platform.SMHP: {
        Model.NOVA_LITE_2: {
            TrainingMethod.RFT: Constraints(
                instance_type_constraints={
                    "ml.p5.48xlarge": InstanceTypeConstraints(
                        allowed_counts={2, 4, 8, 16}
                    )
                }
            ),
            TrainingMethod.RFT_LORA: Constraints(
                instance_type_constraints={
                    "ml.p5.48xlarge": InstanceTypeConstraints(
                        allowed_counts={2, 4, 8, 16}
                    )
                }
            ),
        },
    },
}


class RFTValidator(BaseValidator):
    """
    Validates training configurations against defined constraints.

    Provides clear, actionable error messages when configurations are invalid.
    """

    @classmethod
    def get_constraints_registry(
        cls,
    ) -> Dict[Platform, Dict[Model, Dict[TrainingMethod, Constraints]]]:
        """Get the RFT constraints registry."""
        return CONSTRAINTS

    @classmethod
    def validate_training_config(
        cls,
        platform: Platform,
        model: Model,
        method: TrainingMethod,
        instance_type: str,
        instance_count: int,
        overrides: Optional[Dict[str, Any]] = None,
        rft_lambda_arn: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Validate that the user's inputs are supported.

        This provides clear error messages to help users understand what configurations
        are valid before submitting a job.

        Args:
            platform: SMTJ or SMHP
            model: The Nova model to train
            method: The training method to use
            instance_type: EC2 instance type
            instance_count: Number of instances
            overrides: Optional dictionary of parameter overrides
            rft_lambda_arn: Rewards Lambda ARN

        Raises:
            ValueError: If the combination is not supported, with detailed error message
        """
        errors: List[str] = []

        # Validate model
        if model != Model.NOVA_LITE_2:
            errors.append("RFT is currently only supported on Nova Lite 2.0")

        # Validate Lambda
        if rft_lambda_arn is None or not LAMBDA_ARN_REGEX.match(rft_lambda_arn):
            errors.append(
                f"You must provide a valid Lambda function ARN for 'rft_lambda_arn' when performing RFT training."
            )

        # Validate instance constraints
        constraints = RFTValidator.get_constraints(platform, model, method)
        instance_type_constraint = None
        if constraints:
            allowed_types = constraints.get_all_instance_types()
            if instance_type not in allowed_types:
                errors.append(
                    f"Instance type '{instance_type}' is not supported. "
                    f"Allowed types: {sorted(allowed_types)}"
                )
            allowed_counts = constraints.get_allowed_counts_for_type(instance_type)
            if allowed_counts and instance_count not in allowed_counts:
                errors.append(
                    f"Instance count {instance_count} is not supported for instance type '{instance_type}'. "
                    f"Allowed counts for this type: {sorted(allowed_counts)}"
                )
            instance_type_constraint = constraints.instance_type_constraints.get(
                instance_type
            )

        # Validate overrides
        if overrides:
            recipe_class: type[BaseRecipeConfig]
            if platform is Platform.SMTJ:
                recipe_class = smtj.RFTRecipeConfig
            elif platform is Platform.SMHP:
                recipe_class = smhp.RFTRecipeConfig
            else:
                raise ValueError(f"Unsupported platform: {platform}")

            type_hints = get_all_type_hints(recipe_class)
            key_names = get_all_key_names(recipe_class)

            for key, value in overrides.items():
                # Validate that the overrides actually belong within the RFT recipe
                if key not in key_names:
                    logger.info(f"Unknown override '{key}', will be ignored later")
                    continue
                # Validate that user doesn't override non-overrideable keys
                if key in ["optimizer", "type"]:
                    logger.info(f"'{key}' is not overrideable. Will use default value.")
                    continue
                # Validate that the overrides are the correct type
                expected_type = type_hints[key]
                origin_type = get_origin(expected_type)
                if isinstance(expected_type, type) and issubclass(expected_type, Enum):
                    if not isinstance(value, str):
                        errors.append(
                            f"Override '{key}' expects str (for enum {expected_type.__name__}), got {type(value).__name__}"
                        )
                    continue
                if origin_type is not None:
                    if not isinstance(value, origin_type):
                        errors.append(
                            f"Override '{key}' expects {origin_type.__name__}, got {type(value).__name__}"
                        )
                        continue
                else:
                    if not isinstance(value, expected_type):
                        errors.append(
                            f"Override '{key}' expects {expected_type.__name__}, got {type(value).__name__}"
                        )
                        continue
                # Validate per-override requirements are met
                if key == "global_batch_size":
                    if (
                        instance_type_constraint
                        and instance_type_constraint.global_batch_size_options
                    ):
                        if (
                            value
                            not in instance_type_constraint.global_batch_size_options
                        ):
                            errors.append(
                                f"Override {key}={value} is not valid. "
                                f"Allowed values for instance type '{instance_type}': "
                                f"{sorted(instance_type_constraint.global_batch_size_options)}"
                            )
                    continue
                if key == "weight_decay":
                    if not 0.0 <= value <= 1.0:
                        errors.append(
                            f"Override '{key}' must be between 0.0 and 1.0 (inclusive). You provided {value}."
                        )
                        continue
                if key == "loraplus_lr_ratio":
                    if not 0.0 <= value <= 100.0:
                        errors.append(
                            f"Override '{key}' must be between 0.0 and 100.0 (inclusive). You provided {value}."
                        )
                        continue
                if key == "alpha":
                    if value not in (32, 64, 96, 128, 160, 192):
                        errors.append(
                            f"Override '{key}' must be one of (32, 64, 96, 128, 160, 192). You provided {value}."
                        )
                        continue
                if key == "adapter_dropout":
                    if not 0.0 <= value <= 1.0:
                        errors.append(
                            f"Override '{key}' must be between 0.0 and 1.0 (inclusive). You provided {value}."
                        )
                        continue
                if key == "generation_replicas":
                    if value <= 0:
                        errors.append(
                            f"Override '{key}' must be at least 1. You provided {value}."
                        )
                        continue
                if key == "rollout_worker_replicas":
                    if value <= 0:
                        errors.append(
                            f"Override '{key}' must be at least 1. You provided {value}."
                        )
                        continue

        if errors:
            error_msg = (
                f"Invalid configuration for platform={platform.value}, model={model.value}, "
                f"method={method.value}:\n"
                + "\n".join(f"  - {error}" for error in errors)
            )
            raise ValueError(error_msg)
