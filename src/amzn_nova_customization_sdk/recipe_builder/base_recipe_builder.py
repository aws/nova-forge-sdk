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
Abstract recipe builder that generates YAML configuration files for different training methods.

This module handles all recipe generation logic to create the appropriate YAML file.
"""

import os
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from tempfile import mkdtemp
from typing import Any, Dict, Optional

import yaml

from amzn_nova_customization_sdk.model.model_enums import Platform, TrainingMethod
from amzn_nova_customization_sdk.recipe_config.base_recipe_config import (
    BaseRecipeConfig,
    BaseRunConfig,
)
from amzn_nova_customization_sdk.util.logging import logger
from amzn_nova_customization_sdk.util.recipe import RecipePath

HYPERPOD_RECIPE_PATH = os.path.join(
    "sagemaker_hyperpod_recipes",
    "recipes_collection",
    "recipes",
)


class BaseRecipeBuilder(ABC):
    def __init__(
        self,
        job_name: str,
        platform: Platform,
        method: TrainingMethod,
        model_type: str,
        model_path: str,
        instance_type: str,
        instance_count: int,
        data_s3_path: str,
        output_s3_path: str,
        overrides: Dict[str, Any],
        infra: Optional[Any] = None,
        generated_recipe_dir: Optional[str] = None,
    ):
        self.job_name = job_name
        self.platform = platform
        self.method = method
        self.model_type = model_type
        self.model_path = model_path
        self.instance_type = instance_type
        self.instance_count = instance_count
        self.data_s3_path = data_s3_path
        self.output_s3_path = output_s3_path
        self.overrides = overrides
        self.infra = infra
        self.generated_recipe_dir = generated_recipe_dir

    def _create_base_run_config(self) -> BaseRunConfig:
        return BaseRunConfig(
            name=self.job_name,
            model_type=self.model_type,
            model_name_or_path=self.model_path,
            data_s3_path=self.data_s3_path,
            output_s3_path=self.output_s3_path,
            replicas=self.instance_count,
        )

    def _get_value(self, key: str, default_getter) -> Any:
        """
        Get a configuration value, preferring user override over computed default.

        Args:
            key: The configuration key
            default_getter: A callable that returns the default value

        Returns:
            The override value if present, otherwise the default value
        """
        return self.overrides.get(key, default_getter())

    def _generate_recipe_path(self, provided_recipe_path: Optional[str]) -> RecipePath:
        """
        Generate a path to save a recipe YAML file at

        Args:
            provided_recipe_path: The path specified by callers of `build`, if it is present

        Returns:
            The path where the file will be saved at
        """
        if provided_recipe_path is not None:
            return RecipePath(provided_recipe_path)
        elif self.platform == Platform.SMTJ:
            # Create a temporary directory to hold SMTJ recipe YAMLs
            temp = self.generated_recipe_dir is None
            try:
                root = (
                    mkdtemp()
                    if self.generated_recipe_dir is None
                    else self.generated_recipe_dir
                )
            except Exception as e:
                logger.info(
                    f"Failed to resolve generated_recipes_dir dynamically, using 'generated-recipes'.\nIssue: {e}"
                )
                root = f"generated-recipes-{str(uuid.uuid4())[:8]}"
                temp = True

            path = os.path.join(
                root,
                f"{self.job_name}-{datetime.now():%b_%d}-{str(uuid.uuid4())[:3]}.yaml",
            )

            return RecipePath(path, root=root, temp=temp)
        else:
            try:
                import hyperpod_cli
            except ModuleNotFoundError as e:
                raise RuntimeError(
                    "The HyperPod CLI is a required dependency for running HyperPod jobs. "
                    "Installation details: https://github.com/aws/sagemaker-hyperpod-cli/blob/release_v2/README.md#installation"
                ) from e

            # TODO: Will eventually need to allow "training" for method_infix
            method_infix = (
                "evaluation"
                if self.method == TrainingMethod.EVALUATION
                else "fine-tuning"
            )

            prefix = os.path.join(
                os.path.dirname(hyperpod_cli.__file__), HYPERPOD_RECIPE_PATH
            )

            path = os.path.join(
                prefix,
                method_infix,
                "nova",
                f"{self.job_name}-{str(uuid.uuid4())[:3]}.yaml",
            )

            return RecipePath(path)

    @abstractmethod
    def _validate_user_input(
        self, validation_config: Optional[Dict[str, bool]] = None
    ) -> None:
        """
        Validate that the combination of model, training method, instance type,
        instance count, and overrides are supported with a legitimate recipe.

        This allows the user to know if their input is invalid prior to submitting a job.

        Raises:
            ValueError: If the combination of model, training method, instance type,
                       instance count, or overrides isn't supported
        """
        pass

    @abstractmethod
    def _build_recipe_config(self) -> BaseRecipeConfig:
        """
        Build the recipe configuration based on the user input.

        This method should be implemented by subclasses to create the specific recipe configuration
        for the chosen training method.
        :return: BaseRecipeConfig
        """
        pass

    def build(
        self,
        output_recipe_file: Optional[str] = None,
        validation_config: Optional[Dict[str, bool]] = None,
    ) -> RecipePath:
        """
        Generate the recipe based on the user input.

        Args:
            output_recipe_file: Path where the recipe YAML should be saved
            validation_config: Optional validation configuration dict

        Returns:
            str: Path to the generated recipe file

        Raises:
            ValueError: If the training method is not supported or configuration is invalid
        """
        # Validate user input before building
        self._validate_user_input(validation_config)

        # Build recipe while incorporating overrides
        recipe_config = self._build_recipe_config()

        # Serialize to YAML
        config_dict = recipe_config.to_dict()
        yaml_str = yaml.dump(
            config_dict, default_flow_style=False, sort_keys=False, width=120
        )

        # Save to file
        output_recipe_path = self._generate_recipe_path(output_recipe_file)
        output_recipe_file = output_recipe_path.path

        os.makedirs(os.path.dirname(output_recipe_file), exist_ok=True)
        with open(output_recipe_file, "w") as f:
            f.write(yaml_str)

        logger.info(
            f"Successfully generated recipe and saved it to '{output_recipe_file}'"
        )
        return output_recipe_path
