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
This module handles all recipe generation logic to create the appropriate recipe YAML file.
"""

import os
import uuid
from datetime import datetime
from tempfile import mkdtemp
from typing import Any, Dict, List, Optional

import yaml

from amzn_nova_customization_sdk.manager.runtime_manager import RuntimeManager
from amzn_nova_customization_sdk.model.model_enums import (
    Model,
    Platform,
    TrainingMethod,
    Version,
)
from amzn_nova_customization_sdk.monitor import MLflowMonitor
from amzn_nova_customization_sdk.recipe.recipe_config import (
    EVAL_TASK_METRIC_MAP,
    EVAL_TASK_STRATEGY_MAP,
    HYPERPOD_RECIPE_PATH,
    EvaluationTask,
)
from amzn_nova_customization_sdk.util.checkpoint_util import validate_checkpoint_uri
from amzn_nova_customization_sdk.util.data_mixing import DataMixing
from amzn_nova_customization_sdk.util.logging import logger
from amzn_nova_customization_sdk.util.recipe import (
    RecipePath,
    load_file_as_string,
    load_recipe_templates,
)
from amzn_nova_customization_sdk.validation.validator import Validator


class RecipeBuilder:
    def __init__(
        self,
        region: str,
        job_name: str,
        platform: Platform,
        model: Model,
        method: TrainingMethod,
        instance_type: str,
        instance_count: int,
        infra: RuntimeManager,
        output_s3_path: str,
        data_s3_path: Optional[str] = None,
        model_path: Optional[
            str
        ] = None,  # Should only be provided if user is performing incremental training on a checkpoint
        mlflow_monitor: Optional[MLflowMonitor] = None,
        # Method-specific inputs
        rft_lambda_arn: Optional[str] = None,
        validation_data_s3_path: Optional[str] = None,
        eval_task: Optional[EvaluationTask] = None,
        subtask: Optional[str] = None,
        processor_config: Optional[Dict[str, Any]] = None,
        rl_env_config: Optional[Dict[str, Any]] = None,
        data_mixing_instance: Optional[DataMixing] = None,
        image_uri_override: Optional[str] = None,
    ):
        self.region = region
        self.job_name = job_name
        self.platform = platform

        # Model
        self.model = model
        self.model_type = model.model_type
        self.model_name_or_path = model_path or model.model_path

        self.method = method

        # Infrastructure
        self.instance_type = instance_type
        self.instance_count = instance_count
        self.infra = infra

        # Input / output S3 paths
        self.output_s3_path = output_s3_path
        self.data_s3_path = data_s3_path

        # datamixing
        self.data_mixing_instance = data_mixing_instance

        # Image URI override
        self.image_uri_override = image_uri_override

        # MLflow
        if mlflow_monitor:
            self.mlflow_tracking_uri = mlflow_monitor.tracking_uri
            self.mlflow_experiment_name = (
                mlflow_monitor.experiment_name or self.job_name
            )
            self.mlflow_run_name = (
                mlflow_monitor.run_name or f"{self.job_name}-{str(uuid.uuid4())[:8]}"
            )
        else:
            self.mlflow_tracking_uri = None

        # RFT
        if method == TrainingMethod.RFT_LORA or method == TrainingMethod.RFT_FULL:
            self.rft_lambda_arn = rft_lambda_arn
        elif rft_lambda_arn is not None:
            logger.info("'rft_lambda_arn' is only required for RFT. Will ignore.")

        # CPT
        if method == TrainingMethod.CPT:
            self.validation_data_s3_path = validation_data_s3_path
        elif validation_data_s3_path is not None:
            logger.info(
                "'validation_data_s3_path' is only applicable for CPT. Will ignore."
            )

        # Eval
        if method == TrainingMethod.EVALUATION:
            if eval_task is None:
                raise ValueError(
                    "'eval_task' is a required parameter when calling evaluate()."
                )

            self.eval_task = eval_task
            self.strategy = EVAL_TASK_STRATEGY_MAP[eval_task]
            self.metric = EVAL_TASK_METRIC_MAP[eval_task]
            self.subtask = subtask
            self.processor_config = processor_config
            self.rl_env_config = rl_env_config
        else:
            EVAL_PARAMS = [eval_task, subtask, processor_config, rl_env_config]
            for param in EVAL_PARAMS:
                if param is not None:
                    logger.info(
                        f"'{param}' is only required for evaluation. Will ignore."
                    )

    def _load_input_recipe(self, input_recipe_path: str):
        """
        Load a user's input recipe, transform it into a dict, and then save it for later use

        Args:
            input_recipe_path: String path of where user has an input recipe stored (can be S3 URI or local)
        """

        def convert_scientific_notation_strings(obj):
            """
            Recursively convert string scientific notation to float.

            Args:
                obj: The object to convert
            """
            if isinstance(obj, dict):
                return {
                    k: convert_scientific_notation_strings(v) for k, v in obj.items()
                }
            elif isinstance(obj, list):
                return [convert_scientific_notation_strings(item) for item in obj]
            elif isinstance(obj, str):
                # Check if it looks like scientific notation
                try:
                    if "e-" in obj.lower():
                        return float(obj)
                except (ValueError, AttributeError):
                    pass
                return obj
            else:
                return obj

        input_recipe_str = load_file_as_string(input_recipe_path, ".yaml")

        try:
            input_recipe_dict: Dict[str, Any] = yaml.safe_load(input_recipe_str)
        except yaml.YAMLError:
            raise ValueError(
                f"Failed to parse provided recipe at {input_recipe_path}. Please validate that the file is valid YAML format."
            )

        # Convert string scientific notation to floats (handles PyYAML quirks)
        input_recipe_dict = convert_scientific_notation_strings(input_recipe_dict)

        # Store the full input recipe for later use
        if isinstance(input_recipe_dict, dict):
            self.input_recipe_dict = input_recipe_dict
        else:
            raise ValueError(
                f"Failed to parse provided recipe at {input_recipe_path}. Please validate that the file is valid YAML format."
            )

    def _resolve_user_inputs(
        self,
        recipe_template: Dict[str, Any],
        overrides_template: Dict[str, Any],
        overrides: Dict[str, Any],
        input_recipe_path: Optional[str] = None,
        allowed_instance_count: Optional[int] = None,
        allowed_instance_types: Optional[List[str]] = None,
    ):
        """
        Resolve a user's inputs from overrides and input recipe in order of precedence:
        1. overrides
        2. input_recipe
        3. Other user input via things like NovaModelCustomizer objects and input parameters to train() and evaluate()

        Args:
            recipe_template: Dict of the recipe template corresponding to the training that the user wants to perform
            overrides_template: Dict of recipe field restrictions including default values, type, min/max, enum, etc.
            overrides: Overrides specified by the user (this may be an empty Dict)
            input_recipe_path: Optional local/S3 path to a user's input recipe that they want to use for training
            allowed_instance_count: Allowed instance count according to the DescribeHubContent API
            allowed_instance_types: Allowed instance types according to the DescribeHubContent API
        """

        def get_leaves(d: Dict[str, Any]) -> Dict[str, Any]:
            """
            Recursively gets and returns all key/value leaves for a given Dict.
            Keeps only the first occurrence of duplicate keys (useful for name: distributed_fused_adam)

            Any key/value pair where the value is not of type Dict is considered a leaf.

            Args:
                d: The Dict to traverse

            Returns:
                Dict of leaf key/value pairs of the original Dict
            """
            leaves: Dict[str, Any] = {}
            for key, value in d.items():
                if isinstance(value, dict):
                    new_leaves = get_leaves(d=value)
                    for k, v in new_leaves.items():
                        if k not in leaves:
                            leaves[k] = v
                else:
                    if key not in leaves:
                        leaves[key] = value
            return leaves

        def apply_user_provided_inputs_into_overrides_template():
            """
            Read user provided inputs (that are NOT part of overrides and input_recipes)
            and apply them into the overrides_template that was fetched from Jump Start's S3 bucket.

            These will be values that the user provided via their NovaModelCustomizer object,
            or from things like train() or evaluate() parameters.
            These variables have the lowest precedence when generating recipes (compared with overrides and input recipes).
            """
            # Job name
            overrides_template.setdefault("name", {})["default"] = self.job_name

            # Model
            overrides_template.setdefault("model_type", {})["default"] = self.model_type
            overrides_template.setdefault("model_name_or_path", {})["default"] = (
                self.model_name_or_path
            )

            # Input and output S3 paths
            overrides_template.setdefault("data_s3_path", {})["default"] = (
                self.data_s3_path or ""
            )
            overrides_template.setdefault("output_s3_path", {})["default"] = (
                self.output_s3_path
            )

            # Instance count
            if (
                "replicas" not in overrides_template
                and "run" in recipe_template
                and "replicas" in recipe_template["run"]
            ):
                overrides_template["replicas"] = {
                    "default": self.infra.instance_count,
                    "enum": [recipe_template["run"]["replicas"]],
                    "type": "integer",
                }
            else:
                overrides_template.setdefault("replicas", {})["default"] = (
                    self.infra.instance_count
                )
            if (
                allowed_instance_count is not None
                and allowed_instance_count not in overrides_template["replicas"]["enum"]
            ):
                overrides_template["replicas"]["enum"].append(allowed_instance_count)

            # Instance type
            if allowed_instance_types and "instance_type" not in overrides_template:
                overrides_template["instance_type"] = {"enum": allowed_instance_types}

            # MlFlow
            if self.mlflow_tracking_uri is not None:
                overrides_template.setdefault("mlflow_tracking_uri", {})["default"] = (
                    self.mlflow_tracking_uri
                )
                overrides_template.setdefault("mlflow_experiment_name", {})[
                    "default"
                ] = self.mlflow_experiment_name
                overrides_template.setdefault("mlflow_run_name", {})["default"] = (
                    self.mlflow_run_name
                )

            # RFT
            if (
                self.method == TrainingMethod.RFT_LORA
                or self.method == TrainingMethod.RFT_FULL
            ):
                overrides_template.setdefault("reward_lambda_arn", {})["default"] = (
                    self.rft_lambda_arn
                )

            # CPT
            if (
                self.method == TrainingMethod.CPT
                and self.validation_data_s3_path is not None
            ):
                overrides_template.setdefault("validation_s3_path", {})["default"] = (
                    self.validation_data_s3_path
                )

            # Evaluation
            elif self.method == TrainingMethod.EVALUATION:
                overrides_template.setdefault("task", {})["default"] = (
                    self.eval_task.value
                )
                overrides_template.setdefault("strategy", {})["default"] = (
                    self.strategy.value
                )
                overrides_template.setdefault("metric", {})["default"] = (
                    self.metric.value
                )
                if self.subtask is not None:
                    overrides_template.setdefault("subtask", {})["default"] = (
                        self.subtask
                    )
                if self.processor_config is not None:
                    if self.processor_config.get("lambda_arn"):
                        overrides_template.setdefault("lambda_arn", {})["default"] = (
                            self.processor_config["lambda_arn"]
                        )
                    if self.processor_config.get("lambda_type"):
                        overrides_template.setdefault("lambda_type", {})["default"] = (
                            self.processor_config["lambda_type"]
                        )
                    if (
                        self.processor_config.get("preprocessing")
                        and self.processor_config["preprocessing"].get("enabled")
                        is not None
                    ):
                        overrides_template.setdefault("preprocessing", {})[
                            "default"
                        ] = self.processor_config["preprocessing"]["enabled"]
                        overrides_template.setdefault("enabled", {})["type"] = "boolean"
                    if (
                        self.processor_config.get("postprocessing")
                        and self.processor_config["postprocessing"].get("enabled")
                        is not None
                    ):
                        overrides_template.setdefault("postprocessing", {})[
                            "default"
                        ] = self.processor_config["postprocessing"]["enabled"]
                        overrides_template.setdefault("enabled", {})["type"] = "boolean"
                    if self.processor_config.get("aggregation"):
                        overrides_template.setdefault("aggregation", {})["default"] = (
                            self.processor_config["aggregation"]
                        )
                        overrides_template["aggregation"]["type"] = "string"
                        overrides_template["aggregation"]["enum"] = [
                            "average",
                            "min",
                            "max",
                            "sum",
                        ]
                if self.rl_env_config is not None and self.rl_env_config.get(
                    "reward_lambda_arn"
                ):
                    overrides_template.setdefault("reward_lambda_arn", {})[
                        "default"
                    ] = self.rl_env_config["reward_lambda_arn"]

        def handle_edge_case_keys(key: str) -> bool:
            """
            When resolving user input, we first want to check for certain "edge case" keys to require special attention.
            These mainly include variables that we do not want the user to override.

            Args:
                key: The current recipe_template key to handle

            Returns:
                Boolean. If True, continue resolving the current recipe key/value pair. False otherwise.
            """

            # We won't allow overriding of the model information because it gets too confusing on how to validate. We just want to use the NovaModelCustomizer initialization values.
            if key == "model_type":
                if key in overrides:
                    logger.warning(
                        f"Override for '{key}' will be ignored. If you wish to use a different model than {self.model.name}, please update your NovaModelCustomizer object."
                    )
                    return False
                elif key in input_recipe_key_values:
                    if self.model_type != input_recipe_key_values[key]:
                        logger.warning(
                            f"{key} '{input_recipe_key_values[key]}' will be ignored from your input recipe. If you wish to use a different model than {self.model.name}, please update your NovaModelCustomizer object."
                        )
                        return False
            if key == "model_name_or_path":
                if key in overrides:
                    if not str(overrides[key]).startswith("s3://"):
                        logger.warning(
                            f"Override for '{key}' will be ignored. If you wish to use a different model than {self.model.name}, please update your NovaModelCustomizer object."
                        )
                        return False
                    else:
                        validate_checkpoint_uri(
                            checkpoint_uri=str(overrides[key]), region=self.region
                        )
                elif key in input_recipe_key_values:
                    if not str(input_recipe_key_values[key]).startswith(
                        "s3://"
                    ) and self.model.model_path != str(input_recipe_key_values[key]):
                        logger.warning(
                            f"{key} '{str(input_recipe_key_values[key])}' will be ignored from your input recipe. If you wish to use a different model than {self.model.name}, please update your NovaModelCustomizer object."
                        )
                        return False
                    elif str(input_recipe_key_values[key]).startswith("s3://"):
                        validate_checkpoint_uri(
                            checkpoint_uri=str(input_recipe_key_values[key]),
                            region=self.region,
                        )
            # We won't allow overriding of the evaluation task because it gets too confusing on how to validate. We just want to use the eval_task parameter from evaluate().
            if key == "task":
                if key in overrides:
                    logger.warning(
                        f"Override for '{key}' will be ignored. If you wish to use an evaluation task other than {self.eval_task.name}, please pass a different value for 'eval_task' when calling evaluate()."
                    )
                    return False
                elif key in input_recipe_key_values:
                    if (
                        self.eval_task
                        and self.eval_task.value != input_recipe_key_values[key]
                    ):
                        logger.warning(
                            f"{key} '{input_recipe_key_values[key]}' will be ignored from your input recipe. If you wish to use a different evaluation task than {self.eval_task.name}, please pass a different value for 'eval_task' when calling evaluate()."
                        )
                        return False

            # Certain parameters would cause jobs to fail, so we don't want to allow the user to override them.
            NON_OVERRIDEABLE_PARAMS = ["peft_scheme"]
            for param in NON_OVERRIDEABLE_PARAMS:
                if (overrides.pop(param, None)) is not None:
                    logger.warning(
                        f"'{param}' is not an overrideable parameter. Will be ignored."
                    )
                    return False

            return True

        def update_overrides_template(recipe_template: Dict[str, Any]):
            """
            Recursively update the overrides_template in order of user input precedence:
            1. overrides
            2. input_recipe
            3. Other user input via things like NovaModelCustomizer objects and input parameters to train() and evaluate()

            At the end of this function, the overrides_template will be fully resolved from user input.
            So for every key/value in the recipe template, we can find the corresponding entry in the
            overrides_template variable, see what value to user (based on the resolved user input),
            as well as variable restrictions such as type, min/max, enum, etc.

            Args:
                recipe_template: Dict of the current portion of the recipe_template to process
            """

            for recipe_template_key, recipe_template_value in recipe_template.items():
                # Handle edge case keys first before trying to apply overrides
                if not handle_edge_case_keys(key=recipe_template_key):
                    continue

                if isinstance(recipe_template_value, dict):
                    update_overrides_template(recipe_template=recipe_template_value)
                else:
                    # Override takes highest precedence
                    if recipe_template_key in overrides:
                        # we only use datamixing values from data_mixing_instance
                        if (
                            self.data_mixing_instance
                            and self.data_mixing_instance._is_data_mixing_field(
                                recipe_template_key
                            )
                        ):
                            logger.warning(
                                f"The following data mixing keys in overrides recipe will be ignored: {recipe_template_key}. "
                                f"Data mixing configuration can only be set using set_datamixing_config()."
                            )
                            continue
                        if "{{" in str(recipe_template_value):
                            overrides_template_key = (
                                str(recipe_template_value)
                                .removeprefix("'")
                                .removeprefix("{{")
                                .removesuffix("}}")
                                .removesuffix("'")
                            )
                            if recipe_template_key == overrides_template_key:
                                overrides_template[overrides_template_key][
                                    "default"
                                ] = overrides[recipe_template_key]
                            else:
                                overrides_template[recipe_template_key] = (
                                    overrides_template[overrides_template_key]
                                )
                                overrides_template[overrides_template_key][
                                    "required"
                                ] = False
                                overrides_template[recipe_template_key]["default"] = (
                                    overrides[recipe_template_key]
                                )
                        else:
                            existing_enum = overrides_template.get(
                                recipe_template_key, {}
                            ).get("enum")
                            overrides_template[recipe_template_key] = {
                                "default": overrides[recipe_template_key],
                                "type": type(recipe_template_value).__name__,
                                "required": True,
                            }
                            if existing_enum is not None:
                                overrides_template[recipe_template_key]["enum"] = (
                                    existing_enum
                                )
                    # If no override, check input recipe
                    # we only use datamixing values from data_mixing_instance
                    elif recipe_template_key in input_recipe_key_values:
                        if (
                            self.data_mixing_instance
                            and self.data_mixing_instance._is_data_mixing_field(
                                recipe_template_key
                            )
                        ):
                            logger.warning(
                                f"The following data mixing keys in input recipe will be ignored: {recipe_template_key}. "
                                f"Data mixing configuration can only be set using set_datamixing_config()."
                            )
                            continue
                        if "{{" in str(recipe_template_value):
                            overrides_template_key = (
                                str(recipe_template_value)
                                .removeprefix("'")
                                .removeprefix("{{")
                                .removesuffix("}}")
                                .removesuffix("'")
                            )
                            if recipe_template_key == overrides_template_key:
                                overrides_template[overrides_template_key][
                                    "default"
                                ] = input_recipe_key_values[recipe_template_key]
                            else:
                                overrides_template[recipe_template_key] = (
                                    overrides_template[overrides_template_key]
                                )
                                overrides_template[overrides_template_key][
                                    "required"
                                ] = False
                                overrides_template[recipe_template_key]["default"] = (
                                    input_recipe_key_values[recipe_template_key]
                                )
                        else:
                            existing_enum = overrides_template.get(
                                recipe_template_key, {}
                            ).get("enum")
                            overrides_template[recipe_template_key] = {
                                "default": input_recipe_key_values[recipe_template_key],
                                "type": type(recipe_template_value).__name__,
                                "required": True,
                            }
                            if existing_enum is not None:
                                overrides_template[recipe_template_key]["enum"] = (
                                    existing_enum
                                )
                    # If no override or input recipe, use default value from recipe template
                    else:
                        if "{{" in str(recipe_template_value):
                            overrides_template_key = (
                                str(recipe_template_value)
                                .removeprefix("'")
                                .removeprefix("{{")
                                .removesuffix("}}")
                                .removesuffix("'")
                            )
                            # we only use datamixing values from data_mixing_instance
                            if (
                                not self.data_mixing_instance
                                or not self.data_mixing_instance._is_data_mixing_field(
                                    recipe_template_key
                                )
                            ):
                                if recipe_template_key != overrides_template_key:
                                    overrides_template[recipe_template_key] = (
                                        overrides_template[overrides_template_key]
                                    )
                                    overrides_template[overrides_template_key][
                                        "required"
                                    ] = False

        input_recipe_key_values = {}

        # Load input recipe
        if input_recipe_path is not None:
            self._load_input_recipe(input_recipe_path=input_recipe_path)
            input_recipe_key_values = get_leaves(d=self.input_recipe_dict)

        # Apply user-provided inputs first (that are not part of overrides or input_recipe) into overrides_template
        apply_user_provided_inputs_into_overrides_template()

        # Apply overrides and input_recipe values into overrides_template second
        # This overwrites previous input to ensure proper precedence of user input
        update_overrides_template(recipe_template=recipe_template)

        # Handle data mixing if enabled
        if self.data_mixing_instance:
            data_mixing_config = self.data_mixing_instance.get_config()
            override_keys = {
                k
                for k in overrides.keys()
                if self.data_mixing_instance._is_data_mixing_field(k)
            }
            for datamixing_override_key in override_keys:
                del overrides[datamixing_override_key]
                logger.warning(
                    f"The following data mixing keys in overrides recipe will be ignored: {datamixing_override_key}. "
                    f"Data mixing configuration can only be set using set_datamixing_config()."
                )

            # Update overrides_template with normalized values
            # TODO Investigate some percent params are integer type but default value is float
            # Changing all to float right now
            for key, value in data_mixing_config.items():
                if (
                    key in overrides_template
                    and key != DataMixing.DATASET_CATALOG_FIELD
                ):
                    if key == DataMixing.CUSTOMER_DATA_FIELD:
                        data_mixing_recipe_key = "percent"
                    else:
                        data_mixing_recipe_key = key.removeprefix(
                            DataMixing.NOVA_PREFIX
                        )
                        data_mixing_recipe_key = data_mixing_recipe_key.removesuffix(
                            DataMixing.PERCENT_SUFFIX
                        )

                    overrides_template[data_mixing_recipe_key] = overrides_template[key]
                    overrides_template[key]["default"] = float(value)
                    overrides_template[key]["required"] = False
                    overrides_template[key]["type"] = "float"

    def _build_final_recipe(
        self, recipe_template: Dict[str, Any], overrides_template: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build the final recipe dict by applying user-provided input into the recipe_template

        Args:
            recipe_template: Dict template of a recipe from Jump Start's S3 buckets
            overrides_template: Dict template of override metadata from Jump Start's S3 buckets

        Returns:
            A Dict of the built recipe
        """

        def apply_overrides_template_to_recipe_template(
            recipe: Dict[str, Any],
        ) -> Dict[str, Any]:
            """
            Recursively merge recipe with resolved values.

            Args:
                recipe: The recipe dictionary to process

            Returns:
                The merged dictionary with resolved values
            """
            for key, value in recipe.items():
                if isinstance(value, dict):
                    # Edge case in Eval BYOD where preprocessing and postprocessing have same subkey
                    if key == "preprocessing" or key == "postprocessing":
                        if key in overrides_template:
                            recipe.setdefault(key, {})["enabled"] = overrides_template[
                                key
                            ]["default"]
                        continue
                    else:
                        recipe[key] = apply_overrides_template_to_recipe_template(
                            recipe=value
                        )
                else:
                    # If the original value is "distributed_fused_adam", never replace it
                    if key == "name" and value == "distributed_fused_adam":
                        recipe[key] = value
                    # Use value from overrides_template if it exists
                    elif key in overrides_template:
                        default_value = overrides_template[key]["default"]
                        # Convert int to float for temperature (JumpStart defaults use int)
                        if key == "temperature" and isinstance(default_value, int):
                            default_value = float(default_value)
                        recipe[key] = default_value
                    else:
                        recipe[key] = value
            return recipe

        return apply_overrides_template_to_recipe_template(recipe=recipe_template)

    def _generate_recipe_path(
        self, provided_recipe_path: Optional[str] = None
    ) -> RecipePath:
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
            try:
                root = mkdtemp()
            except Exception as e:
                logger.warning(
                    f"Failed to resolve generated_recipes_dir dynamically, using 'generated-recipes'.\nIssue: {e}"
                )
                root = f"generated-recipes-{str(uuid.uuid4())[:8]}"

            path = os.path.join(
                root,
                f"{self.job_name}-{datetime.now():%b_%d}-{str(uuid.uuid4())[:3]}.yaml",
            )

            return RecipePath(path, root=root, temp=True)
        else:
            try:
                import hyperpod_cli
            except ModuleNotFoundError as e:
                raise RuntimeError(
                    "The HyperPod CLI is a required dependency for running HyperPod jobs. "
                    "Installation details: https://github.com/aws/sagemaker-hyperpod-cli/tree/release_v2?tab=readme-ov-file#installation"
                ) from e

            path_components = [
                os.path.join(
                    os.path.dirname(hyperpod_cli.__file__), HYPERPOD_RECIPE_PATH
                )
            ]

            match self.method:
                case TrainingMethod.EVALUATION:
                    path_components.append("evaluation")
                case TrainingMethod.CPT:
                    path_components.append("training")
                case (
                    TrainingMethod.RFT_LORA
                    | TrainingMethod.RFT_FULL
                    | TrainingMethod.SFT_LORA
                    | TrainingMethod.SFT_FULL
                    | TrainingMethod.DPO_LORA
                    | TrainingMethod.DPO_FULL
                ):
                    path_components.append("fine-tuning")
                case _:
                    raise ValueError(f"Unsupported training method: {self.method.name}")

            path_components.append("nova")

            if self.data_mixing_instance is not None:
                path_components.append("forge")

            match self.model.version:
                case Version.ONE:
                    path_components.append("nova_1_0")
                case Version.TWO:
                    path_components.append("nova_2_0")
                case _:
                    raise ValueError(
                        f"Unsupported Nova version: {self.model.version.name}"
                    )

            path_components.append(
                "nova_lite" if self.model == Model.NOVA_LITE_2 else self.model.value,
            )

            if self.method != TrainingMethod.EVALUATION:
                path_components.append(self.method.name.split("_", 1)[0])

            path_components.append(f"{self.job_name}-{str(uuid.uuid4())[:3]}.yaml")

            return RecipePath(os.path.join(*path_components))

    def build_and_validate(
        self,
        overrides: Optional[Dict[str, Any]] = None,
        input_recipe_path: Optional[str] = None,
        output_recipe_path: Optional[str] = None,
        validation_config: Optional[Dict[str, bool]] = None,
    ) -> tuple:
        """
        Generate the recipe based on the user input.

        Args:
            overrides: Optional dict of user overrides to apply to the recipe
            input_recipe_path: Optional path for a YAML recipe file (both S3 and local paths are accepted)
            output_recipe_path: Optional path where the recipe YAML should be saved (only local path is accepted)
            validation_config: Optional validation configuration dict

        Returns:
            tuple of resolved values: recipe_path str, output_s3_path str, data_s3_path str

        Raises:
            ValueError: If the training method is not supported or configuration is invalid
        """

        recipe_metadata, recipe_template, overrides_template, image_uri = (
            load_recipe_templates(
                model=self.model,
                method=self.method,
                platform=self.platform,
                region=self.region,
                instance_type=self.instance_type,
                data_mixing_enabled=True if self.data_mixing_instance else False,
                eval_task=getattr(self, "eval_task", None),
                image_uri_override=self.image_uri_override,
            )
        )

        # Resolve user inputs
        self._resolve_user_inputs(
            recipe_template=recipe_template,
            overrides_template=overrides_template,
            overrides=overrides or {},
            input_recipe_path=input_recipe_path,
            allowed_instance_count=recipe_metadata.get("InstanceCount")
            if recipe_metadata
            else None,
            allowed_instance_types=recipe_metadata.get("SupportedInstanceTypes")
            if recipe_metadata
            else None,
        )

        # Build recipe using resolved inputs
        final_recipe_dict = self._build_final_recipe(
            recipe_template=recipe_template, overrides_template=overrides_template
        )

        # Validate user configurations and the recipe
        if self.infra.instance_count != overrides_template["replicas"]["default"]:
            self.infra.instance_count = overrides_template["replicas"]["default"]
        Validator.validate(
            platform=self.platform,
            method=self.method,
            infra=self.infra,
            recipe=final_recipe_dict,
            overrides_template=overrides_template,
            output_s3_path=overrides_template.get("output_s3_path", {}).get(
                "default", None
            )
            or None,
            data_s3_path=overrides_template.get("data_s3_path", {}).get("default", None)
            if overrides_template
            else None,
            validation_config=validation_config,
            rft_lambda_arn=overrides_template.get("reward_lambda_arn", {}).get(
                "default", None
            )
            if overrides_template
            else None,
            eval_task=getattr(self, "eval_task", None),
            subtask=overrides_template.get("subtask", {}).get("default", None)
            if overrides_template
            else None,
            processor_config=None
            if (
                getattr(self, "eval_task", None) != EvaluationTask.GEN_QA
                or (
                    overrides_template
                    and overrides_template.get("lambda_arn", {}).get("default", None)
                    is not None
                )
            )
            else {
                "lambda_arn": overrides_template.get("lambda_arn", {}).get(
                    "default", None
                )
                if overrides_template
                else None,
                "lambda_type": overrides_template.get("lambda_type", {}).get(
                    "default", None
                )
                if overrides_template
                else None,
                "preprocessing": overrides_template.get("preprocessing", {}).get(
                    "default", None
                )
                if overrides_template
                else None,
                "postprocessing": overrides_template.get("postprocessing", {}).get(
                    "default", None
                )
                if overrides_template
                else None,
                "aggregation": overrides_template.get("aggregation", {}).get(
                    "default", None
                )
                if overrides_template
                else None,
            },
            rl_env_config=None
            if getattr(self, "eval_task", None) != EvaluationTask.RFT_EVAL
            else {
                "reward_lambda_arn": overrides_template.get(
                    "reward_lambda_arn", {}
                ).get("default", None)
                if overrides_template
                else None
            },
        )

        # Serialize the generated recipe to YAML
        final_recipe_str = yaml.dump(
            final_recipe_dict, default_flow_style=False, sort_keys=False, width=120
        )

        # Save recipe to a file
        output_recipe_path = self._generate_recipe_path(output_recipe_path).path
        os.makedirs(os.path.dirname(output_recipe_path), exist_ok=True)
        with open(output_recipe_path, "w") as f:
            f.write(final_recipe_str)
        logger.info(
            f"Successfully generated recipe and saved it to '{output_recipe_path}'"
        )

        # Return resolved config so RuntimeManager has the information it needs to start a job
        return (
            output_recipe_path,
            overrides_template.get("output_s3_path", {}).get("default", None)
            if overrides_template
            else None,
            overrides_template.get("data_s3_path", {}).get("default", None)
            if overrides_template
            else None,
            image_uri,
        )
