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
Main entrypoint for customizing and training Nova models.

This module provides the NovaModelCustomizer class which orchestrates the training process.
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, cast

import boto3

from amzn_nova_customization_sdk.manager.runtime_manager import (
    JobConfig,
    RuntimeManager,
    SMHPRuntimeManager,
    SMTJRuntimeManager,
)
from amzn_nova_customization_sdk.model.model_config import (
    REGION_TO_ESCROW_ACCOUNT_MAPPING,
    DeploymentResult,
    EndpointInfo,
)
from amzn_nova_customization_sdk.model.model_enums import (
    SUPPORTED_DATAMIXING_METHODS,
    DeploymentMode,
    DeployPlatform,
    Model,
    Platform,
    TrainingMethod,
)
from amzn_nova_customization_sdk.model.nova_model_customizer_util import (
    requires_custom_eval_data,
    resolve_model_checkpoint_path,
    set_output_s3_path,
)
from amzn_nova_customization_sdk.model.result import (
    EvaluationResult,
    SMHPEvaluationResult,
    SMHPTrainingResult,
    SMTJBatchInferenceResult,
    SMTJEvaluationResult,
    SMTJTrainingResult,
    TrainingResult,
)
from amzn_nova_customization_sdk.model.result.inference_result import InferenceResult
from amzn_nova_customization_sdk.monitor.log_monitor import CloudWatchLogMonitor
from amzn_nova_customization_sdk.monitor.mlflow_monitor import MLflowMonitor
from amzn_nova_customization_sdk.recipe.recipe_builder import RecipeBuilder
from amzn_nova_customization_sdk.recipe.recipe_config import EvaluationTask
from amzn_nova_customization_sdk.util.bedrock import (
    BEDROCK_EXECUTION_ROLE_NAME,
    DEPLOYMENT_ARN_NAME,
    check_existing_deployment,
    create_bedrock_execution_role,
    delete_existing_deployment,
    get_required_bedrock_deletion_permissions,
    get_required_bedrock_update_permissions,
    monitor_model_create,
    update_provisioned_throughput_model,
)
from amzn_nova_customization_sdk.util.data_mixing import DataMixing
from amzn_nova_customization_sdk.util.logging import logger
from amzn_nova_customization_sdk.util.platform_util import (
    detect_platform_from_path,
    validate_platform_compatibility,
)
from amzn_nova_customization_sdk.util.recipe import load_recipe_templates
from amzn_nova_customization_sdk.util.sagemaker import get_model_artifacts
from amzn_nova_customization_sdk.validation.validator import Validator


class NovaModelCustomizer:
    # Configs not documented in __init__
    validation_config = None
    generated_recipe_dir = None

    def __init__(
        self,
        model: Model,
        method: TrainingMethod,
        infra: RuntimeManager,
        data_s3_path: Optional[str] = None,
        output_s3_path: Optional[str] = None,
        model_path: Optional[str] = None,
        validation_config: Optional[Dict[str, bool]] = None,
        generated_recipe_dir: Optional[str] = None,
        mlflow_monitor: Optional[MLflowMonitor] = None,
        deployment_mode: DeploymentMode = DeploymentMode.FAIL_IF_EXISTS,
        data_mixing_enabled: bool = False,
        image_uri: Optional[str] = None,
    ):
        """
        Initializes a NovaModelCustomizer instance.

        Args:
            model: The Nova model to be trained (e.g., NOVA_MICRO)
            method: The fine-tuning method (e.g., SFT_LORA, DPO)
            infra: Runtime infrastructure manager (e.g., SMTJRuntimeManager)
            data_s3_path: S3 path to the training dataset
            output_s3_path: Optional S3 path for output artifacts. If not provided, will be auto-generated
            model_path: Optional S3 path for model path
            validation_config: Optional dict to control validation. Keys: 'iam' (bool), 'infra' (bool).
                             Defaults to {'iam': True, 'infra': True}
            generated_recipe_dir: Optional path to save generated recipe YAMLs
            mlflow_monitor: Optional MLflowMonitor instance for experiment tracking
            deployment_mode: Behavior when deploying to existing endpoint name. Options:
                           FAIL_IF_EXISTS (default), UPDATE_IF_EXISTS
            data_mixing: Enable data mixing. Default is False.
            image_uri: Optional custom ECR image URI to override the default training image.
                      Must be in format: <account>.dkr.ecr.<region>.amazonaws.com/<repository>:<tag>

        Raises:
            ValueError: If region is unsupported or model is invalid
        """
        self.job_id: Optional[str] = (
            None  # This will be set after train/eval method invoked
        )
        self.job_started_time: Optional[datetime] = (
            None  # This will be set after train/eval method invoked
        )
        self.cloud_watch_log_monitor: Optional[CloudWatchLogMonitor] = (
            None  # This will be set after get_logs method invoked
        )

        region = boto3.session.Session().region_name or "us-east-1"
        if region not in REGION_TO_ESCROW_ACCOUNT_MAPPING:
            raise ValueError(
                f"Region '{region}' is not supported for Nova training. "
                f"Supported regions are: {list(REGION_TO_ESCROW_ACCOUNT_MAPPING.keys())}"
            )

        self.region = region
        self._model = model
        self._image_uri = image_uri
        self._method = method
        self.infra = infra
        self.data_s3_path = data_s3_path
        self.model_path = model_path
        self.validation_config = validation_config
        self.deployment_mode = deployment_mode
        self._platform = (
            Platform.SMTJ
            if isinstance(self.infra, SMTJRuntimeManager)
            else Platform.SMHP
        )

        self.instance_type = self.infra.instance_type

        self.output_s3_path = set_output_s3_path(
            region=self.region,
            output_s3_path=output_s3_path,
            kms_key_id=self.infra.kms_key_id,
        )

        self.generated_recipe_dir = generated_recipe_dir
        self.mlflow_monitor = mlflow_monitor

        # Initialize data mixing configuration
        self.data_mixing_enabled = data_mixing_enabled
        self.data_mixing = None
        if data_mixing_enabled:
            self.data_mixing = DataMixing()
            self._init_data_mixing(self.model, self.method, self.platform)

    @property
    def model(self) -> Model:
        """Get the model attribute."""
        return self._model

    @model.setter
    def model(self, value: Model) -> None:
        """Set the model attribute and reinitialize data mixing if enabled."""
        if self.data_mixing_enabled:
            self._init_data_mixing(
                model=value, method=self.method, platform=self.platform
            )
            logger.info(
                f"Model changed to {value.name}. Datamixing configs set to default."
            )
        self._model = value

    @property
    def method(self) -> TrainingMethod:
        """Get the method attribute."""
        return self._method

    @method.setter
    def method(self, value: TrainingMethod) -> None:
        """Set the method attribute and reinitialize data mixing if enabled."""
        if self.data_mixing_enabled:
            self._init_data_mixing(
                model=self.model, method=value, platform=self.platform
            )
            logger.info(
                f"Method changed to {value.name}. Datamixing configs set to default."
            )
        self._method = value

    @property
    def platform(self) -> Platform:
        """Get the platform attribute."""
        return self._platform

    @platform.setter
    def platform(self, value: Platform) -> None:
        """Set the platform attribute and reinitialize data mixing if enabled."""
        if self.data_mixing_enabled:
            self._init_data_mixing(model=self.model, method=self.method, platform=value)
            logger.info(
                f"Platform changed to {value.name}. Datamixing configs set to default."
            )
        self._platform = value

    def _init_data_mixing(
        self, model: Model, method: TrainingMethod, platform: Platform
    ) -> None:
        """
        Initialize data mixing configuration.
        """
        if not self.data_mixing_enabled:
            return

        # Data mixing is only supported on HyperPod for certain training methods
        if platform != Platform.SMHP or method not in SUPPORTED_DATAMIXING_METHODS:
            raise ValueError(
                f"Data mixing is only supported for {SUPPORTED_DATAMIXING_METHODS} training methods on SageMaker HyperPod. "
                "Change platform to SMHP or change to a supported training method to use data mixing."
            )

        # Load recipe metadata and templates for non-evaluation methods
        # Eval requires "type" to be passed to load recipes, therefore we load them in evaluate()
        (
            self.recipe_metadata,
            self.recipe_template,
            self.overrides_template,
            self.image_uri,
        ) = load_recipe_templates(
            model=model,
            method=method,
            platform=platform,
            region=self.region,
            data_mixing_enabled=self.data_mixing_enabled,
            instance_type=self.instance_type,
            eval_task=getattr(self, "eval_task", None),
            image_uri_override=self._image_uri,
        )

        # Load default configuration into DataMixing instance if enabled
        if self.data_mixing and self.overrides_template:
            self.data_mixing._load_defaults_from_template(self.overrides_template)

    def get_data_mixing_config(self) -> Dict[str, Any]:
        """
        Get the current data mixing configuration.

        Returns:
            Dictionary containing the data mixing configuration
        """
        if not self.data_mixing:
            return {}
        return self.data_mixing.get_config()

    def set_data_mixing_config(self, config: Dict[str, Any]) -> None:
        """
        Set the data mixing configuration.

        Args:
            config: Dictionary containing the data mixing configuration.
                   Keys should include nova_*_percent fields and customer_data_percent.
                   Any nova_*_percent fields not specified will be set to 0.

        Raises:
            ValueError: If data mixing is not enabled or invalid configuration
        """
        if not self.data_mixing:
            raise ValueError(
                "Data mixing is not enabled for this customizer. Set data_mixing = True in 'NovaModelCustomizer' object."
            )

        self.data_mixing.set_config(config, normalize=True)

    def train(
        self,
        job_name: str,
        recipe_path: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
        rft_lambda_arn: Optional[str] = None,
        validation_data_s3_path: Optional[str] = None,
        dry_run: Optional[bool] = False,
    ) -> TrainingResult | None:
        """
        Generates the recipe YAML, configures runtime, and launches a training job.

        Args:
            job_name: User-defined name for the training job
            recipe_path: Optional path for a YAML recipe file (both S3 and local paths are accepted)
            overrides: Optional dictionary of configuration overrides. Example:
                {
                    'max_epochs': 10,
                    'lr': 5e-6,
                    'warmup_steps': 20,
                    'loraplus_lr_ratio': 16.0,
                    'global_batch_size': 128,
                    'max_length': 16384
                }
            rft_lambda_arn: Optional rewards Lambda ARN, only required for RFT training methods
            validation_data_s3_path: Optional validation S3 path, only applicable for CPT (but is still optional for CPT)
            dry_run: Actually starts a job if False, otherwise just performs validation. Default is False.

        Returns:
            TrainingResult: Metadata object containing job ID, method, start time, and model artifacts
            or None if dry_run is enabled

        Raises:
            Exception: If job execution fails
        """
        # Create RecipeBuilder and let it handle all data mixing logic
        recipe_builder = RecipeBuilder(
            region=self.region,
            job_name=job_name,
            platform=self.platform,
            model=self.model,
            method=self.method,
            instance_type=self.infra.instance_type,
            instance_count=self.infra.instance_count,
            infra=self.infra,
            data_s3_path=self.data_s3_path,
            output_s3_path=self.output_s3_path,
            model_path=self.model_path,
            rft_lambda_arn=rft_lambda_arn,
            validation_data_s3_path=validation_data_s3_path,
            mlflow_monitor=self.mlflow_monitor,
            data_mixing_instance=self.data_mixing,
            image_uri_override=self._image_uri,
        )

        (
            resolved_recipe_path,
            resolved_output_s3_path,
            resolved_data_s3_path,
            resolved_image_uri,
        ) = recipe_builder.build_and_validate(
            overrides=overrides,
            input_recipe_path=recipe_path,
            output_recipe_path=self.generated_recipe_dir,
            validation_config=self.validation_config,
        )

        if dry_run:
            return None

        # Use unique name to actually start the job
        unique_job_name = f"{job_name}-{uuid.uuid4()}"[:63]

        start_time = datetime.now(timezone.utc)
        self.job_started_time = start_time

        self.job_id = self.infra.execute(
            job_config=JobConfig(
                job_name=unique_job_name,
                data_s3_path=resolved_data_s3_path,
                output_s3_path=resolved_output_s3_path,
                image_uri=resolved_image_uri,
                recipe_path=resolved_recipe_path,
                input_s3_data_type="Converse"
                if self.method not in (TrainingMethod.RFT_LORA, TrainingMethod.RFT_FULL)
                else None,
            )
        )

        training_result: TrainingResult
        if self.platform is Platform.SMTJ:
            training_result = SMTJTrainingResult(
                job_id=self.job_id,
                started_time=start_time,
                method=self.method,
                model_artifacts=get_model_artifacts(
                    job_name=unique_job_name,
                    infra=self.infra,
                    output_s3_path=resolved_output_s3_path,
                ),
            )
        else:
            cluster_name = cast(SMHPRuntimeManager, self.infra).cluster_name
            namespace = cast(SMHPRuntimeManager, self.infra).namespace
            training_result = SMHPTrainingResult(
                job_id=self.job_id,
                started_time=start_time,
                method=self.method,
                model_artifacts=get_model_artifacts(
                    job_name=unique_job_name,
                    infra=self.infra,
                    output_s3_path=resolved_output_s3_path,
                ),
                cluster_name=cluster_name,
                namespace=namespace,
            )

        logger.info(f"Started job '{training_result.job_id}'.")
        if training_result.model_artifacts.checkpoint_s3_path:
            logger.info(
                f"Checkpoint S3 path is: {training_result.model_artifacts.checkpoint_s3_path}."
            )
        if training_result.model_artifacts.output_s3_path:
            logger.info(
                f"Output S3 path is: {training_result.model_artifacts.output_s3_path}."
            )

        return training_result

    def evaluate(
        self,
        job_name: str,
        eval_task: EvaluationTask,
        model_path: Optional[str] = None,
        subtask: Optional[str] = None,
        data_s3_path: Optional[str] = None,
        recipe_path: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
        processor: Optional[Dict[str, Any]] = None,
        rl_env: Optional[Dict[str, Any]] = None,
        dry_run: Optional[bool] = False,
        job_result: Optional[TrainingResult] = None,
    ) -> EvaluationResult | None:
        """
        Generates the recipe YAML, configures runtime, and launches an evaluation job.

        :param job_name: User-defined name for the evaluation job
        :param eval_task: The evaluation task to be performed, e.g. mmlu
        :param model_path: Optional S3 path for model path
        :param subtask: Optional subtask for evaluation
        :param data_s3_path: Optional S3 URI for the dataset
        :param recipe_path: Optional path for a YAML recipe file (both S3 and local paths are accepted)
        :param overrides: Optional dictionary of configuration overrides for eval job (Inference config). Example:
                {
                    'max_new_tokens': 2048,
                    'top_k': -1,
                    'top_p': 1.0,
                    'temperature': 0,
                    'top_logprobs': 10
                }
        :param processor: Optional, only needed for Bring Your Own Metrics Configuration. Example:
                {
                    'lambda_arn': 'arn:aws:lambda:<region>:<account_id>:function:<function-name>',
                    'preprocessing': { # Optional, default to True if not provided
                        'enabled': True
                    },
                    'postprocessing': { # Optional, default to True if not provided
                        'enabled': True
                    },
                    # Built-in aggregation function (valid options: min, max, average, sum), default to average
                    'aggregation': 'average'
                }
        :param rl_env: Optional, only needed for Bring your own Reinforcement learning environment (RFT Eval) config.
                Example:
                {
                    'reward_lambda_arn': 'arn:aws:lambda:<region>:<account_id>:function:<reward-function-name>'
                }
        :param dry_run: dry_run: Actually starts a job if False, otherwise just performs validation. Default is False.
        :param job_result: Optional TrainingResult object to extract checkpoint path from.
                          If provided and model_path is None, will automatically extract
                          the checkpoint path from the training job's output.
        :return: EvaluationResult: Metadata object containing job ID, start time, and evaluation output path
                 or None if dry_run is enabled
        """

        # Resolve model checkpoint path
        resolved_model_path = resolve_model_checkpoint_path(
            model_path=model_path,
            job_result=job_result,
            customizer_job_id=self.job_id if hasattr(self, "job_id") else None,
            customizer_output_s3_path=self.output_s3_path,
            customizer_model_path=self.model_path,
        )

        if resolved_model_path is None:
            logger.warning(
                f"Could not resolve model checkpoint path for evaluate job! Falling back to base model {self.model}"
            )

        # Validate platform compatibility
        checkpoint_platform = None
        if resolved_model_path and resolved_model_path.startswith("s3://"):
            checkpoint_platform = detect_platform_from_path(resolved_model_path)

        if checkpoint_platform is None:
            if job_result is not None:
                if job_result.model_artifacts.checkpoint_s3_path:
                    checkpoint_platform = detect_platform_from_path(
                        job_result.model_artifacts.checkpoint_s3_path
                    )
            elif self.output_s3_path and self.output_s3_path.startswith("s3://"):
                checkpoint_platform = detect_platform_from_path(self.output_s3_path)

        validate_platform_compatibility(
            checkpoint_platform=checkpoint_platform,
            execution_platform=self.platform,
            checkpoint_source="evaluation model checkpoint",
        )

        # Only use the cached data_s3_path for BYOD eval tasks
        if requires_custom_eval_data(eval_task):
            customizer_data_s3_path = self.data_s3_path
        else:
            logger.info(
                f"{eval_task} does not use custom data, ignoring customizer data_s3_path."
            )
            customizer_data_s3_path = None

        recipe_builder = RecipeBuilder(
            region=self.region,
            job_name=job_name,
            platform=self.platform,
            model=self.model,
            method=self.method,
            instance_type=self.infra.instance_type,
            instance_count=self.infra.instance_count,
            infra=self.infra,
            data_s3_path=data_s3_path or customizer_data_s3_path,
            output_s3_path=self.output_s3_path,
            model_path=resolved_model_path,
            eval_task=eval_task,
            subtask=subtask,
            processor_config=processor,
            rl_env_config=rl_env,
            mlflow_monitor=self.mlflow_monitor,
            image_uri_override=self._image_uri,
        )

        (
            resolved_recipe_path,
            resolved_output_s3_path,
            resolved_data_s3_path,
            resolved_image_uri,
        ) = recipe_builder.build_and_validate(
            overrides=overrides,
            input_recipe_path=recipe_path,
            output_recipe_path=self.generated_recipe_dir,
            validation_config=self.validation_config,
        )

        if dry_run:
            return None

        # Use unique name to actually start the job
        unique_job_name = f"{job_name}-{uuid.uuid4()}"[:63]

        start_time = datetime.now(timezone.utc)
        self.job_started_time = start_time

        self.job_id = self.infra.execute(
            job_config=JobConfig(
                job_name=unique_job_name,
                data_s3_path=resolved_data_s3_path,
                output_s3_path=resolved_output_s3_path,
                image_uri=resolved_image_uri,
                recipe_path=resolved_recipe_path,
                input_s3_data_type="S3Prefix",
            )
        )

        evaluation_result: EvaluationResult
        if self.platform == Platform.SMTJ:
            eval_output_s3_path = f"{resolved_output_s3_path.rstrip('/')}/{self.job_id}/output/output.tar.gz"
            evaluation_result = SMTJEvaluationResult(
                job_id=self.job_id,
                eval_task=eval_task,
                started_time=start_time,
                eval_output_path=eval_output_s3_path,
            )
        else:
            cluster_name = cast(SMHPRuntimeManager, self.infra).cluster_name
            namespace = cast(SMHPRuntimeManager, self.infra).namespace
            eval_output_s3_path = (
                f"{resolved_output_s3_path.rstrip('/')}/{self.job_id}/eval-result/"
            )
            evaluation_result = SMHPEvaluationResult(
                job_id=self.job_id,
                eval_task=eval_task,
                started_time=start_time,
                eval_output_path=eval_output_s3_path,
                cluster_name=cluster_name,
                namespace=namespace,
            )
        logger.info(
            f"Started eval job '{self.job_id}'. Artifacts will be published to {eval_output_s3_path}"
        )

        return evaluation_result

    def deploy(
        self,
        model_artifact_path: Optional[str] = None,
        deploy_platform: DeployPlatform = DeployPlatform.BEDROCK_OD,
        pt_units: Optional[int] = None,
        endpoint_name: Optional[str] = None,
        job_result: Optional[TrainingResult] = None,
        bedrock_execution_role_name: str = BEDROCK_EXECUTION_ROLE_NAME,
    ) -> DeploymentResult:
        """
        Creates a custom model and deploys it to Bedrock.

        Deployment behavior when endpoint already exists is controlled by the deployment_mode
        parameter set during NovaModelCustomizer initialization:
        - FAIL_IF_EXISTS: Raise error (default, safest)
        - UPDATE_IF_EXISTS: Try in-place update, fail if not supported

        Args:
            model_artifact_path: The s3 path to the training escrow bucket. If not provided, will attempt to extract
                                 from job_result or the `job_id` field of the Customizer.
            deploy_platform: The platform to deploy the model to for inference (Bedrock On-Demand or Provisioned Throughput).
            pt_units: Only needed when Bedrock Provisioned Throughput is chosen. The # of PT to purchase.
            endpoint_name: The name of the deployed model's endpoint -- will be auto generated if not given.
            job_result: Optional training job result object to use for extracting checkpoint path and validating job completion.
            bedrock_execution_role_name: Optional IAM execution role name for Bedrock, defaults to BedrockDeployModelExecutionRole. If this role does not exist, it will be created.

        Returns:
            DeploymentResult: Contains the endpoint information as well as the create time of the deployment.

        Raises:
            Exception: When unable to successfully deploy the model, extract checkpoint path, or handle
                      existing endpoint according to deployment_mode setting.
        """
        bedrock_client = boto3.client("bedrock")
        iam_client = boto3.client("iam")

        # Check if we have a model name (endpoint name) else generate one.
        if endpoint_name is None:
            name_format = f"{self.model}-{self.method}-{self.region}".lower()
            endpoint_name = name_format.replace(".", "-").replace("_", "-")

        # Check for existing deployment with same name
        existing_deployment_arn = check_existing_deployment(
            endpoint_name, deploy_platform
        )
        deleted_existing_deployment = False
        should_delete_existing = False
        attempt_pt_update = False

        # Handle cases where the given endpoint name already has an associated deployment
        if existing_deployment_arn:
            if self.deployment_mode == DeploymentMode.FAIL_IF_EXISTS:
                raise Exception(
                    f"Deployment '{endpoint_name}' already exists on platform {deploy_platform}.\n"
                    f"ARN: {existing_deployment_arn}\n"
                    f"Change deployment_mode to override."
                )

            elif self.deployment_mode == DeploymentMode.UPDATE_IF_EXISTS:
                if deploy_platform != DeployPlatform.BEDROCK_PT:
                    raise Exception(
                        f"UPDATE_IF_EXISTS mode is only supported for Provisioned Throughput deployments.\n"
                        f"Current platform: {deploy_platform}\n"
                        f"Use FORCE_REPLACE mode for On-Demand deployments."
                    )
                logger.info(
                    f"UPDATE_IF_EXISTS mode: Will update existing PT deployment '{endpoint_name}' in-place"
                )
                attempt_pt_update = True

            elif self.deployment_mode in [
                DeploymentMode.UPDATE_OR_REPLACE,
                DeploymentMode.FORCE_REPLACE,
            ]:
                # For PT deployments, try in-place update first (unless FORCE_REPLACE)
                if (
                    deploy_platform == DeployPlatform.BEDROCK_PT
                    and self.deployment_mode == DeploymentMode.UPDATE_OR_REPLACE
                ):
                    logger.info(
                        f"UPDATE_OR_REPLACE mode: Will try to update existing PT deployment '{endpoint_name}' in-place"
                    )
                    attempt_pt_update = True

                # Always mark for deletion as fallback (or primary for FORCE_REPLACE/OD)
                if (
                    not attempt_pt_update
                    or self.deployment_mode == DeploymentMode.FORCE_REPLACE
                ):
                    logger.info(
                        f"{self.deployment_mode.value} mode: Will delete existing deployment '{endpoint_name}'"
                    )
                    should_delete_existing = True

        # Consolidated permission validation (if IAM validation enabled)
        if (
            self.validation_config is None or self.validation_config.get("iam", True)
        ) and existing_deployment_arn:
            if attempt_pt_update:
                required_perms = get_required_bedrock_update_permissions(
                    deploy_platform, existing_deployment_arn
                )
                errors: List[str] = []
                Validator._validate_calling_role_permissions(
                    errors, required_perms, infra=None, region_name=self.region
                )
                if errors:
                    if self.deployment_mode == DeploymentMode.UPDATE_IF_EXISTS:
                        raise Exception(
                            f"Cannot update existing PT deployment '{endpoint_name}': Missing permissions.\n"
                            f"{'; '.join(errors)}\n"
                            f"Please ensure your role has the necessary Bedrock update permissions."
                        )
                    else:
                        # UPDATE_OR_REPLACE: fall back to delete
                        logger.warning(
                            f"Missing update permissions, will fall back to delete/recreate"
                        )
                        attempt_pt_update = False
                        should_delete_existing = True

            if should_delete_existing:
                required_perms = get_required_bedrock_deletion_permissions(
                    deploy_platform, existing_deployment_arn
                )
                errors = []
                Validator._validate_calling_role_permissions(
                    errors, required_perms, infra=None, region_name=self.region
                )
                if errors:
                    raise Exception(
                        f"Cannot delete existing deployment '{endpoint_name}': Missing permissions.\n"
                        f"{'; '.join(errors)}\n"
                        f"Please ensure your role has the necessary Bedrock deletion permissions."
                    )

        # Resolve checkpoint path
        model_artifact_path = resolve_model_checkpoint_path(
            model_path=model_artifact_path,
            job_result=job_result,
            customizer_job_id=self.job_id if hasattr(self, "job_id") else None,
            customizer_output_s3_path=self.output_s3_path,
            customizer_model_path=self.model_path,
            fail_on_error=True,
        )

        # TODO: If given a job ID, check the status before creating the model. If the job isn't completed, tell the user.
        # TODO: If a user already has an arn of a custom model, they should be able to directly deploy it.

        # Check if a Bedrock IAM execution role exists, if not, create one.
        try:
            bedrock_execution_role_arn = create_bedrock_execution_role(
                iam_client, bedrock_execution_role_name
            )["Role"]["Arn"]
        except Exception as e:
            raise Exception(
                f"Failed to find or create the Bedrock IAM Execution Role: {str(e)}"
            )

        model_name = None

        modelKmsKeyArn = None

        try:
            logger.info(f"Creating custom model for endpoint '{endpoint_name}'...")
            model_name = f"{endpoint_name}-{uuid.uuid4()}"[:63]
            if self.infra.kms_key_id:
                sts_client = boto3.client("sts")
                account_id = sts_client.get_caller_identity()["Account"]
                modelKmsKeyArn = f"arn:aws:kms:{self.region}:{account_id}:key/{self.infra.kms_key_id}"
                model = bedrock_client.create_custom_model(
                    modelName=model_name,
                    modelSourceConfig={"s3DataSource": {"s3Uri": model_artifact_path}},
                    roleArn=bedrock_execution_role_arn,
                    modelKmsKeyArn=modelKmsKeyArn,
                )
            else:
                model = bedrock_client.create_custom_model(
                    modelName=model_name,
                    modelSourceConfig={"s3DataSource": {"s3Uri": model_artifact_path}},
                    roleArn=bedrock_execution_role_arn,
                )
        except Exception as e:
            raise Exception(
                f"Failed to create model {model_name} for endpoint {endpoint_name}: {e}"
            )

        # Monitor the model's creation, updating the time stamp every few seconds until the model is created/set as 'active'.
        try:
            monitor_model_create(bedrock_client, model, endpoint_name)
        except Exception as e:
            raise Exception(
                f"Failed to create deployment {endpoint_name} for model {model['modelArn']}: {e}"
            )

        # Delete existing deployment if needed (after model creation succeeds)
        if existing_deployment_arn and should_delete_existing:
            try:
                delete_existing_deployment(
                    existing_deployment_arn, deploy_platform, endpoint_name
                )
                deleted_existing_deployment = True
            except Exception as e:
                raise Exception(
                    f"Failed to create deployment {endpoint_name} for model {model['modelArn']}: Could not delete existing deployment: {e}"
                )

        # Handle deployment creation or update
        deployment = None
        deployment_arn = None
        pt_update_error = None

        # Try PT update if applicable
        if attempt_pt_update and existing_deployment_arn:
            try:
                update_provisioned_throughput_model(
                    existing_deployment_arn, model["modelArn"], endpoint_name
                )
                deployment_arn = existing_deployment_arn
                logger.info(
                    f"Successfully updated existing PT deployment '{endpoint_name}'"
                )
            except Exception as e:
                pt_update_error = str(e)
                if self.deployment_mode == DeploymentMode.UPDATE_IF_EXISTS:
                    raise Exception(
                        f"Failed to create deployment {endpoint_name} for model {model['modelArn']}: {e}"
                    )
                else:
                    # UPDATE_OR_REPLACE: fall back to delete/recreate
                    logger.warning(
                        f"PT update failed, falling back to delete/recreate: {e}"
                    )
                    should_delete_existing = True
                    attempt_pt_update = False

        # Create new deployment if pt update failed or wasn't attempted
        if deployment_arn is None:
            # Delete existing deployment if needed and not already done
            if (
                existing_deployment_arn
                and should_delete_existing
                and not deleted_existing_deployment
            ):
                try:
                    delete_existing_deployment(
                        existing_deployment_arn, deploy_platform, endpoint_name
                    )
                    deleted_existing_deployment = True
                except Exception as e:
                    error_msg = f"Failed to create deployment {endpoint_name} for model {model['modelArn']}: Could not delete existing deployment: {e}"
                    if pt_update_error:
                        error_msg += f". Previous PT update error: {pt_update_error}"
                    raise Exception(error_msg)

            try:
                logger.info(f"Creating deployment for endpoint '{endpoint_name}'...")
                if deploy_platform == DeployPlatform.BEDROCK_PT:
                    deployment = bedrock_client.create_provisioned_model_throughput(
                        modelUnits=pt_units,
                        provisionedModelName=endpoint_name,
                        modelId=model["modelArn"],
                    )
                    deployment_arn = deployment[
                        DEPLOYMENT_ARN_NAME.get(deploy_platform)
                    ]
                elif deploy_platform == DeployPlatform.BEDROCK_OD:
                    deployment = bedrock_client.create_custom_model_deployment(
                        modelDeploymentName=endpoint_name,
                        modelArn=model["modelArn"],
                    )
                    deployment_arn = deployment[
                        DEPLOYMENT_ARN_NAME.get(deploy_platform)
                    ]
                else:
                    raise ValueError(
                        f"Platform '{deploy_platform}' is not supported for Nova training. "
                        f"Supported platforms are: {list(DeployPlatform)}"
                    )
            except Exception as e:
                raise Exception(
                    f"Failed to create deployment {endpoint_name} for model {model['modelArn']}: {e}"
                )

        # Creates EndpointInfo and DeploymentResult objects.
        create_time = datetime.now(timezone.utc)
        self.job_started_time = create_time
        endpoint = EndpointInfo(
            platform=deploy_platform,
            endpoint_name=endpoint_name,
            uri=deployment_arn,
            model_artifact_path=self.output_s3_path,
        )
        result = DeploymentResult(endpoint=endpoint, created_at=create_time)

        # Log message to the user with information about the deployment.
        logger.info(
            f"\nSuccessfully started deploying {endpoint.endpoint_name}: \n"
            f"- Platform: {endpoint.platform}:\n"
            f"- ARN: {endpoint.uri}\n"
            f"- Created: {result.created_at.strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
            f"- ETA: Deployment should be completed in about 30-45 minutes"
        )
        return result

    def predict(self):
        pass

    def batch_inference(
        self,
        job_name: str,
        input_path: str,
        output_s3_path: str,
        model_path: Optional[str] = None,
        recipe_path: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
        dry_run: Optional[bool] = False,
        job_result: Optional[TrainingResult] = None,
    ) -> InferenceResult | None:
        """
        Launches a batch inference job on a trained model.

        :param job_name: Name for the batch inference job
        :param input_path: S3 path to input data for inference
        :param output_s3_path: S3 path for inference outputs
        :param model_path: Optional S3 path to the model
        :param recipe_path: Optional path for a YAML recipe file
        :param overrides: Optional configuration overrides for inference
        :param dry_run: Actually starts a job if False, otherwise just performs validation. Default is False.
        :param job_result: Optional TrainingResult object to extract checkpoint path from.
                          If provided and model_path is None, will automatically extract
                          the checkpoint path from the training job's output.
        :return: InferenceResult or None if dry_run is enabled
        """

        # Resolve model checkpoint path
        resolved_model_path = resolve_model_checkpoint_path(
            model_path=model_path,
            job_result=job_result,
            customizer_job_id=self.job_id if hasattr(self, "job_id") else None,
            customizer_output_s3_path=self.output_s3_path,
            customizer_model_path=self.model_path,
        )

        if resolved_model_path is None:
            logger.warning(
                f"Could not resolve model checkpoint path for evaluate job! Falling back to base model {self.model}"
            )

        # Validate platform compatibility
        checkpoint_platform = None
        if resolved_model_path and resolved_model_path.startswith("s3://"):
            checkpoint_platform = detect_platform_from_path(resolved_model_path)

        if checkpoint_platform is None and job_result is not None:
            if job_result.model_artifacts.checkpoint_s3_path:
                checkpoint_platform = detect_platform_from_path(
                    job_result.model_artifacts.checkpoint_s3_path
                )

        if (
            checkpoint_platform is None
            and self.model_path
            and self.model_path.startswith("s3://")
        ):
            checkpoint_platform = detect_platform_from_path(self.model_path)

        validate_platform_compatibility(
            checkpoint_platform=checkpoint_platform,
            execution_platform=self.platform,
            checkpoint_source="batch inference model checkpoint",
        )

        recipe_builder = RecipeBuilder(
            region=self.region,
            job_name=job_name,
            platform=self.platform,
            model=self.model,
            method=self.method,
            instance_type=self.infra.instance_type,
            instance_count=self.infra.instance_count,
            infra=self.infra,
            data_s3_path=input_path,
            output_s3_path=output_s3_path or self.output_s3_path,
            model_path=resolved_model_path,
            mlflow_monitor=self.mlflow_monitor,
            image_uri_override=self._image_uri,
        )

        (
            resolved_recipe_path,
            resolved_output_s3_path,
            resolved_data_s3_path,
            resolved_image_uri,
        ) = recipe_builder.build_and_validate(
            overrides=overrides,
            input_recipe_path=recipe_path,
            output_recipe_path=self.generated_recipe_dir,
            validation_config=self.validation_config,
        )

        if dry_run:
            return None

        # Use unique name to actually start the job
        unique_job_name = f"{job_name}-{uuid.uuid4()}"[:63]

        start_time = datetime.now(timezone.utc)
        self.job_started_time = start_time

        self.job_id = self.infra.execute(
            job_config=JobConfig(
                job_name=unique_job_name,
                data_s3_path=resolved_data_s3_path,
                output_s3_path=resolved_output_s3_path,
                image_uri=resolved_image_uri,
                recipe_path=resolved_recipe_path,
                input_s3_data_type="S3Prefix",
            )
        )

        # TODO: Implement for SMHP jobs. I'm not sure how different the infrastructure is.
        inference_output_s3_path = (
            f"{resolved_output_s3_path.rstrip('/')}/{job_name}/output/output.tar.gz"
        )
        batch_inference_result = SMTJBatchInferenceResult(
            job_id=self.job_id,
            started_time=start_time,
            inference_output_path=inference_output_s3_path,
        )
        logger.info(
            f"Started batch inference job '{self.job_id}'. \nArtifacts will be published to {inference_output_s3_path}.\n"
            f"After opening the tar file, look for {recipe_builder.job_name}/eval_results/inference_output.jsonl."
        )
        return batch_inference_result

    def get_logs(
        self,
        limit: Optional[int] = None,
        start_from_head: bool = False,
        end_time: Optional[int] = None,
    ):
        if self.job_id and self.job_started_time:
            kwargs = {}
            if self.platform == Platform.SMHP:
                kwargs["cluster_name"] = cast(
                    SMHPRuntimeManager, self.infra
                ).cluster_name
                kwargs["namespace"] = cast(SMHPRuntimeManager, self.infra).namespace
            self.cloud_watch_log_monitor = (
                self.cloud_watch_log_monitor
                or CloudWatchLogMonitor(
                    job_id=self.job_id,
                    platform=self.platform,
                    started_time=int(self.job_started_time.timestamp() * 1000),
                    **kwargs,
                )
            )
            self.cloud_watch_log_monitor.show_logs(
                limit=limit, start_from_head=start_from_head, end_time=end_time
            )
        else:
            print(
                "No job_id and job_started_time found for this model, please call .train() or .evaluate() first."
            )

    def monitor_metrics(self):
        pass
