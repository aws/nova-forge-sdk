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
from typing import Any, Dict, Optional, cast

import boto3

import amzn_nova_customization_sdk.recipe_config.v_one.sft_config as v1_sft
import amzn_nova_customization_sdk.recipe_config.v_two.rft_config_smhp as rft_smhp
import amzn_nova_customization_sdk.recipe_config.v_two.rft_config_smtj as rft_smtj
import amzn_nova_customization_sdk.recipe_config.v_two.sft_config as v2_sft
from amzn_nova_customization_sdk.manager.runtime_manager import (
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
    DeployPlatform,
    Model,
    Platform,
    TrainingMethod,
    Version,
)
from amzn_nova_customization_sdk.model.nova_model_customizer_util import (
    set_image_uri,
    set_output_s3_path,
)
from amzn_nova_customization_sdk.model.result import (
    BaseJobResult,
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
from amzn_nova_customization_sdk.recipe_builder.base_recipe_builder import (
    BaseRecipeBuilder,
)
from amzn_nova_customization_sdk.recipe_builder.batch_inference_recipe_builder import (
    BatchInferenceRecipeBuilder,
)
from amzn_nova_customization_sdk.recipe_builder.eval_recipe_builder import (
    EvalRecipeBuilder,
)
from amzn_nova_customization_sdk.recipe_builder.rft_recipe_builder import (
    RFTRecipeBuilder,
)
from amzn_nova_customization_sdk.recipe_builder.sft_recipe_builder import (
    SFTRecipeBuilder,
)
from amzn_nova_customization_sdk.recipe_config.base_recipe_config import (
    BaseRecipeConfig,
)
from amzn_nova_customization_sdk.recipe_config.eval_config import (
    EvalRecipeConfig,
    EvaluationTask,
)
from amzn_nova_customization_sdk.util.bedrock import (
    BEDROCK_EXECUTION_ROLE_NAME,
    DEPLOYMENT_ARN_NAME,
    create_bedrock_execution_role,
    monitor_model_create,
)
from amzn_nova_customization_sdk.util.logging import logger
from amzn_nova_customization_sdk.util.recipe import resolve_overrides
from amzn_nova_customization_sdk.util.sagemaker import get_model_artifacts


class NovaModelCustomizer:
    # Configs not documented in __init__
    validation_config = None
    generated_recipe_dir = None

    def __init__(
        self,
        model: Model,
        method: TrainingMethod,
        infra: RuntimeManager,
        data_s3_path: str,
        output_s3_path: Optional[str] = None,
        model_path: Optional[str] = None,
        validation_config: Optional[Dict[str, bool]] = None,
        generated_recipe_dir: Optional[str] = None,
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
        self.model = model
        self.method = method
        self.infra = infra
        self.data_s3_path = data_s3_path
        self.model_path = model_path
        self.validation_config = validation_config
        self.platform = (
            Platform.SMTJ
            if isinstance(self.infra, SMTJRuntimeManager)
            else Platform.SMHP
        )

        self.output_s3_path = set_output_s3_path(
            region=self.region, output_s3_path=output_s3_path
        )
        self.image_uri = set_image_uri(
            region=self.region,
            method=self.method,
            version=self.model.version,
            infra=self.infra,
        )

        self.generated_recipe_dir = generated_recipe_dir

    def _validate_model_config(self, model=None, model_path=None):
        model = model or self.model
        model_path = model_path or self.model_path

        if model_path == model.model_path:
            return True
        else:
            try:
                s3_client = boto3.client("s3", region_name=self.region)
                # Parse S3 URI
                if not model_path.startswith("s3://"):
                    raise ValueError(f"Model path must be an S3 URI, got: {model_path}")

                # Remove s3:// prefix and split bucket/key
                s3_path = model_path[5:]  # Remove "s3://"
                bucket, key = s3_path.split("/", 1)

                # Check if object exists
                s3_client.head_object(Bucket=bucket, Key=key)
                return True

            except Exception as e:
                if "NoSuchBucket" in str(e):
                    raise ValueError(
                        f"S3 bucket {bucket} does not exist when validating model checkpoint {model_path}: {str(e)}"
                    )
                elif "NoSuchKey" in str(e):
                    raise ValueError(
                        f"Model checkpoint does not exist at {model_path}: {str(e)}"
                    )
                elif "AccessDenied" in str(e):
                    raise ValueError(
                        f"Access denied when validating model checkpoint {model_path}: {str(e)}"
                    )
                else:
                    raise ValueError(
                        f"Cannot validate model checkpoint {model_path}: {str(e)}"
                    )

    def _prepare_recipe_builder(
        self,
        job_name: str,
        recipe_path: Optional[str] = None,
        recipe_class: type = BaseRecipeConfig,
        overrides: Optional[Dict[str, Any]] = None,
        rft_lambda_arn: Optional[str] = None,
    ) -> BaseRecipeBuilder:
        """
        Create recipe builder based on inputs and training configuration.

        Returns:
            BaseRecipeBuilder: Subclass of the base recipe builder for the specified method
        """
        overrides = resolve_overrides(
            recipe_path=recipe_path, recipe_class=recipe_class, overrides=overrides
        )
        job_name = overrides.get("name") or job_name
        platform = self.platform
        model = (
            Model.from_model_type(overrides["model_type"])
            if "model_type" in overrides
            else self.model
        )
        method = self.method
        instance_type = self.infra.instance_type
        instance_count = overrides.get("replicas") or self.infra.instance_count
        data_s3_path = overrides.get("data_s3_path") or self.data_s3_path
        output_s3_path = overrides.get("output_s3_path") or self.output_s3_path
        model_path = (
            overrides.get("model_name_or_path") or self.model_path or model.model_path
        )
        rft_lambda_arn = (
            overrides.get("reward_lambda_arn")
            or overrides.get("lambda_arn")
            or rft_lambda_arn
        )

        self._validate_model_config(model, model_path)

        if (
            rft_lambda_arn is not None
            and method is not TrainingMethod.RFT
            and method is not TrainingMethod.RFT_LORA
        ):
            logger.info(
                f"rft_lambda_arn is only used for RFT training, value will be ignored."
            )

        # TODO: Abstract away the training method
        recipe_builder: BaseRecipeBuilder
        if self.method in (TrainingMethod.SFT_LORA, TrainingMethod.SFT_FULLRANK):
            recipe_builder = SFTRecipeBuilder(
                job_name=job_name,
                platform=platform,
                model=model,
                method=method,
                instance_type=instance_type,
                instance_count=instance_count,
                data_s3_path=data_s3_path,
                output_s3_path=output_s3_path,
                overrides=overrides or {},
                infra=self.infra,
                model_path=model_path,
            )
        elif self.method in (TrainingMethod.RFT_LORA, TrainingMethod.RFT):
            recipe_builder = RFTRecipeBuilder(
                job_name=job_name,
                platform=platform,
                model=model,
                method=method,
                instance_type=instance_type,
                instance_count=instance_count,
                data_s3_path=data_s3_path,
                output_s3_path=output_s3_path,
                rft_lambda_arn=rft_lambda_arn,
                overrides=overrides or {},
                infra=self.infra,
                model_path=model_path,
            )
        else:
            raise ValueError(f"{method.value} is not yet supported.")

        # Set this independently to avoid duplicate code across RecipeBuilder subclasses
        recipe_builder.generated_recipe_dir = self.generated_recipe_dir

        return recipe_builder

    def validate(
        self,
        job_name: str,
        recipe_path: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
        rft_lambda_arn: Optional[str] = None,
    ) -> None:
        """
        Validate training configuration without side effects.
        Has the same input args as `train`

        Args:
            job_name: User-defined name for the training job
            recipe_path: Optional path for a YAML recipe file (both S3 and local paths are accepted)
            overrides: Optional dictionary of configuration overrides
            rft_lambda_arn: Optional rewards Lambda ARN, only used for RFT training methods

        Raises:
            ValueError: If the configuration is invalid
        """
        recipe_class: type = BaseRecipeConfig
        if self.method is TrainingMethod.EVALUATION:
            recipe_class = EvalRecipeConfig
        elif self.method in (TrainingMethod.SFT_LORA, TrainingMethod.SFT_FULLRANK):
            recipe_class = (
                v1_sft.SFTRecipeConfig
                if self.model.version is Version.ONE
                else v2_sft.SFTRecipeConfig
            )
        elif self.method in (TrainingMethod.RFT_LORA, TrainingMethod.RFT):
            recipe_class = (
                rft_smtj.RFTRecipeConfig
                if self.platform is Platform.SMTJ
                else rft_smhp.RFTRecipeConfig
            )

        recipe_builder = self._prepare_recipe_builder(
            job_name=job_name,
            recipe_path=recipe_path,
            recipe_class=recipe_class,
            overrides=overrides,
            rft_lambda_arn=rft_lambda_arn,
        )

        recipe_builder._validate_user_input(validation_config=self.validation_config)

    def train(
        self,
        job_name: str,
        recipe_path: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
        rft_lambda_arn: Optional[str] = None,
    ) -> TrainingResult:
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
            rft_lambda_arn: Optional rewards Lambda ARN, only used for RFT training methods

        Returns:
            TrainingResult: Metadata object containing job ID, method, start time, and model artifacts

        Raises:
            Exception: If job execution fails
        """
        recipe_class: type
        if self.method in (TrainingMethod.SFT_LORA, TrainingMethod.SFT_FULLRANK):
            recipe_class = (
                v1_sft.SFTRecipeConfig
                if self.model.version is Version.ONE
                else v2_sft.SFTRecipeConfig
            )
        elif self.method in (TrainingMethod.RFT_LORA, TrainingMethod.RFT):
            recipe_class = (
                rft_smtj.RFTRecipeConfig
                if self.platform is Platform.SMTJ
                else rft_smhp.RFTRecipeConfig
            )
        else:
            raise ValueError(
                f"Training method {self.method} not supported in .train() function"
            )

        recipe_builder = self._prepare_recipe_builder(
            job_name=job_name,
            recipe_path=recipe_path,
            recipe_class=recipe_class,
            overrides=overrides,
            rft_lambda_arn=rft_lambda_arn,
        )

        # Validate and build the recipe
        job_name = f"{recipe_builder.job_name}-{uuid.uuid4()}"[:63]

        with recipe_builder.build(validation_config=self.validation_config) as recipe:
            start_time = datetime.now(timezone.utc)
            self.job_started_time = start_time

            self.job_id = self.infra.execute(
                job_name=job_name,
                data_s3_path=recipe_builder.data_s3_path,
                output_s3_path=recipe_builder.output_s3_path,
                image_uri=self.image_uri,
                recipe=recipe.path,
                input_s3_data_type="Converse"
                if recipe_builder.method
                not in (TrainingMethod.RFT_LORA, TrainingMethod.RFT)
                else None,
            )

        training_result: TrainingResult
        if self.platform is Platform.SMTJ:
            training_result = SMTJTrainingResult(
                job_id=self.job_id,
                started_time=start_time,
                method=recipe_builder.method,
                model_artifacts=get_model_artifacts(
                    job_name=job_name,
                    infra=self.infra,
                    output_s3_path=recipe_builder.output_s3_path,
                ),
            )
        else:
            cluster_name = cast(SMHPRuntimeManager, self.infra).cluster_name
            namespace = cast(SMHPRuntimeManager, self.infra).namespace
            training_result = SMHPTrainingResult(
                job_id=self.job_id,
                started_time=start_time,
                method=recipe_builder.method,
                model_artifacts=get_model_artifacts(
                    job_name=job_name,
                    infra=self.infra,
                    output_s3_path=recipe_builder.output_s3_path,
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
    ) -> EvaluationResult:
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
        :return: BaseJobResult: Metadata object containing job ID, start time, and evaluation output path
        """
        overrides = resolve_overrides(
            recipe_path=recipe_path, recipe_class=EvalRecipeConfig, overrides=overrides
        )
        job_name = overrides.get("name") or job_name
        platform = self.platform
        model = (
            Model.from_model_type(overrides["model_type"])
            if "model_type" in overrides
            else self.model
        )
        model_path = (
            overrides.get("model_name_or_path") or model_path or self.model.model_path
        )
        instance_type = self.infra.instance_type
        instance_count = overrides.get("replicas") or self.infra.instance_count
        data_s3_path = overrides.get("data_s3_path") or data_s3_path
        output_s3_path = overrides.get("output_s3_path") or self.output_s3_path
        eval_task = overrides.get("task") or eval_task
        subtask = overrides.get("subtask") or subtask

        self._validate_model_config(model, model_path)

        recipe_builder = EvalRecipeBuilder(
            job_name=job_name,
            platform=platform,
            model=model,
            model_path=model_path,
            instance_type=instance_type,
            instance_count=instance_count,
            data_s3_path=data_s3_path,
            output_s3_path=output_s3_path,
            eval_task=eval_task,
            subtask=subtask,
            overrides=overrides,
            processor_config=processor,
            rl_env_config=rl_env,
        )
        recipe_builder.generated_recipe_dir = self.generated_recipe_dir

        job_name = f"{job_name}-{uuid.uuid4()}"[:63]

        with recipe_builder.build(validation_config=self.validation_config) as recipe:
            start_time = datetime.now(timezone.utc)
            self.job_started_time = start_time

            self.job_id = self.infra.execute(
                job_name=job_name,  # For SMHP, it won't use this field and still use base_job_name in recipe
                data_s3_path=data_s3_path,
                output_s3_path=output_s3_path,
                image_uri=self.image_uri,
                recipe=recipe.path,
                input_s3_data_type="S3Prefix",
            )

        evaluation_result: EvaluationResult
        if self.platform == Platform.SMTJ:
            eval_output_s3_path = (
                f"{output_s3_path.rstrip('/')}/{self.job_id}/output/output.tar.gz"
            )
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
                f"{output_s3_path.rstrip('/')}/{self.job_id}/eval-result/"
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
        model_artifact_path: str,
        deploy_platform: DeployPlatform = DeployPlatform.BEDROCK_OD,
        pt_units: Optional[int] = None,
        endpoint_name: Optional[str] = None,
    ) -> DeploymentResult:
        """
        Creates a custom model and deploys it to Bedrock.

        Args:
            model_artifact_path: The s3 path to the training escrow bucket.
            deploy_platform: The platform to deploy the model to for inference (Bedrock On-Demand or Provisioned Throughput).
            pt_units: Only needed when Bedrock Provisioned Throughput is chosen. The # of PT to purchase.
            endpoint_name: The name of the deployed model's endpoint -- will be auto generated if not given.

        Returns:
            DeploymentResult: Contains the endpoint information as well as the create time of the deployment.

        Raises:
            Exception: When unable to successfully deploy the model.
        """
        bedrock_client = boto3.client("bedrock")
        iam_client = boto3.client("iam")

        # Check if we have a model name (endpoint name) else generate one.
        if endpoint_name is None:
            name_format = f"{self.model}-{self.method}-{self.region}".lower()
            endpoint_name = name_format.replace(".", "-").replace("_", "-")

        # TODO: If given a job ID, check the status before creating the model. If the job isn't completed, tell the user.
        # TODO: If a user already has an arn of a custom model, they should be able to directly deploy it.

        # Check if a BedrockDeployModelExecutionRole exists, if not, create one.
        try:
            bedrock_execution_role_arn = create_bedrock_execution_role(
                iam_client, BEDROCK_EXECUTION_ROLE_NAME
            )["Role"]["Arn"]
        except Exception as e:
            raise Exception(
                f"Failed to find or create the BedrockDeployModelExecutionRole: {str(e)}"
            )

        try:
            model = bedrock_client.create_custom_model(
                modelName=f"{endpoint_name}-{uuid.uuid4()}"[:63],
                modelSourceConfig={"s3DataSource": {"s3Uri": model_artifact_path}},
                roleArn=bedrock_execution_role_arn,
            )
        except Exception as e:
            raise Exception(f"Failed to create custom model {endpoint_name}: {e}")

        # Monitor the model's creation, updating the time stamp every few seconds until the model is created/set as 'active'.
        monitor_model_create(bedrock_client, model, endpoint_name)

        # Updates the deployment based on whether PT or OD was selected.
        if deploy_platform == DeployPlatform.BEDROCK_PT:
            deployment = bedrock_client.create_provisioned_model_throughput(
                modelUnits=pt_units,
                provisionedModelName=endpoint_name,
                modelId=model["modelArn"],
            )
        elif deploy_platform == DeployPlatform.BEDROCK_OD:
            deployment = bedrock_client.create_custom_model_deployment(
                modelDeploymentName=endpoint_name,
                modelArn=model["modelArn"],
            )
        else:
            raise ValueError(
                f"Platform '{deploy_platform}' is not supported for Nova training. "
                f"Supported platforms are: {list(DeployPlatform)}"
            )

        # Creates EndpointInfo and DeploymentResult objects.
        create_time = datetime.now(timezone.utc)
        self.job_started_time = create_time
        endpoint = EndpointInfo(
            platform=deploy_platform,
            endpoint_name=endpoint_name,
            uri=deployment[DEPLOYMENT_ARN_NAME.get(deploy_platform)],
            model_artifact_path=self.output_s3_path,
        )
        result = DeploymentResult(endpoint=endpoint, created_at=create_time)

        # Log message to the user with information about the deployment.
        logger.info(
            f"\n✅ Successfully started deploying {endpoint.endpoint_name}: \n"
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
        endpoint: Optional[EndpointInfo] = None,
        recipe_path: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> InferenceResult:
        recipe_class = EvalRecipeConfig

        overrides = resolve_overrides(
            recipe_path=recipe_path, recipe_class=recipe_class, overrides=overrides
        )
        job_name = overrides.get("name") or job_name
        platform = self.platform
        model = (
            Model.from_model_type(overrides["model_type"])
            if "model_type" in overrides
            else self.model
        )
        model_path = (
            overrides.get("model_name_or_path") or model_path or self.model.model_path
        )
        instance_type = self.infra.instance_type
        instance_count = overrides.get("replicas") or self.infra.instance_count
        input_path = overrides.get("data_s3_path") or input_path
        output_s3_path = overrides.get("output_s3_path") or output_s3_path

        self._validate_model_config(model, model_path)

        # TODO: P1 - Add functionality for the 'endpoint' parameter (SM or Bedrock batch inference).
        recipe_builder = BatchInferenceRecipeBuilder(
            job_name=job_name,
            platform=platform,
            model=model,
            model_path=model_path,
            instance_type=instance_type,
            instance_count=instance_count,
            data_s3_path=input_path,
            output_s3_path=output_s3_path,
            overrides=overrides,
        )
        recipe_builder.generated_recipe_dir = self.generated_recipe_dir

        recipe = recipe_builder.build(validation_config=self.validation_config)
        job_name = f"{job_name}-{uuid.uuid4()}"[:63]

        with recipe_builder.build(validation_config=self.validation_config) as recipe:
            start_time = datetime.now(timezone.utc)
            self.job_started_time = start_time

            job_id: str = self.infra.execute(
                job_name=job_name,
                data_s3_path=input_path,
                output_s3_path=output_s3_path,
                image_uri=self.image_uri,
                recipe=recipe.path,
                input_s3_data_type="S3Prefix",
            )

        # TODO: Implement for SMHP jobs. I'm not sure how different the infrastructure is.
        inference_output_s3_path = (
            f"{output_s3_path.rstrip('/')}/{job_name}/output/output.tar.gz"
        )
        batch_inference_result = SMTJBatchInferenceResult(
            job_id=job_id,
            started_time=start_time,
            inference_output_path=inference_output_s3_path,
        )
        logger.info(
            f"Started batch inference job '{job_id}'. \nArtifacts will be published to {inference_output_s3_path}.\n"
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
