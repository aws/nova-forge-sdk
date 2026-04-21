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
"""ForgeTrainer — owns the training workflow."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, Optional, cast

import boto3

from amzn_nova_forge.core.constants import DEFAULT_REGION, SUPPORTED_DATAMIXING_METHODS
from amzn_nova_forge.core.enums import Model, Platform, TrainingMethod
from amzn_nova_forge.core.job_cache import (
    build_cache_context,
    load_existing_result,
    persist_result,
)
from amzn_nova_forge.core.result import (
    BedrockTrainingResult,
    SMHPTrainingResult,
    SMTJTrainingResult,
    TrainingResult,
)
from amzn_nova_forge.core.runtime import RuntimeManager
from amzn_nova_forge.core.types import (
    ForgeConfig,
    JobConfig,
    ModelArtifacts,
    validate_region,
)
from amzn_nova_forge.manager.runtime_manager import SMHPRuntimeManager
from amzn_nova_forge.model.nova_model_customizer_util import set_output_s3_path
from amzn_nova_forge.monitor.log_monitor import CloudWatchLogMonitor
from amzn_nova_forge.recipe.recipe_builder import RecipeBuilder
from amzn_nova_forge.telemetry import Feature, _telemetry_emitter
from amzn_nova_forge.util.data_mixing import DataMixing
from amzn_nova_forge.util.data_utils import is_multimodal_data
from amzn_nova_forge.util.logging import logger
from amzn_nova_forge.util.recipe import load_recipe_templates
from amzn_nova_forge.util.sagemaker import get_model_artifacts
from amzn_nova_forge.validation.endpoint_validator import is_sagemaker_arn
from amzn_nova_forge.validation.validator import validate_rft_lambda_name

if TYPE_CHECKING:
    from amzn_nova_forge.rft_multiturn import RFTMultiturnInfrastructure


class ForgeTrainer:
    """Encapsulates the training workflow for Nova model customization.

    Configuration is provided in the constructor; ``train()`` accepts only
    per-job parameters.  Follows the RecipeBuilder pattern — no shared
    mutable state between calls.
    """

    def __init__(
        self,
        model: Model,
        method: TrainingMethod,
        infra: RuntimeManager,
        training_data_s3_path: Optional[str] = None,
        model_s3_path: Optional[str] = None,
        data_mixing_enabled: bool = False,
        holdout_data_s3_path: Optional[str] = None,
        config: Optional[ForgeConfig] = None,
        region: Optional[str] = None,
        is_multimodal: Optional[bool] = None,
    ) -> None:
        self.model = model
        self.method = method
        self.infra = infra
        self.training_data_s3_path = training_data_s3_path
        self.model_s3_path = model_s3_path
        self.holdout_data_s3_path = holdout_data_s3_path
        self._config = config or ForgeConfig()

        self.region = region or boto3.session.Session().region_name or DEFAULT_REGION
        validate_region(self.region)

        self._platform = infra.platform

        if self._platform == Platform.SMTJServerless and self.model_s3_path is not None:
            if not is_sagemaker_arn(self.model_s3_path):
                raise ValueError(
                    f"For SMTJServerless, model_path must be a SageMaker model package ARN, "
                    f"got: '{self.model_s3_path}'."
                )
        if self._platform == Platform.BEDROCK and self.model_s3_path is not None:
            logger.warning(
                "model_path is not used for Bedrock platform. "
                "To specify a base model, pass base_model_identifier to "
                "BedrockRuntimeManager constructor instead."
            )

        self.output_s3_path = set_output_s3_path(
            region=self.region,
            output_s3_path=self._config.output_s3_path,
            kms_key_id=self.infra.kms_key_id,
        )

        # is_multimodal resolution (matches original NovaModelCustomizer lines 230-252)
        if not data_mixing_enabled:
            if is_multimodal is not None:
                logger.warning("is_multimodal is ignored because data_mixing_enabled=False.")
            self._is_multimodal = False
        elif is_multimodal is not None:
            self._is_multimodal = is_multimodal
        elif training_data_s3_path:
            self._is_multimodal = is_multimodal_data(training_data_s3_path)
            if self._is_multimodal:
                logger.info(
                    "Multimodal data detected. Using multimodal datamix recipes. "
                    "To skip auto-detection, pass is_multimodal=False."
                )
        else:
            self._is_multimodal = False

        # Data mixing setup (SMHP only, CPT/SFT methods)
        self.data_mixing: Optional[DataMixing] = None
        if data_mixing_enabled:
            if self._platform != Platform.SMHP or method not in SUPPORTED_DATAMIXING_METHODS:
                raise ValueError(
                    f"Data mixing is only supported for {SUPPORTED_DATAMIXING_METHODS} "
                    "training methods on SageMaker HyperPod."
                )
            self.data_mixing = DataMixing()
            (
                _metadata,
                _template,
                overrides_template,
                _image_uri,
            ) = load_recipe_templates(
                model=model,
                method=method,
                platform=self._platform,
                region=self.region,
                data_mixing_enabled=True,
                instance_type=self.infra.instance_type,
                image_uri_override=self._config.image_uri,
                is_multimodal=self._is_multimodal,
            )
            if overrides_template:
                self.data_mixing._load_defaults_from_template(overrides_template)

        # Job caching context
        self._cache_context = build_cache_context(
            self._config,
            model=model,
            method=method,
            data_s3_path=training_data_s3_path,
            model_path=model_s3_path,
            output_s3_path=self.output_s3_path,
            instance_type=infra.instance_type,
            instance_count=infra.instance_count,
        )

    @_telemetry_emitter(
        Feature.TRAINING,
        "train",
        extra_info_fn=lambda self, *args, **kwargs: {
            "method": self.method,
            "model": self.model.value,
            "platform": self._platform,
            "dryRun": kwargs.get("dry_run", False),
        },
    )
    def train(
        self,
        job_name: str,
        recipe_path: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
        rft_lambda_arn: Optional[str] = None,
        dry_run: bool = False,
        rft_multiturn_infra: Optional[RFTMultiturnInfrastructure] = None,
    ) -> Optional[TrainingResult]:
        """Launch a training job.

        Args:
            job_name: User-defined name for the training job.
            recipe_path: Optional path to a YAML recipe file.
            overrides: Optional dictionary of configuration overrides.
            rft_lambda_arn: Optional RFT Lambda ARN. Falls back to infra attribute.
            dry_run: If True, only validate — do not start a job.
            rft_multiturn_infra: Optional RFT multiturn infrastructure (passed
                through to RecipeBuilder for multi-turn RFT workflows).

        Returns:
            TrainingResult on success, None if dry_run is True.
        """
        # Check job cache
        cached = load_existing_result(
            self._cache_context,
            job_name=job_name,
            job_type="train",
            recipe_path=recipe_path,
            overrides=overrides or {},
        )
        if cached:
            logger.info("Returning cached result for '%s'.", job_name)
            return cached  # type: ignore[return-value]

        rft_lambda_arn = rft_lambda_arn or getattr(self.infra, "rft_lambda_arn", None)

        if rft_lambda_arn:
            validate_rft_lambda_name(rft_lambda_arn.split(":")[-1], self._platform)
            logger.info(f"Using reward lambda: {rft_lambda_arn}")

        recipe_builder = RecipeBuilder(
            region=self.region,
            job_name=job_name,
            platform=self._platform,
            model=self.model,
            method=self.method,
            instance_type=self.infra.instance_type,
            instance_count=self.infra.instance_count,
            infra=self.infra,
            data_s3_path=self.training_data_s3_path,
            output_s3_path=self.output_s3_path,
            model_path=self.model_s3_path,
            rft_lambda_arn=rft_lambda_arn,
            validation_data_s3_path=self.holdout_data_s3_path,
            data_mixing_instance=self.data_mixing,
            image_uri_override=self._config.image_uri,
            is_multimodal=self._is_multimodal,
            mlflow_monitor=self._config.mlflow_monitor,
            rft_multiturn_infra=rft_multiturn_infra,
        )

        (
            resolved_recipe_path,
            resolved_output_s3_path,
            resolved_data_s3_path,
            resolved_image_uri,
        ) = recipe_builder.build_and_validate(
            overrides=overrides,
            input_recipe_path=recipe_path,
            output_recipe_path=self._config.generated_recipe_dir,
            validation_config=self._config.validation_config,
        )

        if dry_run:
            return None

        unique_job_name = f"{job_name}-{uuid.uuid4()}"[:63]
        start_time = datetime.now(timezone.utc)

        job_config_params: Dict[str, Any] = {
            "job_name": unique_job_name,
            "data_s3_path": resolved_data_s3_path,
            "output_s3_path": resolved_output_s3_path,
            "image_uri": resolved_image_uri,
            "recipe_path": resolved_recipe_path,
            "rft_lambda_arn": rft_lambda_arn,
            "validation_data_s3_path": self.holdout_data_s3_path,
            "input_s3_data_type": "Converse"
            if self.method not in (TrainingMethod.RFT_LORA, TrainingMethod.RFT_FULL)
            else "S3Prefix",
        }

        if self._platform in (Platform.BEDROCK, Platform.SMTJServerless):
            job_config_params["method"] = self.method

        job_id = self.infra.execute(job_config=JobConfig(**job_config_params))

        training_result: TrainingResult
        if self._platform in (Platform.SMTJ, Platform.SMTJServerless):
            training_result = SMTJTrainingResult(
                job_id=job_id,
                started_time=start_time,
                method=self.method,
                model_type=self.model,
                model_artifacts=get_model_artifacts(
                    job_name=job_id,
                    infra=self.infra,
                    output_s3_path=resolved_output_s3_path,
                ),
            )
        elif self._platform is Platform.BEDROCK:
            training_result = BedrockTrainingResult(
                job_id=job_id,
                started_time=start_time,
                method=self.method,
                model_type=self.model,
                model_artifacts=ModelArtifacts(
                    checkpoint_s3_path=None,
                    output_s3_path=resolved_output_s3_path,
                ),
            )
        else:
            cluster_name = cast(SMHPRuntimeManager, self.infra).cluster_name
            namespace = cast(SMHPRuntimeManager, self.infra).namespace
            training_result = SMHPTrainingResult(
                job_id=job_id,
                started_time=start_time,
                method=self.method,
                model_type=self.model,
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
            logger.info(f"Output S3 path is: {training_result.model_artifacts.output_s3_path}.")

        persist_result(
            self._cache_context,
            training_result,
            job_name=job_name,
            job_type="train",
            recipe_path=recipe_path,
            overrides=overrides or {},
        )

        return training_result

    @_telemetry_emitter(Feature.TRAINING, "get_logs")
    def get_logs(
        self,
        job_result: Optional[TrainingResult] = None,
        job_id: Optional[str] = None,
        started_time: Optional[datetime] = None,
        limit: Optional[int] = None,
        start_from_head: bool = False,
        end_time: Optional[int] = None,
    ) -> None:
        """Stream CloudWatch logs for a training job.

        Provide either a ``job_result`` or explicit ``job_id`` + ``started_time``.
        """
        resolved_job_id = job_result.job_id if job_result else job_id
        resolved_started = job_result.started_time if job_result else started_time

        if not resolved_job_id or not resolved_started:
            logger.info("Provide either a job_result or explicit job_id and started_time.")
            return

        kwargs: Dict[str, Any] = {}
        if self._platform == Platform.SMHP:
            kwargs["cluster_name"] = cast(SMHPRuntimeManager, self.infra).cluster_name
            kwargs["namespace"] = cast(SMHPRuntimeManager, self.infra).namespace

        monitor = CloudWatchLogMonitor(
            job_id=resolved_job_id,
            platform=self._platform,
            started_time=int(resolved_started.timestamp() * 1000),
            **kwargs,
        )
        monitor.show_logs(limit=limit, start_from_head=start_from_head, end_time=end_time)
