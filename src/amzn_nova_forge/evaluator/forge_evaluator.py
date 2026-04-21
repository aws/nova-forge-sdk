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
"""ForgeEvaluator — owns the evaluation workflow."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, Optional, cast

import boto3

from amzn_nova_forge.core.constants import DEFAULT_REGION
from amzn_nova_forge.core.enums import EvaluationTask, Model, Platform, TrainingMethod
from amzn_nova_forge.core.job_cache import (
    build_cache_context,
    load_existing_result,
    persist_result,
)
from amzn_nova_forge.core.result import (
    EvaluationResult,
    SMHPEvaluationResult,
    SMTJEvaluationResult,
    TrainingResult,
)
from amzn_nova_forge.core.runtime import RuntimeManager
from amzn_nova_forge.core.types import (
    ForgeConfig,
    JobConfig,
    validate_region,
)
from amzn_nova_forge.manager.runtime_manager import SMHPRuntimeManager
from amzn_nova_forge.model.nova_model_customizer_util import (
    requires_custom_eval_data,
    resolve_model_checkpoint_path,
    set_output_s3_path,
)
from amzn_nova_forge.monitor.log_monitor import CloudWatchLogMonitor
from amzn_nova_forge.recipe.recipe_builder import RecipeBuilder
from amzn_nova_forge.telemetry import Feature, _telemetry_emitter
from amzn_nova_forge.util.logging import logger
from amzn_nova_forge.util.platform_util import (
    detect_platform_from_path,
    validate_platform_compatibility,
)
from amzn_nova_forge.validation.validator import validate_rft_lambda_name

if TYPE_CHECKING:
    from amzn_nova_forge.rft_multiturn import RFTMultiturnInfrastructure


@dataclass
class EvalTaskConfig:
    """Per-task configuration for an evaluation job."""

    subtask: Optional[str] = None
    processor: Optional[Dict[str, Any]] = None
    rl_env: Optional[Dict[str, Any]] = None
    override_data_s3_path: Optional[str] = None


class ForgeEvaluator:
    """Encapsulates the evaluation workflow for Nova model customization.

    Configuration is provided in the constructor; ``evaluate()`` accepts
    per-job parameters including the evaluation task.
    """

    def __init__(
        self,
        model: Model,
        infra: RuntimeManager,
        data_s3_path: Optional[str] = None,
        config: Optional[ForgeConfig] = None,
        region: Optional[str] = None,
    ) -> None:
        self.model = model
        self.infra = infra
        self.data_s3_path = data_s3_path
        self._config = config or ForgeConfig()

        self.region = region or boto3.session.Session().region_name or DEFAULT_REGION
        validate_region(self.region)

        self._platform = infra.platform
        self._is_multimodal = False  # Default; evaluation multimodal support can be extended later

        self.output_s3_path = set_output_s3_path(
            region=self.region,
            output_s3_path=self._config.output_s3_path,
            kms_key_id=self.infra.kms_key_id,
        )

        # Job caching context
        self._cache_context = build_cache_context(
            self._config,
            model=model,
            method=TrainingMethod.EVALUATION,
            data_s3_path=data_s3_path,
            output_s3_path=self.output_s3_path,
            instance_type=infra.instance_type,
            instance_count=infra.instance_count,
        )

    @_telemetry_emitter(
        Feature.EVAL,
        "evaluate",
        extra_info_fn=lambda self, *args, **kwargs: {
            "model": self.model.value,
            "platform": self._platform,
            "dryRun": kwargs.get("dry_run", False),
        },
    )
    def evaluate(
        self,
        job_name: str,
        eval_task: EvaluationTask,
        model_path: Optional[str] = None,
        task_config: Optional[EvalTaskConfig] = None,
        recipe_path: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
        dry_run: bool = False,
        job_result: Optional[TrainingResult] = None,
        rft_multiturn_infra: Optional[RFTMultiturnInfrastructure] = None,
    ) -> Optional[EvaluationResult]:
        """Launch an evaluation job.

        Args:
            job_name: User-defined name for the evaluation job.
            eval_task: The evaluation task to perform (e.g. MMLU).
            model_path: Optional S3 path to the model checkpoint.
            task_config: Optional per-task configuration.
            recipe_path: Optional path to a YAML recipe file.
            overrides: Optional dictionary of configuration overrides.
            dry_run: If True, only validate — do not start a job.
            job_result: Optional TrainingResult to extract checkpoint path from.
            rft_multiturn_infra: Optional RFT multiturn infrastructure (passed
                through to RecipeBuilder for multi-turn RFT evaluation).

        Returns:
            EvaluationResult on success, None if dry_run is True.
        """
        if self._platform == Platform.BEDROCK:
            raise NotImplementedError(
                "Evaluation is not supported on the Bedrock platform. "
                "Use SageMaker platforms (SMTJ, SMHP) instead."
            )

        # Check job cache
        cached = load_existing_result(
            self._cache_context,
            job_name=job_name,
            job_type="eval",
            model_path=model_path,
            recipe_path=recipe_path,
            overrides=overrides or {},
        )
        if cached:
            logger.info("Returning cached result for '%s'.", job_name)
            return cached  # type: ignore[return-value]

        tc = task_config or EvalTaskConfig()

        if tc.rl_env and tc.rl_env.get("reward_lambda_arn"):
            validate_rft_lambda_name(tc.rl_env["reward_lambda_arn"].split(":")[-1], self._platform)

        # Resolve model checkpoint
        resolved_model_path = resolve_model_checkpoint_path(
            model_path=model_path,
            job_result=job_result,
            customizer_job_id=None,
            customizer_output_s3_path=self.output_s3_path,
            customizer_model_path=None,
        )

        if resolved_model_path is None:
            logger.warning(
                f"Could not resolve model checkpoint path for evaluate job! "
                f"Falling back to base model {self.model}"
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
            execution_platform=self._platform,
            checkpoint_source="evaluation model checkpoint",
        )

        # Resolve data path
        data_s3_path_for_job = tc.override_data_s3_path
        if data_s3_path_for_job is None and requires_custom_eval_data(eval_task):
            data_s3_path_for_job = self.data_s3_path

        if not requires_custom_eval_data(eval_task) and self.data_s3_path:
            logger.info(
                f"{eval_task} does not use custom data, ignoring ForgeEvaluator's data_s3_path."
            )

        # Resolve processor / rl_env with lambda auto-population
        resolved_processor = tc.processor
        resolved_rl_env = tc.rl_env
        infra_lambda_arn = getattr(self.infra, "rft_lambda_arn", None)

        if (
            eval_task == EvaluationTask.RFT_EVAL
            and resolved_processor
            and resolved_processor.get("lambda_arn")
        ):
            if resolved_rl_env is None:
                resolved_rl_env = {"reward_lambda_arn": resolved_processor["lambda_arn"]}
                logger.info(f"Using reward_lambda_arn: {resolved_processor['lambda_arn']}")
            resolved_processor = None

        if infra_lambda_arn and resolved_rl_env is None and eval_task == EvaluationTask.RFT_EVAL:
            resolved_rl_env = {"reward_lambda_arn": infra_lambda_arn}
            logger.info(f"Using reward_lambda_arn: {infra_lambda_arn}")

        recipe_builder = RecipeBuilder(
            region=self.region,
            job_name=job_name,
            platform=self._platform,
            model=self.model,
            method=TrainingMethod.EVALUATION,
            instance_type=self.infra.instance_type,
            instance_count=self.infra.instance_count,
            infra=self.infra,
            data_s3_path=data_s3_path_for_job,
            output_s3_path=self.output_s3_path,
            model_path=resolved_model_path,
            eval_task=eval_task,
            subtask=tc.subtask,
            processor_config=resolved_processor,
            rl_env_config=resolved_rl_env,
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

        job_id = self.infra.execute(
            job_config=JobConfig(
                job_name=unique_job_name,
                data_s3_path=resolved_data_s3_path,
                output_s3_path=resolved_output_s3_path,
                image_uri=resolved_image_uri,
                recipe_path=resolved_recipe_path,
                input_s3_data_type="S3Prefix",
                method=TrainingMethod.EVALUATION,
            )
        )

        evaluation_result: EvaluationResult
        if self._platform in (Platform.SMTJ, Platform.SMTJServerless):
            eval_output_s3_path = (
                f"{resolved_output_s3_path.rstrip('/')}/{job_id}/output/output.tar.gz"
            )
            evaluation_result = SMTJEvaluationResult(
                job_id=job_id,
                eval_task=eval_task,
                started_time=start_time,
                eval_output_path=eval_output_s3_path,
            )
        else:
            cluster_name = cast(SMHPRuntimeManager, self.infra).cluster_name
            namespace = cast(SMHPRuntimeManager, self.infra).namespace
            eval_output_s3_path = f"{resolved_output_s3_path.rstrip('/')}/{job_id}/eval-result/"
            evaluation_result = SMHPEvaluationResult(
                job_id=job_id,
                eval_task=eval_task,
                started_time=start_time,
                eval_output_path=eval_output_s3_path,
                cluster_name=cluster_name,
                namespace=namespace,
            )

        logger.info(
            f"Started eval job '{job_id}'. Artifacts will be published to {eval_output_s3_path}"
        )
        persist_result(
            self._cache_context,
            evaluation_result,
            job_name=job_name,
            job_type="eval",
            model_path=model_path,
            recipe_path=recipe_path,
            overrides=overrides or {},
        )

        return evaluation_result

    @_telemetry_emitter(Feature.EVAL, "get_logs")
    def get_logs(
        self,
        job_result: Optional[EvaluationResult] = None,
        job_id: Optional[str] = None,
        started_time: Optional[datetime] = None,
        limit: Optional[int] = None,
        start_from_head: bool = False,
        end_time: Optional[int] = None,
    ) -> None:
        """Stream CloudWatch logs for an evaluation job."""
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
