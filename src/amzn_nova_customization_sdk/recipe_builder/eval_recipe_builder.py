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
import inspect
from typing import Any, Dict, Optional

from amzn_nova_customization_sdk.model.model_enums import (
    Model,
    Platform,
    TrainingMethod,
)
from amzn_nova_customization_sdk.recipe_builder.base_recipe_builder import (
    BaseRecipeBuilder,
)
from amzn_nova_customization_sdk.recipe_config.base_recipe_config import (
    BaseRecipeConfig,
    BaseRunConfig,
)
from amzn_nova_customization_sdk.recipe_config.eval_config import (
    EvalRecipeConfig,
    EvaluationConfig,
    EvaluationMetric,
    EvaluationStrategy,
    EvaluationTask,
    InferenceConfig,
    ProcessorConfig,
    RLEnvConfig,
)
from amzn_nova_customization_sdk.validation.evaluation_validator import (
    EVAL_TASK_METRIC_MAP,
    EVAL_TASK_STRATEGY_MAP,
    EvaluationValidator,
)


class EvalRecipeBuilder(BaseRecipeBuilder):
    def __init__(
        self,
        job_name: str,
        platform: Platform,
        model: Model,
        model_path: str,
        instance_type: str,
        instance_count: int,
        data_s3_path: Optional[str],
        output_s3_path: str,
        eval_task: EvaluationTask,
        overrides: Dict[str, Any],
        infra: Optional[Any] = None,
        subtask: Optional[str] = None,
        method: TrainingMethod = TrainingMethod.EVALUATION,
        processor_config: Optional[Dict[str, Any]] = None,
        rl_env_config: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.eval_task = eval_task
        self.subtask = subtask
        self.version = model.version
        self.processor_config = processor_config or {}
        self.rl_env_config = rl_env_config or {}
        super().__init__(
            job_name=job_name,
            platform=platform,
            method=method,
            model_type=model.model_type,
            model_path=model_path,
            instance_type=instance_type,
            instance_count=instance_count,
            data_s3_path=data_s3_path or "",
            output_s3_path=output_s3_path,
            overrides=overrides,
            infra=infra,
        )

    def _validate_user_input(
        self, validation_config: Optional[Dict[str, bool]] = None
    ) -> None:
        EvaluationValidator.validate(
            eval_task=self.eval_task,
            data_s3_path=self.data_s3_path,
            subtask=self.subtask,
            model=self.model,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            overrides=self.overrides,
            processor_config=self.processor_config,
            rl_env_config=self.rl_env_config,
        )

    def _build_recipe_config(self) -> BaseRecipeConfig:
        run = self._create_base_run_config()
        strategy = EVAL_TASK_STRATEGY_MAP[self.eval_task]
        metric = EVAL_TASK_METRIC_MAP[self.eval_task]

        evaluation = EvaluationConfig(
            task=self.eval_task, strategy=strategy, metric=metric, subtask=self.subtask
        )
        # Get inference config params from overrides, ignore un-needed fields
        inference = InferenceConfig(
            **{k: v for k, v in self.overrides.items() if hasattr(InferenceConfig, k)}
        )
        # Get processor config if provided
        processor = (
            ProcessorConfig.from_dict(self.processor_config)
            if self.processor_config
            else None
        )
        # Get rl_env config if provided
        rl_env_param = {
            k: v
            for k, v in self.rl_env_config.items()
            if k in inspect.signature(RLEnvConfig).parameters.keys()
        }
        rl_env = RLEnvConfig(**rl_env_param) if rl_env_param else None
        return EvalRecipeConfig(
            run=run,
            evaluation=evaluation,
            inference=inference,
            processor=processor,
            rl_env=rl_env,
        )
