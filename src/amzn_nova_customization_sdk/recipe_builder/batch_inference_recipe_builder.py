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
from typing import Any, Dict, Optional

from amzn_nova_customization_sdk.model.model_enums import (
    Model,
    Platform,
    TrainingMethod,
)
from amzn_nova_customization_sdk.recipe_builder.base_recipe_builder import (
    BaseRecipeBuilder,
)
from amzn_nova_customization_sdk.recipe_config.eval_config import (
    EvalRecipeConfig,
    EvaluationConfig,
    EvaluationMetric,
    EvaluationStrategy,
    EvaluationTask,
    InferenceConfig,
)
from amzn_nova_customization_sdk.validation.evaluation_validator import (
    EvaluationValidator,
)


class BatchInferenceRecipeBuilder(BaseRecipeBuilder):
    def __init__(
        self,
        job_name: str,
        platform: Platform,
        model: Model,
        model_path: str,
        instance_type: str,
        instance_count: int,
        data_s3_path: str,
        output_s3_path: str,
        overrides: Dict[str, Any],
        method: TrainingMethod = TrainingMethod.EVALUATION,
        infra: Optional[Any] = None,
    ):
        self.model = model
        self.instance_type = instance_type
        self.instance_count = instance_count
        super().__init__(
            job_name=job_name,
            platform=platform,
            method=method,
            model_type=model.model_type,
            model_path=model_path,
            instance_type=instance_type,
            instance_count=instance_count,
            data_s3_path=data_s3_path,
            output_s3_path=output_s3_path,
            overrides=overrides,
            infra=infra,
        )

    def _validate_user_input(
        self, validation_config: Optional[Dict[str, bool]] = None
    ) -> None:
        # This contains the same main parameters as Eval, so reusing the validator.
        EvaluationValidator.validate(
            eval_task=EvaluationTask.GEN_QA,
            data_s3_path=self.data_s3_path,
            model=self.model,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            overrides=self.overrides,
        )

    # Inference can use the same config as Evaluation.
    def _build_recipe_config(self) -> EvalRecipeConfig:
        run = self._create_base_run_config()

        # GenQA and All needs to be selected to generate the inference.jsonl file.
        evaluation = EvaluationConfig(
            task=EvaluationTask.GEN_QA,
            strategy=EvaluationStrategy.GEN_QA,
            metric=EvaluationMetric.ALL,
        )

        # Get inference config params from overrides, ignore un-needed fields.
        inference = InferenceConfig(
            **{k: v for k, v in self.overrides.items() if hasattr(InferenceConfig, k)}
        )

        return EvalRecipeConfig(run=run, evaluation=evaluation, inference=inference)
