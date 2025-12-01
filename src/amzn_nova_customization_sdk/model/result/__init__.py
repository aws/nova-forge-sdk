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
from amzn_nova_customization_sdk.model.result.eval_result import (
    EvaluationResult,
    SMHPEvaluationResult,
    SMTJEvaluationResult,
)
from amzn_nova_customization_sdk.model.result.inference_result import (
    InferenceResult,
    SMTJBatchInferenceResult,
)
from amzn_nova_customization_sdk.model.result.job_result import BaseJobResult, JobStatus
from amzn_nova_customization_sdk.model.result.training_result import (
    SMHPTrainingResult,
    SMTJTrainingResult,
    TrainingResult,
)
