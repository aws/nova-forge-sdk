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
from amzn_nova_customization_sdk.dataset.dataset_format_schema import (
    CPT_FORMAT,
    EVALUATION_FORMAT,
    OPENAI_FORMAT,
    RFT_OPENAI_FORMAT,
    SFT_NOVA_ONE_CONVERSE_2024,
    SFT_NOVA_TWO_CONVERSE_2024,
)
from amzn_nova_customization_sdk.model.model_enums import Model, TrainingMethod

TRANSFORM_CONFIG = {
    (
        (TrainingMethod.SFT_LORA, TrainingMethod.SFT_FULL),
        (Model.NOVA_MICRO, Model.NOVA_LITE, Model.NOVA_PRO),
    ): {
        "schema": SFT_NOVA_ONE_CONVERSE_2024,
        "success_msg": "Dataset is already in converse format for SFT, no transformation needed.",
        "transformers": [
            {
                "source_schema": OPENAI_FORMAT,
                "method": "convert_openai_to_converse_sft_nova_one",
                "msg": "Dataset is in OpenAI format. Attempting to transform to converse format for SFT.",
            },
            {
                "source_schema": None,  # Generic/fallback - no schema validation
                "method": "convert_to_converse_sft_nova_one",
                "msg": "Dataset is not in converse format. Attempting to transform to converse format for SFT.",
            },
        ],
    },
    ((TrainingMethod.SFT_LORA, TrainingMethod.SFT_FULL), Model.NOVA_LITE_2): {
        "schema": SFT_NOVA_TWO_CONVERSE_2024,
        "success_msg": "Dataset is already in converse format for SFT, no transformation needed.",
        "transformers": [
            {
                "source_schema": OPENAI_FORMAT,
                "method": "convert_openai_to_converse_sft_nova_two",
                "msg": "Dataset is in OpenAI format. Attempting to transform to converse format for SFT.",
            },
            {
                "source_schema": None,  # Generic/fallback
                "method": "convert_to_converse_sft_nova_two",
                "msg": "Dataset is not in converse format. Attempting to transform to converse format for SFT.",
            },
        ],
    },
    ((TrainingMethod.RFT_FULL, TrainingMethod.RFT_LORA), Model.NOVA_LITE_2): {
        "schema": RFT_OPENAI_FORMAT,
        "success_msg": "Dataset is already in OpenAI format for RFT, no transformation needed.",
        "transformers": [
            {
                "source_schema": None,  # Generic/fallback
                "method": "convert_to_openai_rft",
                "msg": "Dataset is not in OpenAI format for RFT. Attempting to transform to OpenAI format.",
            },
        ],
    },
    ((TrainingMethod.EVALUATION,), None): {
        "schema": EVALUATION_FORMAT,
        "success_msg": "Dataset is already in proper JSONL format, no transformation needed.",
        "transformers": [
            {
                "source_schema": None,  # Generic/fallback
                "method": "convert_to_evaluation",
                "msg": "Dataset is not in proper JSONL format for Evaluation. Attempting to transform to the correct format.",
            },
        ],
    },
    ((TrainingMethod.CPT,), None): {
        "schema": CPT_FORMAT,
        "success_msg": "Dataset is already in proper JSONL format, no transformation needed.",
        "transformers": [
            {
                "source_schema": None,  # Generic/fallback
                "method": "convert_to_cpt",
                "msg": "Dataset is not in proper JSONL format for CPT. Attempting to transform to the correct format.",
            },
        ],
    },
}
