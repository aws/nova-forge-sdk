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
import re
from typing import Dict, List, Optional

import pandas

from amzn_nova_forge.model.model_enums import Platform, TrainingMethod

GLOBAL_STEP_REGEX = r"global_step[=:]\s*([\d.]+)"
TRAINING_LOSS_REGEX = r"reduced_train_loss[=:]\s*(-?[\d.]+(?:[eE][+-]?\d+)?)"
SMHP_RFT_REWARD_SCORE_REGEX = r"train_rm_score:\s*(-?[\d.]+(?:[eE][+-]?\d+)?)"
SMTJ_RFT_REWARD_SCORE_REGEX = r"critic/rewards/mean[=:]\s*(-?[\d.]+(?:[eE][+-]?\d+)?)"
CPT = "cpt"
SFT = "sft"
RFT = "rft"
AVAILABLE_METRICS = {
    Platform.SMTJ: {
        CPT: {"training_loss": TRAINING_LOSS_REGEX},
        SFT: {"training_loss": TRAINING_LOSS_REGEX},
        RFT: {"reward_score": SMTJ_RFT_REWARD_SCORE_REGEX},
    },
    Platform.SMHP: {
        CPT: {"training_loss": TRAINING_LOSS_REGEX},
        SFT: {"training_loss": TRAINING_LOSS_REGEX},
        RFT: {"reward_score": SMHP_RFT_REWARD_SCORE_REGEX},
    },
}


def get_metrics(
    platform: Platform,
    training_method: TrainingMethod,
    logs: Optional[List[Dict]] = None,
    metrics: Optional[List] = None,
) -> pandas.DataFrame:
    patterns = []
    training_category = training_method.value[:3]
    if not metrics:
        metrics = list(AVAILABLE_METRICS[platform][training_category].keys())
    for metric in metrics:
        if metric not in AVAILABLE_METRICS[platform][training_category]:
            raise NotImplementedError(
                f"Unsupported metric for {training_category} on {platform}: {metric}"
            )
        patterns.append(AVAILABLE_METRICS[platform][training_category][metric])

    all_metrics: List[List[float]] = []
    log_lines = [line for log in (logs or []) for line in log["message"].splitlines()]

    for line in log_lines:
        global_step_match = re.search(GLOBAL_STEP_REGEX, line)
        if not global_step_match:
            continue
        try:
            step_metrics: List[float] = [int(float(global_step_match.group(1)))]
            for pattern in patterns:
                match = re.search(pattern, line)
                if match is None:
                    raise ValueError(f"Pattern {pattern} not found in line")
                step_metrics.append(float(match.group(1)))
            all_metrics.append(step_metrics)
        except Exception:
            pass

    return pandas.DataFrame(all_metrics, columns=["global_step"] + metrics)
