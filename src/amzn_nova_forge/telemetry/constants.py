# Copyright Amazon.com, Inc. or its affiliates

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
from __future__ import absolute_import

import platform
import sys
from enum import Enum
from importlib.metadata import version
from urllib.parse import quote

from amzn_nova_forge.core.enums import DeployPlatform, Platform, TrainingMethod

DEFAULT_AWS_REGION = "us-east-1"

UNKNOWN = "UNKNOWN"


class Feature(Enum):
    """Enumeration of feature names used in telemetry."""

    DATA_PREP = 1
    INFRA = 2
    EVAL = 3
    DEPLOY = 4
    MONITOR = 5
    TRAINING = 6
    BATCH_INFERENCE = 7

    def __str__(self):
        """Return the feature name."""
        return self.name


class Status(Enum):
    """Enumeration of status values used in telemetry."""

    SUCCESS = 1
    FAILURE = 0

    def __str__(self):
        """Return the status name."""
        return self.name


OS_NAME = platform.system() or "UnresolvedOS"
OS_VERSION = platform.release() or "UnresolvedOSVersion"
OS_NAME_VERSION = quote("{}/{}".format(OS_NAME, OS_VERSION), safe="")
PYTHON_VERSION = "{}.{}.{}".format(
    sys.version_info.major, sys.version_info.minor, sys.version_info.micro
)

TELEMETRY_OPT_OUT_MESSAGING = (
    "Nova Forge SDK will collect telemetry to help us better understand our user's needs, "
    "diagnose issues, and deliver additional features.\n"
    "To opt out of telemetry, please disable via TELEMETRY_OPT_OUT environment variable."
)

SDK_VERSION = version("amzn-nova-forge") if sys.modules.get("amzn_nova_forge") else UNKNOWN

PLATFORM_TO_CODE = {
    Platform.SMTJ: 1,
    Platform.SMHP: 2,
    Platform.BEDROCK: 3,
    Platform.SMTJServerless: 4,
    Platform.GLUE: 5,
    Platform.LOCAL: 6,
    DeployPlatform.BEDROCK_OD: 11,
    DeployPlatform.BEDROCK_PT: 12,
    DeployPlatform.SAGEMAKER: 13,
}

TRAINING_METHOD_TO_CODE = {
    TrainingMethod.CPT: 1,
    TrainingMethod.DPO_LORA: 2,
    TrainingMethod.DPO_FULL: 3,
    TrainingMethod.RFT_LORA: 4,
    TrainingMethod.RFT_FULL: 5,
    TrainingMethod.RFT_MULTITURN_LORA: 6,
    TrainingMethod.RFT_MULTITURN_FULL: 7,
    TrainingMethod.SFT_LORA: 8,
    TrainingMethod.SFT_FULL: 9,
    TrainingMethod.EVALUATION: 10,
}
