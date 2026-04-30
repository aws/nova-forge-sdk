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
"""
Model and training method enumerations for Nova customization.

Re-exported from amzn_nova_forge.core.enums for backward compatibility.
"""

from amzn_nova_forge.core.constants import SUPPORTED_DATAMIXING_METHODS
from amzn_nova_forge.core.enums import (
    DeploymentMode,
    DeployPlatform,
    Model,
    Platform,
    TrainingMethod,
    Version,
)

__all__ = [
    "Platform",
    "Version",
    "Model",
    "TrainingMethod",
    "DeployPlatform",
    "DeploymentMode",
    "SUPPORTED_DATAMIXING_METHODS",
]
