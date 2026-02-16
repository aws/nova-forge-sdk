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
"""RFT Multiturn Infrastructure Module - Modular platform-specific implementations."""

from .base_infra import (
    RFT_EXECUTION_ROLE_NAME,
    EnvType,
    StackOutputs,
    VFEnvId,
    create_rft_execution_role,
)
from .custom_environment import CustomEnvironment
from .rft_multiturn import RFTMultiturnInfrastructure, list_rft_stacks

__all__ = [
    "RFTMultiturnInfrastructure",
    "list_rft_stacks",
    "create_rft_execution_role",
    "RFT_EXECUTION_ROLE_NAME",
    "EnvType",
    "VFEnvId",
    "StackOutputs",
    "CustomEnvironment",
]
