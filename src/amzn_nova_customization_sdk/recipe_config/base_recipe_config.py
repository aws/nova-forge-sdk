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
"""
Generic recipe configuration dataclasses that are shared across multiple training methods.

This module defines the base structure for all recipe configurations.
Each training method should extend or use these base configs.
"""

from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from typing import Any, Dict


def to_primitive(value):
    """Convert dataclasses and Enums to plain Python types."""
    if isinstance(value, Enum):
        return value.value
    elif is_dataclass(value):
        return {k: to_primitive(v) for k, v in asdict(value).items() if v is not None}
    elif isinstance(value, dict):
        return {k: to_primitive(v) for k, v in value.items() if v is not None}
    elif isinstance(value, (list, tuple)):
        return [to_primitive(v) for v in value if v is not None]
    else:
        return value


@dataclass
class BaseRunConfig:
    name: str
    model_type: str
    model_name_or_path: str
    replicas: int
    data_s3_path: str
    output_s3_path: str

    def to_dict(self) -> Dict[str, Any]:
        return to_primitive(self)


@dataclass
class BaseRecipeConfig:
    run: BaseRunConfig

    def to_dict(self) -> Dict[str, Any]:
        return to_primitive(self)
