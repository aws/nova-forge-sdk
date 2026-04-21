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
"""A lightweight runtime manager for operations that run entirely local."""

from __future__ import annotations

from amzn_nova_forge.manager.runtime_manager import RuntimeManager
from amzn_nova_forge.model.model_enums import Platform


class LocalRuntimeManager(RuntimeManager):
    """Runtime manager for operations that need no external compute (no Glue, no SageMaker).

    Satisfies the ``RuntimeManager`` interface so operations can declare
    ``get_supported_runtimes() -> (LocalRuntimeManager,)``
    """

    def __init__(self) -> None:
        super().__init__(
            instance_type=None,
            instance_count=None,
            kms_key_id=None,
        )

    @property
    def platform(self) -> Platform:
        return Platform.LOCAL

    @property
    def runtime_name(self) -> str:
        return "Local"

    def setup(self) -> None:
        pass

    def execute(self, job_config):
        raise NotImplementedError(
            "LocalRuntimeManager does not submit remote jobs. "
            "Operations using it should run logic locally."
        )

    def cleanup(self, job_id: str) -> None:
        pass
