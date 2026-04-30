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
"""Pytest configuration for Nova Forge SDK tests.

Mocks sagemaker submodules that are only available in internal builds
(ai_registry, train, core.*).  These modules either create boto3 clients
at import time or simply don't exist in the public sagemaker package.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

_sagemaker_mocks = [
    "sagemaker.ai_registry",
    "sagemaker.ai_registry.air_hub",
    "sagemaker.ai_registry.dataset",
    "sagemaker.core",
    "sagemaker.core.helper",
    "sagemaker.core.helper.session_helper",
    "sagemaker.core.image_uris",
    "sagemaker.core.shapes",
    "sagemaker.core.training",
    "sagemaker.core.training.configs",
    "sagemaker.train",
    "sagemaker.train.model_trainer",
]
for mod in _sagemaker_mocks:
    sys.modules.setdefault(mod, MagicMock())


@pytest.fixture(autouse=True)
def _mock_telemetry(request):
    """Prevent telemetry from making real network/STS calls during tests.

    Tests in the telemetry/ directory test the telemetry module itself and
    manage their own mocks, so this fixture is skipped for them.
    """
    if "no_auto_mock_telemetry" in request.keywords:
        yield
        return
    with patch("amzn_nova_forge.telemetry.telemetry_logging._send_telemetry_request"):
        yield
