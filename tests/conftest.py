"""Pytest configuration for Nova Forge SDK tests.

Mocks sagemaker.ai_registry modules that create boto3 clients at import time.
"""

import sys
from unittest.mock import MagicMock

# Mock sagemaker.ai_registry.air_hub before it's imported
# This module creates boto3.client("sagemaker") at class definition time
mock_air_hub = MagicMock()
mock_air_hub.AIRHub = MagicMock()
sys.modules["sagemaker.ai_registry.air_hub"] = mock_air_hub

# Mock sagemaker.ai_registry.dataset which imports air_hub
mock_dataset = MagicMock()
mock_dataset.DataSet = MagicMock()
sys.modules["sagemaker.ai_registry.dataset"] = mock_dataset
