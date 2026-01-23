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
MLflow monitoring for Nova model training.

This module provides MLflowMonitor class for tracking experiments and metrics.
"""

from dataclasses import dataclass, field
from typing import Optional

from amzn_nova_customization_sdk.util.mlflow import (
    get_default_mlflow_tracking_uri,
    validate_mlflow_overrides,
)


@dataclass
class MLflowMonitor:
    """
    MLflow monitoring configuration for Nova model training.

    This class encapsulates MLflow tracking configuration and can be passed
    to NovaModelCustomizer for experiment tracking.

    If tracking_uri is not provided, it will attempt to use a default
    SageMaker MLflow tracking server if one exists.

    Example:
        >>> # With explicit tracking URI
        >>> monitor = MLflowMonitor(
        ...     tracking_uri="arn:aws:sagemaker:us-east-1:123456:mlflow-app/app-xxx",
        ...     experiment_name="nova-customization",
        ...     run_name="sft-run-1"
        ... )

        >>> # With default tracking URI (if available)
        >>> monitor = MLflowMonitor(
        ...     experiment_name="nova-customization",
        ...     run_name="sft-run-1"
        ... )

        >>> customizer = NovaModelCustomizer(
        ...     model=Model.NOVA_LITE_2,
        ...     method=TrainingMethod.SFT_LORA,
        ...     infra=runtime_manager,
        ...     data_s3_path="s3://bucket/data",
        ...     mlflow_monitor=monitor
        ... )
    """

    tracking_uri: Optional[str] = field(default_factory=get_default_mlflow_tracking_uri)
    """MLflow tracking server URI or SageMaker MLflow app ARN. If not provided, attempts to use default."""

    experiment_name: Optional[str] = None
    """Name of the MLflow experiment. If not provided, will use job name."""

    run_name: Optional[str] = None
    """Name of the MLflow run. If not provided, will be auto-generated."""

    def __post_init__(self):
        """Validate MLflow configuration after initialization."""
        # Only validate if we have a tracking URI
        if self.tracking_uri:
            config = {
                "mlflow_tracking_uri": self.tracking_uri,
                "mlflow_experiment_name": self.experiment_name,
                "mlflow_run_name": self.run_name,
            }

            # Validate the MLflow configuration
            errors = validate_mlflow_overrides(config, check_exists=True)
            if errors:
                error_msg = "MLflow configuration validation failed:\n" + "\n".join(
                    f"  - {error}" for error in errors
                )
                raise ValueError(error_msg)

    def to_dict(self) -> dict:
        """
        Convert MLflow configuration to dictionary for use in overrides.

        Returns:
            Dictionary with mlflow_* keys for recipe configuration.
            Returns empty dict if no tracking URI is available.
        """
        if not self.tracking_uri:
            # No MLflow tracking if URI not available
            return {}

        config = {"mlflow_tracking_uri": self.tracking_uri}

        if self.experiment_name:
            config["mlflow_experiment_name"] = self.experiment_name

        if self.run_name:
            config["mlflow_run_name"] = self.run_name

        return config
