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
Data mixing configuration for Nova models.

This module provides the DataMixing dataclass for configuring data mixing
between customer data and Nova curated data for SFT training.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from amzn_nova_customization_sdk.util.logging import logger
from amzn_nova_customization_sdk.validation.validator import Validator


@dataclass
class DataMixing:
    """
    Data mixing configuration for Nova models.

    This class manages datamixing configuration for SFT training, including
    validation and format conversion. It provides methods to get, set, and
    validate data mixing configurations.

    Attributes:
        config: Dictionary containing the datamixing configuration
                Keys and values are dynamically determined from recipe templates
        _default_nova_fields: Set of known nova field names for normalization
        _dataset_catalog: Stores the dataset catalog value (read-only from template)
    """

    # Constants for field name patterns
    NOVA_PREFIX = "nova_"
    PERCENT_SUFFIX = "_percent"
    CUSTOMER_DATA_FIELD = "customer_data_percent"
    DATASET_CATALOG_FIELD = "dataset_catalog"

    config: Dict[str, Any] = field(default_factory=dict)
    _default_nova_fields: set = field(
        default_factory=lambda: set(), init=False, repr=False
    )
    _dataset_catalog: Optional[Any] = field(default=None, init=False, repr=False)

    def get_config(self) -> Dict[str, Any]:
        """
        Get the current data mixing configuration.

        Returns:
            A copy of the current data mixing configuration dictionary including dataset_catalog
        """
        result = self.config.copy()
        if self._dataset_catalog is not None:
            result[self.DATASET_CATALOG_FIELD] = self._dataset_catalog
        return result

    def set_config(self, config: Dict[str, Any], normalize: bool = True) -> None:
        """
        Set the data mixing configuration.

        Args:
            config: Dictionary containing the data mixing configuration.
                   Keys should include nova_*_percent fields and customer_data_percent.
                   Any nova_*_percent fields not specified will be set to 0 if normalize=True.
            normalize: If True, unspecified nova fields will be set to 0. Default is True.

        Raises:
            ValueError: If configuration is invalid or contains unknown nova fields
        """
        if self.DATASET_CATALOG_FIELD in config:
            logger.warning(
                f"{self.DATASET_CATALOG_FIELD} cannot be set in data mixing configuration. "
                f"Ignoring value: {config[self.DATASET_CATALOG_FIELD]}"
            )

        # Create new config with normalization if needed
        new_config = {}

        if normalize:
            # Start with all known fields set to 0
            if (
                self._default_nova_fields
                or self.CUSTOMER_DATA_FIELD in self._default_nova_fields
            ):
                # Include nova fields
                for field in self._default_nova_fields:
                    new_config[field] = 0
                # Include customer_data_percent if it's in defaults
                if self.CUSTOMER_DATA_FIELD in self._default_nova_fields:
                    new_config[self.CUSTOMER_DATA_FIELD] = 0

        # Update with provided config
        new_config.update(config)

        # Validate the configuration
        Validator.validate_data_mixing_config(
            new_config,
            self.NOVA_PREFIX,
            self.PERCENT_SUFFIX,
            self.CUSTOMER_DATA_FIELD,
            self.DATASET_CATALOG_FIELD,
            self._default_nova_fields,
        )

        # Store the new config
        self.config = new_config

        logger.info(
            f"Setting other data mixing fields to 0. Data mixing configuration set to: {self.config}"
        )

    def _load_defaults_from_template(self, overrides_template: Dict[str, Any]) -> None:
        """
        Load default configuration from an overrides template.

        Args:
            overrides_template: Template dictionary containing default values
        """
        default_config = {}
        for key, value in overrides_template.items():
            if key.startswith(self.NOVA_PREFIX) and key.endswith(self.PERCENT_SUFFIX):
                self._default_nova_fields.add(key)
                if isinstance(value, dict) and "default" in value:
                    default_config[key] = value["default"]
            elif key == self.CUSTOMER_DATA_FIELD:
                # Add customer_data_percent to default fields
                self._default_nova_fields.add(key)
                if isinstance(value, dict) and "default" in value:
                    default_config[key] = value["default"]
            elif key == self.DATASET_CATALOG_FIELD:
                # Store dataset_catalog separately as it's read-only
                if isinstance(value, dict) and "default" in value:
                    self._dataset_catalog = value["default"]

        self.config.update(default_config)

    def _is_data_mixing_field(self, key: str) -> bool:
        return (
            (
                key.startswith(DataMixing.NOVA_PREFIX)
                and key.endswith(DataMixing.PERCENT_SUFFIX)
            )
            or key == DataMixing.CUSTOMER_DATA_FIELD
            or DataMixing.NOVA_PREFIX + key + DataMixing.PERCENT_SUFFIX
            in self.get_config()
            or key == DataMixing.DATASET_CATALOG_FIELD
        )
