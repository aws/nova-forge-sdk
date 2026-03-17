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
"""Base classes for all dataset operations, organized by operation family."""

from abc import ABC, abstractmethod
from typing import Any


class DataPrepError(Exception):
    """Custom exception for data preparation errors."""

    pass


class BaseOperation(ABC):
    """
    Abstract base class for all dataset operations.

    Each operation reads from and/or writes to the loader's dataset state.
    """

    @abstractmethod
    def execute(self, loader: Any, **kwargs) -> Any:
        """
        Execute this operation against the given DatasetLoader.

        Args:
            loader: The DatasetLoader instance to operate on.
            **kwargs: Operation-specific arguments.

        Returns:
            Operation-specific return value.
        """
        pass


# --- Typed operation families ---
# Each DatasetLoader method accepts only its corresponding operation type.


class NovaForgeTransformOperation(BaseOperation):
    """Base class for all transform operations. Produces a new dataset."""

    pass


class NovaForgeValidateOperation(BaseOperation):
    """Base class for all validate operations. Reads dataset without modifying it."""

    pass


class NovaForgeShowOperation(BaseOperation):
    """Base class for all show/display operations."""

    pass


class NovaForgeSaveOperation(BaseOperation):
    """Base class for all save/export operations."""

    pass


class NovaForgeSplitOperation(BaseOperation):
    """Base class for all dataset splitting operations."""

    pass
