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

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from amzn_nova_forge.dataset.data_state import DataState


class DataPrepError(Exception):
    """Custom exception for data preparation errors."""

    pass


@dataclass
class OperationResult:
    """Common result returned by all dataset operations."""

    status: str
    output_state: Optional["DataState"] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict (mirrors the legacy return format)."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class FilterOperationResult(OperationResult):
    """Result for local filter operations (e.g. invalid-records filter).

    Inherits ``status`` from ``OperationResult``; callers must supply it
    (e.g. ``"SUCCEEDED"`` or ``"SKIPPED"``).
    """

    filtered_count: int = 0
    total_count: int = 0
    filters_applied: List[str] = field(default_factory=list)


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

    def prepare_input(self, state: DataState, **kwargs) -> DataState:
        """Prepare the input data for this operation if needed.

        The default implementation is a no-op passthrough. Operations
        with specific runtime constraints (e.g. requiring S3 paths or
        a particular format) override this to convert or upload data.

        Args:
            state: Current data state.
            **kwargs: Operation-specific arguments (e.g. output_path).

        Returns:
            A (possibly new) DataState ready for this operation.
        """
        return state


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


class NovaForgeFilterOperation(BaseOperation):
    """Base class for all data filtering operations.

    Filter operations accept a ``DataState`` describing the current data
    and call ``prepare_input()`` internally to handle any conversion or
    upload before running the filter pipeline.
    """

    pass


class NovaForgeAnalyzeOperation(BaseOperation):
    """Base class for all analyze operations. Reads dataset and produces analysis results."""

    pass
