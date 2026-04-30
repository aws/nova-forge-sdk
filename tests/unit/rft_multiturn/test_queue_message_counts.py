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
"""Unit tests for the QueueMessageCounts Pydantic model."""

import pytest
from pydantic import ValidationError

from amzn_nova_forge.rft_multiturn.base_infra import QueueMessageCounts


class TestQueueMessageCounts:
    """Test suite for QueueMessageCounts model."""

    def test_basic_construction(self):
        """Test construction with valid integer values."""
        counts = QueueMessageCounts(visible=10, in_flight=5, last_receive_timestamp=1700000000)
        assert counts.visible == 10
        assert counts.in_flight == 5
        assert counts.last_receive_timestamp == 1700000000

    def test_zero_values(self):
        """Test construction with all zero values."""
        counts = QueueMessageCounts(visible=0, in_flight=0, last_receive_timestamp=0)
        assert counts.visible == 0
        assert counts.in_flight == 0
        assert counts.last_receive_timestamp == 0

    def test_missing_visible_raises(self):
        """Test that omitting visible raises a validation error."""
        with pytest.raises(ValidationError):
            QueueMessageCounts(in_flight=5, last_receive_timestamp=0)  # type: ignore[call-arg]

    def test_missing_in_flight_raises(self):
        """Test that omitting in_flight raises a validation error."""
        with pytest.raises(ValidationError):
            QueueMessageCounts(visible=10, last_receive_timestamp=0)  # type: ignore[call-arg]

    def test_missing_last_receive_timestamp_raises(self):
        """Test that omitting last_receive_timestamp raises a validation error."""
        with pytest.raises(ValidationError):
            QueueMessageCounts(visible=10, in_flight=5)  # type: ignore[call-arg]

    def test_missing_all_fields_raises(self):
        """Test that constructing with no arguments raises a validation error."""
        with pytest.raises(ValidationError):
            QueueMessageCounts()  # type: ignore[call-arg]

    def test_type_coercion_from_string(self):
        """Test that Pydantic coerces string values to int."""
        counts = QueueMessageCounts(
            visible="5",  # type: ignore[arg-type]
            in_flight="3",  # type: ignore[arg-type]
            last_receive_timestamp="1700000000",  # type: ignore[arg-type]
        )
        assert counts.visible == 5
        assert counts.in_flight == 3
        assert counts.last_receive_timestamp == 1700000000

    def test_type_coercion_from_float(self):
        """Test that Pydantic coerces float values to int."""
        counts = QueueMessageCounts(
            visible=5.0,  # type: ignore[arg-type]
            in_flight=3.0,  # type: ignore[arg-type]
            last_receive_timestamp=1700000000.0,  # type: ignore[arg-type]
        )
        assert counts.visible == 5
        assert counts.in_flight == 3
        assert counts.last_receive_timestamp == 1700000000

    def test_invalid_type_raises(self):
        """Test that non-numeric values raise a validation error."""
        with pytest.raises(ValidationError):
            QueueMessageCounts(
                visible="not_a_number",  # type: ignore[arg-type]
                in_flight=5,
                last_receive_timestamp=0,
            )

    def test_attribute_access(self):
        """Test that fields are accessible as attributes, not dict keys."""
        counts = QueueMessageCounts(visible=10, in_flight=5, last_receive_timestamp=1700000000)
        # Attribute access works
        assert counts.visible == 10
        assert counts.in_flight == 5
        assert counts.last_receive_timestamp == 1700000000

    def test_model_dump_returns_expected_dict(self):
        """Test that model_dump() returns the expected dict representation."""
        counts = QueueMessageCounts(visible=10, in_flight=5, last_receive_timestamp=0)
        # Verify it's a Pydantic BaseModel instance
        assert hasattr(counts, "model_dump")
        result = counts.model_dump()
        assert result == {"visible": 10, "in_flight": 5, "last_receive_timestamp": 0}

    def test_re_exported_from_package(self):
        """Test that QueueMessageCounts is importable from the rft_multiturn package."""
        from amzn_nova_forge.rft_multiturn import QueueMessageCounts as PackageLevelImport

        counts = PackageLevelImport(visible=1, in_flight=2, last_receive_timestamp=3)
        assert counts.visible == 1
