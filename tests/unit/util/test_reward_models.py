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
"""Tests for RewardMetric, RewardFunctionOutput models and _validate_output_format."""

import unittest

from pydantic import ValidationError

from amzn_nova_forge.util.reward_verifier import (
    RewardFunctionOutput,
    RewardMetric,
    _validate_output_format,
)


class TestRewardMetric(unittest.TestCase):
    """Tests for the RewardMetric Pydantic model."""

    def test_valid_metric_type(self):
        metric = RewardMetric(name="accuracy", value=0.95, type="Metric")
        self.assertEqual(metric.name, "accuracy")
        self.assertEqual(metric.value, 0.95)
        self.assertEqual(metric.type, "Metric")

    def test_valid_reward_type(self):
        metric = RewardMetric(name="final_reward", value=1.0, type="Reward")
        self.assertEqual(metric.type, "Reward")

    def test_integer_value(self):
        metric = RewardMetric(name="score", value=5, type="Metric")
        self.assertEqual(metric.value, 5)

    def test_invalid_type_value(self):
        with self.assertRaises(ValidationError) as ctx:
            RewardMetric(name="accuracy", value=0.95, type="Invalid")
        errors = ctx.exception.errors()
        self.assertEqual(len(errors), 1)
        self.assertIn("type", errors[0]["loc"])

    def test_non_numeric_value(self):
        with self.assertRaises(ValidationError) as ctx:
            RewardMetric(name="accuracy", value="not_a_number", type="Metric")
        errors = ctx.exception.errors()
        # Union[int, float] produces one error per union member in Pydantic v2
        self.assertGreaterEqual(len(errors), 1)

    def test_missing_name(self):
        with self.assertRaises(ValidationError):
            RewardMetric(value=0.5, type="Metric")

    def test_missing_value(self):
        with self.assertRaises(ValidationError):
            RewardMetric(name="accuracy", type="Metric")

    def test_missing_type(self):
        with self.assertRaises(ValidationError):
            RewardMetric(name="accuracy", value=0.5)


class TestRewardFunctionOutput(unittest.TestCase):
    """Tests for the RewardFunctionOutput Pydantic model."""

    def test_valid_without_metrics(self):
        output = RewardFunctionOutput(id="sample_1", aggregate_reward_score=0.8)
        self.assertEqual(output.id, "sample_1")
        self.assertEqual(output.aggregate_reward_score, 0.8)
        self.assertIsNone(output.metrics_list)

    def test_valid_with_metrics(self):
        output = RewardFunctionOutput(
            id="sample_2",
            aggregate_reward_score=0.9,
            metrics_list=[
                {"name": "accuracy", "value": 0.95, "type": "Metric"},
                {"name": "reward", "value": 1.0, "type": "Reward"},
            ],
        )
        self.assertEqual(len(output.metrics_list), 2)
        self.assertIsInstance(output.metrics_list[0], RewardMetric)
        self.assertIsInstance(output.metrics_list[1], RewardMetric)

    def test_valid_with_integer_score(self):
        output = RewardFunctionOutput(id="sample_3", aggregate_reward_score=1)
        self.assertEqual(output.aggregate_reward_score, 1)

    def test_missing_id(self):
        with self.assertRaises(ValidationError) as ctx:
            RewardFunctionOutput(aggregate_reward_score=0.5)
        error_locs = [tuple(e["loc"]) for e in ctx.exception.errors()]
        self.assertIn(("id",), error_locs)

    def test_missing_aggregate_reward_score(self):
        with self.assertRaises(ValidationError) as ctx:
            RewardFunctionOutput(id="sample_1")
        error_locs = [tuple(e["loc"]) for e in ctx.exception.errors()]
        self.assertIn(("aggregate_reward_score",), error_locs)

    def test_non_numeric_score(self):
        with self.assertRaises(ValidationError):
            RewardFunctionOutput(id="sample_1", aggregate_reward_score="bad")

    def test_metrics_list_with_invalid_metric(self):
        with self.assertRaises(ValidationError):
            RewardFunctionOutput(
                id="sample_1",
                aggregate_reward_score=0.5,
                metrics_list=[{"name": "acc", "value": "bad", "type": "Metric"}],
            )

    def test_explicit_none_metrics_list(self):
        output = RewardFunctionOutput(id="sample_1", aggregate_reward_score=0.5, metrics_list=None)
        self.assertIsNone(output.metrics_list)

    def test_empty_metrics_list(self):
        output = RewardFunctionOutput(id="sample_1", aggregate_reward_score=0.5, metrics_list=[])
        self.assertEqual(output.metrics_list, [])


class TestValidateOutputFormat(unittest.TestCase):
    """Tests for the _validate_output_format function."""

    def test_valid_dict_returns_empty_list(self):
        result = {"id": "s1", "aggregate_reward_score": 0.8}
        errors = _validate_output_format(result, 0)
        self.assertEqual(errors, [])

    def test_valid_with_metrics(self):
        result = {
            "id": "s1",
            "aggregate_reward_score": 0.9,
            "metrics_list": [
                {"name": "acc", "value": 0.95, "type": "Metric"},
            ],
        }
        errors = _validate_output_format(result, 0)
        self.assertEqual(errors, [])

    def test_non_dict_input(self):
        errors = _validate_output_format("not_a_dict", 0)
        self.assertEqual(len(errors), 1)
        self.assertIn("Expected dict", errors[0])
        self.assertIn("Output 0", errors[0])

    def test_non_dict_input_list(self):
        errors = _validate_output_format([1, 2, 3], 2)
        self.assertEqual(len(errors), 1)
        self.assertIn("Expected dict", errors[0])
        self.assertIn("Output 2", errors[0])

    def test_missing_required_fields(self):
        errors = _validate_output_format({}, 0)
        self.assertGreater(len(errors), 0)
        error_text = " ".join(errors)
        self.assertIn("id - Field required", error_text)
        self.assertIn("aggregate_reward_score - Field required", error_text)

    def test_missing_id_only(self):
        errors = _validate_output_format({"aggregate_reward_score": 0.5}, 1)
        self.assertEqual(len(errors), 1)
        self.assertIn("id - Field required", errors[0])

    def test_missing_aggregate_reward_score_only(self):
        errors = _validate_output_format({"id": "s1"}, 3)
        self.assertEqual(len(errors), 1)
        self.assertIn("aggregate_reward_score - Field required", errors[0])

    def test_invalid_metric_type_in_metrics_list(self):
        result = {
            "id": "s1",
            "aggregate_reward_score": 0.5,
            "metrics_list": [
                {"name": "acc", "value": 0.9, "type": "BadType"},
            ],
        }
        errors = _validate_output_format(result, 0)
        self.assertGreater(len(errors), 0)

    def test_non_numeric_aggregate_reward_score(self):
        result = {"id": "s1", "aggregate_reward_score": "bad"}
        errors = _validate_output_format(result, 0)
        self.assertGreater(len(errors), 0)

    def test_index_appears_in_error_messages(self):
        errors = _validate_output_format({}, 42)
        for err in errors:
            self.assertIn("Output 42", err)

    def test_extra_fields_are_allowed(self):
        """Extra fields should not cause validation errors (Pydantic default)."""
        result = {
            "id": "s1",
            "aggregate_reward_score": 0.8,
            "extra_field": "ignored",
        }
        errors = _validate_output_format(result, 0)
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
