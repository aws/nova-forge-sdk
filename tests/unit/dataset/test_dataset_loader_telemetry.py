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
"""Tests for telemetry extra_info_fn callbacks in dataset_loader.py."""

import unittest
from unittest.mock import MagicMock

from amzn_nova_forge.core.enums import (
    EvaluationTask,
    FilterMethod,
    Model,
    Platform,
    TrainingMethod,
)
from amzn_nova_forge.dataset.dataset_loader import (
    _extract_filter_method,
    _extract_model_training_method,
)


class TestExtractFilterMethod(unittest.TestCase):
    """Tests for _extract_filter_method callback."""

    def test_with_runtime_manager_and_method(self):
        """Platform and filterMethod are both present when runtime_manager and method provided."""
        runtime_manager = MagicMock()
        runtime_manager.platform = Platform.SMTJ

        result = _extract_filter_method(
            method=FilterMethod.DEFAULT_TEXT_FILTER,
            runtime_manager=runtime_manager,
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["filterMethod"], "default_text_filter")
        self.assertIs(result["platform"], Platform.SMTJ)

    def test_without_runtime_manager(self):
        """Only filterMethod is returned when runtime_manager is absent."""
        result = _extract_filter_method(method=FilterMethod.EXACT_DEDUP)

        self.assertIsNotNone(result)
        self.assertEqual(result["filterMethod"], "exact_dedup_filter")
        self.assertNotIn("platform", result)

    def test_with_neither_method_nor_runtime_manager(self):
        """Returns None when both method and runtime_manager are absent."""
        result = _extract_filter_method()

        self.assertIsNone(result)


class TestExtractModelTrainingMethod(unittest.TestCase):
    """Tests for _extract_model_training_method callback."""

    def test_with_model_method_and_eval_task(self):
        """evalTask is present alongside model/method when eval_task provided."""
        result = _extract_model_training_method(
            model=Model.NOVA_LITE_2,
            training_method=TrainingMethod.EVALUATION,
            eval_task=EvaluationTask.MMLU,
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["model"], "nova_lite_2")
        self.assertIs(result["method"], TrainingMethod.EVALUATION)
        self.assertEqual(result["evalTask"], "mmlu")

    def test_with_model_and_method_only(self):
        """Only model and method returned when no eval_task provided."""
        result = _extract_model_training_method(
            model=Model.NOVA_MICRO,
            training_method=TrainingMethod.CPT,
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["model"], "nova_micro")
        self.assertIs(result["method"], TrainingMethod.CPT)
        self.assertNotIn("evalTask", result)

    def test_with_no_kwargs_at_all(self):
        """Returns None when no kwargs are provided."""
        result = _extract_model_training_method()

        self.assertIsNone(result)

    def test_eval_task_without_value_attribute(self):
        """evalTask falls back to str() when eval_task has no .value attribute."""
        result = _extract_model_training_method(eval_task="custom_task")

        self.assertIsNotNone(result)
        self.assertEqual(result["evalTask"], "custom_task")
