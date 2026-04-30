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
"""Tests for _run_validations_in_scope dispatch logic."""

import unittest
from unittest.mock import MagicMock, patch

from pydantic import BaseModel, ValidationInfo

from amzn_nova_forge.core.enums import Model, Platform, TrainingMethod
from amzn_nova_forge.dataset.configs.dataset_checks_config import DATASET_CHECKS
from amzn_nova_forge.dataset.dataset_validator.dataset_validator import (
    _run_validations_in_scope,
)


class _FakeModel(BaseModel):
    text: str = "hello"


def _make_info(context=None):
    info = MagicMock(spec=ValidationInfo)
    info.context = context if context is not None else {"model": Model.NOVA_LITE_2}
    return info


_BASE_CHECK = {
    "name": "test_check",
    "type": "keyword",
    "scope": {"_FakeModel"},
    "filterable": True,
    "invalid_keywords": ["bad"],
    "applicable_training_methods": {TrainingMethod.SFT_LORA},
    "applicable_platforms": {Platform.SMTJ},
    "applicable_models": {Model.NOVA_LITE_2},
}


def _check(**overrides):
    c = dict(_BASE_CHECK)
    c.update(overrides)
    return c


_MODULE = "amzn_nova_forge.dataset.dataset_validator.dataset_validator"


class TestRunValidationsInScope(unittest.TestCase):
    @patch(f"{_MODULE}.DATASET_CHECKS")
    def test_skipped_when_filterable_is_false(self, mock_checks):
        mock_checks.__iter__ = lambda self: iter([_check(filterable=False)])
        executor = MagicMock()
        with patch(f"{_MODULE}._EXECUTORS", {"keyword": executor}):
            _run_validations_in_scope(_FakeModel(), _make_info(), TrainingMethod.SFT_LORA)
        executor.assert_not_called()

    @patch(f"{_MODULE}.DATASET_CHECKS")
    def test_skipped_when_class_not_in_scope(self, mock_checks):
        mock_checks.__iter__ = lambda self: iter([_check(scope=["OtherModel"])])
        executor = MagicMock()
        with patch(f"{_MODULE}._EXECUTORS", {"keyword": executor}):
            _run_validations_in_scope(_FakeModel(), _make_info(), TrainingMethod.SFT_LORA)
        executor.assert_not_called()

    @patch(f"{_MODULE}.DATASET_CHECKS")
    def test_skipped_when_training_method_does_not_match(self, mock_checks):
        mock_checks.__iter__ = lambda self: iter([_check()])
        executor = MagicMock()
        with patch(f"{_MODULE}._EXECUTORS", {"keyword": executor}):
            _run_validations_in_scope(_FakeModel(), _make_info(), TrainingMethod.CPT)
        executor.assert_not_called()

    @patch(f"{_MODULE}.DATASET_CHECKS")
    def test_skipped_when_scope_absent(self, mock_checks):
        no_scope = _check()
        del no_scope["scope"]
        mock_checks.__iter__ = lambda self: iter([no_scope])
        executor = MagicMock()
        with patch(f"{_MODULE}._EXECUTORS", {"keyword": executor}):
            _run_validations_in_scope(_FakeModel(), _make_info(), TrainingMethod.SFT_LORA)
        executor.assert_not_called()

    @patch(f"{_MODULE}.DATASET_CHECKS")
    def test_executor_dispatched_for_matching_check(self, mock_checks):
        check = _check()
        mock_checks.__iter__ = lambda self: iter([check])
        executor = MagicMock()
        instance = _FakeModel()
        info = _make_info()
        with patch(f"{_MODULE}._EXECUTORS", {"keyword": executor}):
            _run_validations_in_scope(instance, info, TrainingMethod.SFT_LORA)
        executor.assert_called_once_with(instance, check, info.context)


if __name__ == "__main__":
    unittest.main()
