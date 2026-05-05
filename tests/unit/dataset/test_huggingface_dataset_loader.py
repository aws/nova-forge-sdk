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
"""Unit tests for HuggingFaceDatasetLoader.

All HuggingFace library calls are mocked via ``sys.modules`` patching so
no real network calls are made. Telemetry is auto-mocked globally by
``tests/unit/conftest.py::_mock_telemetry``.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

from amzn_nova_forge.dataset.huggingface_dataset_loader import (
    HuggingFaceDatasetLoader,
)
from amzn_nova_forge.dataset.operations.base import DataPrepError


def _make_fake_datasets_module(
    load_dataset_return_value=None,
    load_dataset_side_effect=None,
):
    """Build a fake ``datasets`` module with a mocked ``load_dataset``.

    Installable into ``sys.modules["datasets"]`` so the lazy
    ``from datasets import load_dataset`` inside ``load()`` resolves to it.
    """
    fake_mod = MagicMock()
    if load_dataset_side_effect is not None:
        fake_mod.load_dataset.side_effect = load_dataset_side_effect
    else:
        fake_mod.load_dataset.return_value = (
            [] if load_dataset_return_value is None else load_dataset_return_value
        )
    return fake_mod


def test_load_without_datasets_installed():
    """Missing ``datasets`` package → ImportError with install command."""
    # Setting sys.modules["datasets"] = None makes ``from datasets import ...``
    # raise ImportError.
    with patch.dict(sys.modules, {"datasets": None}):
        loader = HuggingFaceDatasetLoader()
        with pytest.raises(ImportError) as exc_info:
            loader.load("some/dataset")

        assert "pip install amzn-nova-forge[huggingface]" in str(exc_info.value)


def test_load_is_lazy():
    """``load()`` stores the factory; ``load_dataset()`` only runs on iteration."""
    fake_mod = _make_fake_datasets_module(load_dataset_return_value=iter([{"a": 1}]))
    with patch.dict(sys.modules, {"datasets": fake_mod}):
        loader = HuggingFaceDatasetLoader()
        loader.load("owner/ds", split="train")

        # No network/library calls yet.
        assert fake_mod.load_dataset.call_count == 0

        # Dataset accessor is a zero-arg factory; calling it returns the
        # generator but pulling the first record forces load_dataset().
        gen = loader.dataset()
        next(iter(gen))

        assert fake_mod.load_dataset.call_count >= 1


def test_generator_yields_dicts():
    """Generator yields plain ``dict`` objects in source order."""
    records = [
        {"id": 1, "text": "alpha"},
        {"id": 2, "text": "beta"},
        {"id": 3, "text": "gamma"},
    ]
    fake_mod = _make_fake_datasets_module(load_dataset_return_value=records)
    with patch.dict(sys.modules, {"datasets": fake_mod}):
        loader = HuggingFaceDatasetLoader()
        loader.load("owner/ds", split="train")

        out = list(loader.dataset())

        assert out == records
        for rec in out:
            assert isinstance(rec, dict)


def test_parameter_forwarding():
    """``path``, ``name``, ``split``, ``revision`` are forwarded to ``load_dataset``."""
    fake_mod = _make_fake_datasets_module(load_dataset_return_value=[])
    with patch.dict(sys.modules, {"datasets": fake_mod}):
        loader = HuggingFaceDatasetLoader()
        loader.load(
            "owner/multi_config_ds",
            split="validation",
            name="config_v2",
            revision="abc123",
        )

        # Force the generator to invoke load_dataset().
        list(loader.dataset())

        fake_mod.load_dataset.assert_called_once()
        call_args = fake_mod.load_dataset.call_args
        # First positional arg is the dataset path (normalized to hf://datasets/ form).
        assert call_args.args[0] == "hf://datasets/owner/multi_config_ds"
        assert call_args.kwargs["name"] == "config_v2"
        assert call_args.kwargs["split"] == "validation"
        assert call_args.kwargs["revision"] == "abc123"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"split": "train"},
        {"split": None, "name": None, "revision": None},
        {"split": "test", "name": "cfg", "revision": "v1.0"},
    ],
)
def test_streaming_always_true(kwargs):
    """``streaming=True`` is always forwarded, regardless of other params."""
    fake_mod = _make_fake_datasets_module(load_dataset_return_value=[])
    # When split=None the generator probes splits first; configure the
    # probe to return a single split so load_dataset is still reached.
    fake_mod.get_dataset_split_names.return_value = ["train"]
    with patch.dict(sys.modules, {"datasets": fake_mod}):
        loader = HuggingFaceDatasetLoader()
        loader.load("owner/ds", **kwargs)
        list(loader.dataset())

        fake_mod.load_dataset.assert_called_once()
        assert fake_mod.load_dataset.call_args.kwargs["streaming"] is True


def test_make_single_file_generator_raises():
    """``_make_single_file_generator`` raises NotImplementedError."""
    loader = HuggingFaceDatasetLoader()
    with pytest.raises(NotImplementedError) as exc_info:
        loader._make_single_file_generator("anything")

    assert "not file-based" in str(exc_info.value)


def test_class_attributes():
    """``_EXTENSIONS`` is empty, ``_FORMAT`` is ``"huggingface"``."""
    assert HuggingFaceDatasetLoader._EXTENSIONS == set()
    assert HuggingFaceDatasetLoader._FORMAT == "huggingface"


def test_load_returns_self():
    """``load()`` returns ``self`` and initializes session/state fields."""
    fake_mod = _make_fake_datasets_module(load_dataset_return_value=[])
    with patch.dict(sys.modules, {"datasets": fake_mod}):
        loader = HuggingFaceDatasetLoader()
        result = loader.load("owner/ds", split="train")

        assert result is loader
        assert loader._last_state is None
        assert loader._is_materialized is False


@pytest.mark.parametrize(
    "error_message",
    [
        "401 Client Error: Unauthorized",
        "403 Forbidden",
        "Unauthorized access to dataset",
    ],
)
def test_auth_error_mapping(error_message):
    """Auth failures (401/403/Unauthorized) → DataPrepError with HF_TOKEN guidance."""
    fake_mod = _make_fake_datasets_module(load_dataset_side_effect=RuntimeError(error_message))
    with patch.dict(sys.modules, {"datasets": fake_mod}):
        loader = HuggingFaceDatasetLoader()
        loader.load("private/ds", split="train")

        with pytest.raises(DataPrepError) as exc_info:
            list(loader.dataset())

        msg = str(exc_info.value)
        # Credential guidance mentions at least one of the documented paths.
        assert ("HF_TOKEN" in msg) or ("huggingface-cli login" in msg)


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("404 Client Error: Not Found"),
        RuntimeError("Dataset 'owner/missing_ds' doesn't exist on the Hub"),
        ConnectionError("connection reset"),
        TimeoutError("request timed out"),
        ValueError("BuilderConfig 'bad_name' not found"),
        ValueError("Invalid config for dataset"),
        RuntimeError("something unexpected happened"),
    ],
)
def test_default_error_mapping(error):
    """Non-auth errors → DataPrepError with dataset path and original error string."""
    fake_mod = _make_fake_datasets_module(load_dataset_side_effect=error)
    with patch.dict(sys.modules, {"datasets": fake_mod}):
        loader = HuggingFaceDatasetLoader()
        loader.load("owner/some_ds", split="train")

        with pytest.raises(DataPrepError) as exc_info:
            list(loader.dataset())

        msg = str(exc_info.value)
        # The dataset path is always included for context.
        assert "owner/some_ds" in msg
        # The original error string is surfaced so the user (or inspector of
        # __cause__) has the underlying signal.
        assert str(error) in msg
        # Original exception is chained via ``raise ... from error``.
        assert exc_info.value.__cause__ is error


@pytest.mark.parametrize(
    "bare_input,expected_normalized",
    [
        ("bar", "hf://datasets/bar"),
        ("faz/baz", "hf://datasets/faz/baz"),
    ],
)
def test_bare_identifier_normalized_in_library_calls(bare_input, expected_normalized):
    """Bare identifiers are normalized to ``hf://datasets/<path>`` for ``load_dataset``."""
    fake_mod = _make_fake_datasets_module(load_dataset_return_value=[])
    with patch.dict(sys.modules, {"datasets": fake_mod}):
        loader = HuggingFaceDatasetLoader()
        loader.load(bare_input, split="train")

        # Iterate to trigger load_dataset.
        list(loader.dataset())

        fake_mod.load_dataset.assert_called_once()
        assert fake_mod.load_dataset.call_args.args[0] == expected_normalized


@pytest.mark.parametrize(
    "prefixed_input",
    [
        "hf://datasets/faz/baz",
        "buckets/user/bucket/name",
    ],
)
def test_prefixed_paths_passthrough_to_library_calls(prefixed_input):
    """Prefixed paths are passed unchanged to ``load_dataset``."""
    fake_mod = _make_fake_datasets_module(load_dataset_return_value=[])
    with patch.dict(sys.modules, {"datasets": fake_mod}):
        loader = HuggingFaceDatasetLoader()
        loader.load(prefixed_input, split="train")

        # Iterate to trigger load_dataset.
        list(loader.dataset())

        fake_mod.load_dataset.assert_called_once()
        assert fake_mod.load_dataset.call_args.args[0] == prefixed_input


@pytest.mark.parametrize(
    "path,split,expected_load_path",
    [
        ("foo", None, "hf://foo"),
        ("faz", "train", "hf://faz/train"),
        ("hf://datasets/foo", None, "hf://datasets/foo"),
        ("hf://datasets/foo", "train", "hf://datasets/foo/train"),
        ("buckets/user/b/name", None, "hf://buckets/user/b/name"),
        ("single_component_name", None, "hf://single_component_name"),
        (
            "foo/bar",
            "train_sft",
            "hf://foo/bar/train_sft",
        ),
    ],
)
def test_load_path_reflects_user_input(path, split, expected_load_path):
    fake_mod = _make_fake_datasets_module()
    with patch.dict(sys.modules, {"datasets": fake_mod}):
        loader = HuggingFaceDatasetLoader()
        loader.load(path, split=split)

        assert loader._load_path == expected_load_path


@pytest.mark.parametrize(
    "data_files",
    [
        "https://example.com/data.json",
        ["a.json", "b.json"],
        {"train": "train.json", "test": ["test-a.json", "test-b.json"]},
    ],
    ids=["str", "list", "dict"],
)
def test_data_files_forwarded_when_provided(data_files):
    """``data_files`` is forwarded unchanged to ``load_dataset`` when non-None."""
    fake_mod = _make_fake_datasets_module(load_dataset_return_value=[])
    with patch.dict(sys.modules, {"datasets": fake_mod}):
        loader = HuggingFaceDatasetLoader()
        loader.load("owner/ds", split="train", data_files=data_files)

        list(loader.dataset())

        fake_mod.load_dataset.assert_called_once()
        assert fake_mod.load_dataset.call_args.kwargs["data_files"] == data_files


def test_data_dir_forwarded_when_provided():
    """``data_dir`` is forwarded unchanged to ``load_dataset`` when non-None."""
    fake_mod = _make_fake_datasets_module(load_dataset_return_value=[])
    with patch.dict(sys.modules, {"datasets": fake_mod}):
        loader = HuggingFaceDatasetLoader()
        loader.load("owner/ds", split="train", data_dir="en")

        list(loader.dataset())

        fake_mod.load_dataset.assert_called_once()
        assert fake_mod.load_dataset.call_args.kwargs["data_dir"] == "en"


@pytest.mark.parametrize("kwarg", ["data_files", "data_dir"])
def test_optional_kwargs_omitted_when_none(kwarg):
    """``data_files`` and ``data_dir`` are NOT in ``load_dataset`` kwargs when caller omits them."""
    fake_mod = _make_fake_datasets_module(load_dataset_return_value=[])
    with patch.dict(sys.modules, {"datasets": fake_mod}):
        loader = HuggingFaceDatasetLoader()
        loader.load("owner/ds", split="train")

        list(loader.dataset())

        fake_mod.load_dataset.assert_called_once()
        assert kwarg not in fake_mod.load_dataset.call_args.kwargs


import logging


def test_split_none_single_split_auto_selects(caplog):
    fake_mod = _make_fake_datasets_module(load_dataset_return_value=[{"x": 1}])
    fake_mod.get_dataset_split_names.return_value = ["only_split"]

    with patch.dict(sys.modules, {"datasets": fake_mod}):
        loader = HuggingFaceDatasetLoader()
        loader.load("owner/ds")  # split=None (default)

        with caplog.at_level(
            logging.INFO,
            logger="amzn_nova_forge.dataset.huggingface_dataset_loader",
        ):
            list(loader.dataset())

        # Probe was called once with the normalized path and forwarded params.
        fake_mod.get_dataset_split_names.assert_called_once_with(
            "hf://datasets/owner/ds",
            config_name=None,
            data_files=None,
            revision=None,
            data_dir=None,
        )

        # load_dataset received the auto-selected split.
        assert fake_mod.load_dataset.call_args.kwargs["split"] == "only_split"

        # INFO log emitted.
        assert any(
            "No split specified for 'owner/ds'; auto-selected 'only_split'" in rec.message
            for rec in caplog.records
        )

        # _load_path is NOT rewritten to include the auto-selected split.
        assert loader._load_path == "hf://owner/ds"


@pytest.mark.parametrize(
    "split_names",
    [
        ["train", "test"],
        ["train", "test", "validation"],
        ["a", "b", "c", "d"],
    ],
    ids=["2-splits", "3-splits", "4-splits"],
)
def test_split_none_multi_split_raises_with_guidance(split_names):
    fake_mod = _make_fake_datasets_module(load_dataset_return_value=[])
    fake_mod.get_dataset_split_names.return_value = split_names

    with patch.dict(sys.modules, {"datasets": fake_mod}):
        loader = HuggingFaceDatasetLoader()
        loader.load("owner/ds")  # split=None

        with pytest.raises(DataPrepError) as exc_info:
            list(loader.dataset())

        msg = str(exc_info.value)

        # Message contains the user-supplied path.
        assert "owner/ds" in msg

        # Message contains every split name.
        for name in split_names:
            assert name in msg

        # Message contains the concatenation syntax wording.
        assert "a+b" in msg or "+" in msg

        # Message contains a concrete example invocation.
        example = f"loader.load('owner/ds', split='{split_names[0]}+{split_names[1]}')"
        assert example in msg

        # Deliberate rejection — no __cause__ chain.
        assert exc_info.value.__cause__ is None

        # load_dataset was NOT called (probe rejected before reaching it).
        fake_mod.load_dataset.assert_not_called()


@pytest.mark.parametrize(
    "probe_error",
    [
        RuntimeError("401 Client Error"),
        RuntimeError("403 Forbidden"),
        RuntimeError("Unauthorized access"),
    ],
    ids=["401", "403", "Unauthorized"],
)
def test_split_none_probe_auth_error(probe_error):
    fake_mod = _make_fake_datasets_module(load_dataset_return_value=[])
    fake_mod.get_dataset_split_names.side_effect = probe_error

    with patch.dict(sys.modules, {"datasets": fake_mod}):
        loader = HuggingFaceDatasetLoader()
        loader.load("owner/ds")  # split=None

        with pytest.raises(DataPrepError) as exc_info:
            list(loader.dataset())

        msg = str(exc_info.value)
        assert ("HF_TOKEN" in msg) or ("huggingface-cli login" in msg)

        # Original exception chained via ``from error``.
        assert exc_info.value.__cause__ is probe_error


@pytest.mark.parametrize(
    "probe_error",
    [
        RuntimeError("404 Client Error"),
        ConnectionError("connection reset"),
        TimeoutError("request timed out"),
        ValueError("BuilderConfig 'bad_name' not found"),
    ],
    ids=["404", "ConnectionError", "TimeoutError", "ValueError"],
)
def test_split_none_probe_non_auth_error(probe_error):
    fake_mod = _make_fake_datasets_module(load_dataset_return_value=[])
    fake_mod.get_dataset_split_names.side_effect = probe_error

    with patch.dict(sys.modules, {"datasets": fake_mod}):
        loader = HuggingFaceDatasetLoader()
        loader.load("owner/ds")  # split=None

        with pytest.raises(DataPrepError) as exc_info:
            list(loader.dataset())

        msg = str(exc_info.value)
        # Message contains the user-supplied path.
        assert "owner/ds" in msg
        # Message contains the original error string.
        assert str(probe_error) in msg

        # Original exception chained via ``from error``.
        assert exc_info.value.__cause__ is probe_error


def test_split_non_none_skips_probe():
    fake_mod = _make_fake_datasets_module(load_dataset_return_value=[])

    with patch.dict(sys.modules, {"datasets": fake_mod}):
        loader = HuggingFaceDatasetLoader()
        loader.load("owner/ds", split="train")

        list(loader.dataset())

        # Probe was never invoked.
        fake_mod.get_dataset_split_names.assert_not_called()

        # load_dataset received the user-supplied split.
        assert fake_mod.load_dataset.call_args.kwargs["split"] == "train"


def test_load_time_is_still_lazy_for_split_none():
    fake_mod = _make_fake_datasets_module(load_dataset_return_value=[])
    fake_mod.get_dataset_split_names.return_value = ["train"]

    with patch.dict(sys.modules, {"datasets": fake_mod}):
        loader = HuggingFaceDatasetLoader()
        loader.load("owner/ds")  # split=None

        # Immediately after load(), no probe call yet.
        assert fake_mod.get_dataset_split_names.call_count == 0


def test_no_token_parameter():
    fake_mod = _make_fake_datasets_module(load_dataset_return_value=[])
    with patch.dict(sys.modules, {"datasets": fake_mod}):
        loader = HuggingFaceDatasetLoader()
        with pytest.raises(TypeError):
            loader.load("foo", token="abc")


def test_lazy_import_site_shared():
    with patch.dict(sys.modules, {"datasets": None}):
        loader = HuggingFaceDatasetLoader()
        with pytest.raises(ImportError) as exc_info:
            loader.load("foo")

        assert "pip install amzn-nova-forge[huggingface]" in str(exc_info.value)
