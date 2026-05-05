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
"""HuggingFace dataset loader."""

from typing import Mapping, Optional, Sequence, Union

from ..telemetry import Feature, _telemetry_emitter
from ..util.logging import logger
from .dataset_loader import DatasetLoader
from .operations.base import DataPrepError

_HF_URL_PREFIX = "hf://datasets/"
_HF_BUCKETS_PREFIX = "buckets/"


class HuggingFaceDatasetLoader(DatasetLoader):
    """Load datasets from HuggingFace Hub into the Nova Forge pipeline."""

    _EXTENSIONS: set[str] = set()
    _FORMAT: str = "huggingface"

    def _make_single_file_generator(self, path: str):
        raise NotImplementedError(
            "HuggingFace datasets are not file-based. Use load(path, split=...) instead."
        )

    @staticmethod
    def _normalize_hf_path(path: str) -> str:
        """Prefix plain HF identifiers with ``hf://datasets/`` for the datasets library.
        Paths already starting with ``hf://datasets/`` or ``buckets/`` pass through unchanged.
        We do this to make sure HF library will not load from local files even if user passes a local path
        """
        if path.startswith(_HF_URL_PREFIX) or path.startswith(_HF_BUCKETS_PREFIX):
            return path
        return f"{_HF_URL_PREFIX}{path}"

    @staticmethod
    def _handle_hf_error(error: Exception, dataset_path: str) -> None:
        """Map HuggingFace exceptions to DataPrepError with actionable messages."""
        error_str = str(error)

        if "401" in error_str or "403" in error_str or "Unauthorized" in error_str:
            raise DataPrepError(
                f"Failed to load dataset '{dataset_path}'. "
                "Ensure your HuggingFace credentials are set using the HF_TOKEN "
                "environment variable or run 'huggingface-cli login', and that you "
                f"have access to the dataset. Original error: {error_str}"
            ) from error

        raise DataPrepError(
            f"Failed to load HuggingFace dataset '{dataset_path}': {error_str}"
        ) from error

    @staticmethod
    def _resolve_split(
        user_path: str,
        normalized_path: str,
        config_name: Optional[str],
        revision: Optional[str],
        data_files: Optional[Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]],
        data_dir: Optional[str] = None,
    ) -> str:
        """Auto-select split when only one exists, raise with guidance for multiple."""
        from datasets import get_dataset_split_names

        split_names = get_dataset_split_names(
            normalized_path,
            config_name=config_name,
            data_files=data_files,
            revision=revision,
            data_dir=data_dir,
        )
        if len(split_names) == 1:
            logger.info(
                "No split specified for '%s'; auto-selected '%s'",
                user_path,
                split_names[0],
            )
            return split_names[0]

        raise DataPrepError(
            f"Dataset '{user_path}' has multiple splits "
            f"({split_names}). "
            "Pass split=... to select one, or use the "
            "HuggingFace 'a+b+c' concatenation syntax to "
            "stream multiple splits as one sequence. "
            f"Example: loader.load('{user_path}', "
            f"split='{split_names[0]}+{split_names[1]}')"
        )

    @_telemetry_emitter(Feature.DATA_PREP, "load")
    def load(
        self,
        path: str,
        split: Optional[str] = None,
        name: Optional[str] = None,
        revision: Optional[str] = None,
        data_files: Optional[
            Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]
        ] = None,
        data_dir: Optional[str] = None,
    ) -> "HuggingFaceDatasetLoader":
        """Load a dataset from HuggingFace Hub.

        Args:
            path: Dataset identifier.
            split: Split name (e.g. "train"). None auto-selects if only one exists.
            name: Configuration name for multi-config datasets.
            revision: Git tag or commit hash to pin the dataset version.
            data_files: Specific file(s) within the dataset repo to load.
            data_dir: Subdirectory to load from.

        Returns:
            self (for method chaining)

        Raises:
            ImportError: If the datasets package is not installed.
            DataPrepError: If the dataset cannot be loaded.
        """
        try:
            from datasets import get_dataset_split_names, load_dataset
        except ImportError:
            raise ImportError(
                "The 'datasets' package is required for HuggingFace integration. "
                "Install it with: pip install amzn-nova-forge[huggingface]"
            )

        normalized_path = self._normalize_hf_path(path)
        hf_split = split
        hf_name = name
        hf_revision = revision
        hf_data_files = data_files
        hf_data_dir = data_dir

        def _generator():
            try:
                effective_split = hf_split or self._resolve_split(
                    path,
                    normalized_path,
                    hf_name,
                    hf_revision,
                    hf_data_files,
                    hf_data_dir,
                )

                load_kwargs = {
                    "name": hf_name,
                    "split": effective_split,
                    "revision": hf_revision,
                    "streaming": True,
                }
                if hf_data_files is not None:
                    load_kwargs["data_files"] = hf_data_files
                if hf_data_dir is not None:
                    load_kwargs["data_dir"] = hf_data_dir

                for record in load_dataset(normalized_path, **load_kwargs):
                    yield dict(record)
            except DataPrepError:
                raise
            except Exception as e:
                self._handle_hf_error(e, path)

        self.dataset = _generator

        base = path if path.startswith("hf://") else f"hf://{path}"
        if split:
            self._load_path = f"{base}/{split}"
        else:
            self._load_path = base

        self._last_state = None
        self._is_materialized = False
        self._session_id = None
        return self
