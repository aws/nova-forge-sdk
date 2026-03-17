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
"""Operation for displaying dataset rows."""

import json
from itertools import islice
from typing import Any

from ...util.iterator_utils import peek
from ...util.logging import logger
from .base import NovaForgeShowOperation


class ShowOperation(NovaForgeShowOperation):
    """Display the first n rows of the dataset."""

    def execute(self, loader: Any, **kwargs) -> None:
        """
        Display the first n rows of the current dataset.

        Args:
            loader: The DatasetLoader instance.
            n: Number of rows to display (default: 10).
        """
        n = kwargs.get("n", 10)

        dataset_iter = loader.dataset()
        peeked_value, dataset_iter = peek(dataset_iter)

        if peeked_value:
            logger.info("Showing dataset:")
            items = islice(dataset_iter, n)
            for i, row in enumerate(items):
                logger.info(json.dumps(row))
        else:
            logger.info("Dataset is empty. Call load() method to load data first")
