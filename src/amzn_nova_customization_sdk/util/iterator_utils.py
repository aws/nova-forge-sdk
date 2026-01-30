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
Iterator utility functions for working with iterators and generators.
"""

from itertools import tee
from typing import Iterator, Tuple, TypeVar

T = TypeVar("T")


def peek(iterator: Iterator[T]) -> Tuple[T | None, Iterator[T]]:
    """
    Look at the next value without consuming it from the original iterator.

    This function uses itertools.tee() to create two independent iterators
    sharing the same underlying data. One is used to peek at the next value,
    and the other is returned for continued iteration. The flattening property
    of tee() ensures that nested peek() calls remain efficient.

    Args:
        iterator: The iterator to peek into

    Returns:
        A tuple of (peeked_value, iterator) where:
        - peeked_value is the next value from the iterator, or None if exhausted
        - iterator is the original iterator that can continue to be used

    Example:
        >>> it = iter([1, 2, 3])
        >>> peeked_value, it = peek(it)
        >>> print(f"Next value will be: {peeked_value}")  # 1
        >>> print(f"Actual next: {next(it)}")              # 1
    """
    if not iterator:
        return None, iterator

    a, b = tee(iterator)
    try:
        return next(a), b
    except StopIteration:
        return None, b
