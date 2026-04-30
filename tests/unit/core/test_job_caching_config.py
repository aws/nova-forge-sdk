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
"""Unit tests for JobCachingConfig Pydantic model."""

import unittest

from pydantic import ValidationError

from amzn_nova_forge.core.job_cache import JobCachingConfig
from amzn_nova_forge.core.result.job_result import JobStatus


class TestJobCachingConfigDefaults(unittest.TestCase):
    """Test that defaults match the old _default_job_caching_config() dict."""

    def test_include_core_defaults_true(self):
        config = JobCachingConfig()
        self.assertTrue(config.include_core)

    def test_include_recipe_defaults_true(self):
        config = JobCachingConfig()
        self.assertTrue(config.include_recipe)

    def test_include_infra_defaults_false(self):
        config = JobCachingConfig()
        self.assertFalse(config.include_infra)

    def test_include_params_defaults_empty(self):
        config = JobCachingConfig()
        self.assertEqual(config.include_params, [])

    def test_exclude_params_defaults_empty(self):
        config = JobCachingConfig()
        self.assertEqual(config.exclude_params, [])

    def test_allowed_statuses_defaults(self):
        config = JobCachingConfig()
        self.assertEqual(
            config.allowed_statuses,
            [JobStatus.COMPLETED, JobStatus.IN_PROGRESS],
        )

    def test_default_factory_returns_independent_lists(self):
        """Each instance should get its own list, not a shared mutable default."""
        c1 = JobCachingConfig()
        c2 = JobCachingConfig()
        c1.include_params.append("extra")
        self.assertEqual(c2.include_params, [])


class TestJobCachingConfigExplicitKwargs(unittest.TestCase):
    """Test construction from explicit keyword arguments."""

    def test_all_fields_explicit(self):
        config = JobCachingConfig(
            include_core=False,
            include_recipe=False,
            include_infra=True,
            include_params=["custom_param"],
            exclude_params=["model"],
            allowed_statuses=[JobStatus.COMPLETED],
        )
        self.assertFalse(config.include_core)
        self.assertFalse(config.include_recipe)
        self.assertTrue(config.include_infra)
        self.assertEqual(config.include_params, ["custom_param"])
        self.assertEqual(config.exclude_params, ["model"])
        self.assertEqual(config.allowed_statuses, [JobStatus.COMPLETED])

    def test_partial_override(self):
        config = JobCachingConfig(include_infra=True)
        # Overridden field
        self.assertTrue(config.include_infra)
        # Defaults preserved for non-overridden fields
        self.assertTrue(config.include_core)
        self.assertTrue(config.include_recipe)
        self.assertEqual(config.include_params, [])

    def test_empty_allowed_statuses(self):
        config = JobCachingConfig(allowed_statuses=[])
        self.assertEqual(config.allowed_statuses, [])

    def test_single_allowed_status(self):
        config = JobCachingConfig(allowed_statuses=[JobStatus.FAILED])
        self.assertEqual(config.allowed_statuses, [JobStatus.FAILED])


class TestJobCachingConfigValidation(unittest.TestCase):
    """Test that Pydantic rejects invalid types at construction time."""

    def test_invalid_include_core_type_rejected(self):
        """Non-bool-coercible value for include_core should raise ValidationError."""
        with self.assertRaises(ValidationError):
            JobCachingConfig(include_core="not_a_bool")

    def test_invalid_allowed_statuses_string_rejected(self):
        with self.assertRaises(ValidationError):
            JobCachingConfig(allowed_statuses="not-a-list")

    def test_unknown_status_string_coerces_to_failed(self):
        """JobStatus._missing_ maps unknown strings to FAILED, so Pydantic coerces."""
        config = JobCachingConfig(allowed_statuses=["not-a-status"])
        self.assertEqual(config.allowed_statuses, [JobStatus.FAILED])

    def test_invalid_include_params_type_rejected(self):
        with self.assertRaises(ValidationError):
            JobCachingConfig(include_params="not-a-list")

    def test_bool_coercion_for_include_core(self):
        """Pydantic coerces int 0/1 to bool for bool fields."""
        config_false = JobCachingConfig(include_core=0)
        self.assertFalse(config_false.include_core)
        config_true = JobCachingConfig(include_core=1)
        self.assertTrue(config_true.include_core)


if __name__ == "__main__":
    unittest.main()
