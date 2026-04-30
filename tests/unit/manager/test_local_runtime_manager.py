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
"""Tests for LocalRuntimeManager."""

import unittest

from amzn_nova_forge.manager.local_runtime_manager import LocalRuntimeManager
from amzn_nova_forge.model.model_enums import Platform


class TestLocalRuntimeManager(unittest.TestCase):
    def test_construction_sets_none_attributes(self):
        """Constructor should succeed with all attributes set to None."""
        mgr = LocalRuntimeManager()
        self.assertIsNone(mgr.instance_type)
        self.assertIsNone(mgr.instance_count)
        self.assertIsNone(mgr.kms_key_id)

    def test_execute_raises_not_implemented(self):
        """execute() should raise NotImplementedError with descriptive message."""
        mgr = LocalRuntimeManager()
        with self.assertRaises(NotImplementedError) as ctx:
            mgr.execute(job_config={})
        self.assertIn("LocalRuntimeManager does not submit remote jobs.", str(ctx.exception))

    def test_platform_returns_local(self):
        mgr = LocalRuntimeManager()
        self.assertEqual(mgr.platform, Platform.LOCAL)

    def test_setup_is_noop(self):
        mgr = LocalRuntimeManager()
        mgr.setup()  # should not raise

    def test_cleanup_is_noop(self):
        mgr = LocalRuntimeManager()
        mgr.cleanup("some-job-id")  # should not raise


if __name__ == "__main__":
    unittest.main()
