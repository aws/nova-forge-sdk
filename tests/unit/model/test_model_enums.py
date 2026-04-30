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
import unittest

from amzn_nova_forge.core.enums import (
    DeployPlatform,
    Model,
    Platform,
    TrainingMethod,
    Version,
)


class TestEnumsDynamic(unittest.TestCase):
    def test_platform_enum_values(self):
        for member in Platform:
            self.assertEqual(member.value, member.name)

    def test_version_enum_auto_values(self):
        for i, member in enumerate(Version, start=1):
            self.assertEqual(member.value, i)

    def test_model_enum_attributes(self):
        for model in Model:
            self.assertTrue(hasattr(model, "version"))
            self.assertTrue(hasattr(model, "model_type"))
            self.assertTrue(hasattr(model, "model_path"))
            self.assertIsInstance(model.version, Version)
            self.assertIsInstance(model.model_type, str)
            self.assertIsInstance(model.model_path, str)
            self.assertEqual(Model.from_model_type(model.model_type), model)

    def test_model_from_model_type_failure(self):
        with self.assertRaises(ValueError):
            Model.from_model_type("nonexistent-model")

    def test_training_method_enum(self):
        for member in TrainingMethod:
            self.assertIsInstance(member.value, str)
            self.assertTrue(len(member.value) > 0)

    def test_deploy_platform_enum(self):
        for member in DeployPlatform:
            self.assertIsInstance(member.value, str)
            self.assertTrue(len(member.value) > 0)


if __name__ == "__main__":
    unittest.main()
