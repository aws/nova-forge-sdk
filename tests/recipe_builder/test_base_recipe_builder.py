import os
import tempfile
import unittest
from typing import Dict, Optional
from unittest.mock import MagicMock, patch

import yaml

from amzn_nova_customization_sdk.model.model_enums import Platform, TrainingMethod
from amzn_nova_customization_sdk.recipe_builder.base_recipe_builder import (
    BaseRecipeBuilder,
)
from amzn_nova_customization_sdk.recipe_config.base_recipe_config import (
    BaseRecipeConfig,
    BaseRunConfig,
)


# ----------------------------------------------------------------------
# Concrete implementation for testing
# ----------------------------------------------------------------------
class TestRecipeBuilder(BaseRecipeBuilder):
    def _validate_user_input(
        self, validation_config: Optional[Dict[str, bool]] = None
    ) -> None:
        # Minimal validation so tests can run
        if self.instance_count <= 0:
            raise ValueError("instance_count must be > 0")

    def _build_recipe_config(self) -> BaseRecipeConfig:
        run = self._create_base_run_config()
        return BaseRecipeConfig(run=run)


class TestBaseRecipeBuilder(unittest.TestCase):
    def setUp(self):
        self.job_name = "unit-test-job"
        self.method = TrainingMethod.SFT_LORA
        self.model_type = "test-model"
        self.model_path = "models/test"
        self.instance_type = "ml.g5.12xlarge"
        self.data_s3 = "s3://bucket/data"
        self.output_s3 = "s3://bucket/output"
        self.instance_count = 1
        self.overrides = {"test_key": "override_val"}

    def test_initialization(self):
        builder = TestRecipeBuilder(
            job_name=self.job_name,
            platform=Platform.SMTJ,
            method=self.method,
            model_type=self.model_type,
            model_path=self.model_path,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            data_s3_path=self.data_s3,
            output_s3_path=self.output_s3,
            overrides=self.overrides,
        )

        self.assertEqual(builder.job_name, self.job_name)
        self.assertEqual(builder.platform, Platform.SMTJ)
        self.assertEqual(builder.method, self.method)
        self.assertEqual(builder.model_type, self.model_type)
        self.assertEqual(builder.model_path, self.model_path)
        self.assertEqual(builder.data_s3_path, self.data_s3)
        self.assertEqual(builder.output_s3_path, self.output_s3)
        self.assertEqual(builder.instance_count, 1)
        self.assertEqual(builder.overrides["test_key"], "override_val")

    def test_create_base_run_config(self):
        builder = TestRecipeBuilder(
            job_name=self.job_name,
            platform=Platform.SMTJ,
            method=self.method,
            model_type=self.model_type,
            model_path=self.model_path,
            instance_type=self.instance_type,
            instance_count=2,
            data_s3_path=self.data_s3,
            output_s3_path=self.output_s3,
            overrides={},
        )

        run_config = builder._create_base_run_config()
        self.assertIsInstance(run_config, BaseRunConfig)
        self.assertEqual(run_config.name, self.job_name)
        self.assertEqual(run_config.model_type, self.model_type)
        self.assertEqual(run_config.model_name_or_path, self.model_path)
        self.assertEqual(run_config.data_s3_path, self.data_s3)
        self.assertEqual(run_config.output_s3_path, self.output_s3)
        self.assertEqual(run_config.replicas, 2)

    def test_get_value_override(self):
        builder = TestRecipeBuilder(
            job_name=self.job_name,
            platform=Platform.SMTJ,
            method=self.method,
            model_type=self.model_type,
            model_path=self.model_path,
            instance_type=self.instance_type,
            instance_count=1,
            data_s3_path=self.data_s3,
            output_s3_path=self.output_s3,
            overrides={"foo": "bar"},
        )

        result = builder._get_value("foo", lambda: "default")
        self.assertEqual(result, "bar")

    def test_get_value_default(self):
        builder = TestRecipeBuilder(
            job_name=self.job_name,
            platform=Platform.SMTJ,
            method=self.method,
            model_type=self.model_type,
            model_path=self.model_path,
            instance_type=self.instance_type,
            instance_count=1,
            data_s3_path=self.data_s3,
            output_s3_path=self.output_s3,
            overrides={},
        )

        result = builder._get_value("missing", lambda: "default")
        self.assertEqual(result, "default")

    def test_generate_recipe_path_smtj(self):
        builder = TestRecipeBuilder(
            job_name=self.job_name,
            platform=Platform.SMTJ,
            method=self.method,
            model_type=self.model_type,
            model_path=self.model_path,
            instance_type=self.instance_type,
            instance_count=1,
            data_s3_path=self.data_s3,
            output_s3_path=self.output_s3,
            overrides={},
        )

        recipe_path = builder._generate_recipe_path(None)
        path = recipe_path.path

        self.assertTrue(path.endswith(".yaml"))

    @patch(
        "builtins.__import__", side_effect=ModuleNotFoundError("hyperpod_cli missing")
    )
    def test_generate_recipe_path_missing_hyperpod_cli(self, _):
        builder = TestRecipeBuilder(
            job_name=self.job_name,
            platform=Platform.SMHP,
            method=self.method,
            model_type=self.model_type,
            model_path=self.model_path,
            instance_type=self.instance_type,
            instance_count=1,
            data_s3_path=self.data_s3,
            output_s3_path=self.output_s3,
            overrides={},
        )

        with self.assertRaises(RuntimeError):
            builder._generate_recipe_path(None)

    def test_generate_recipe_path_hyperpod(self):
        mock_hyperpod_cli = MagicMock()
        mock_hyperpod_cli.__file__ = "/tmp/hyperpod_cli/__init__.py"

        with patch.dict("sys.modules", {"hyperpod_cli": mock_hyperpod_cli}):
            builder = TestRecipeBuilder(
                job_name=self.job_name,
                platform=Platform.SMHP,
                method=self.method,
                model_type=self.model_type,
                model_path=self.model_path,
                instance_type=self.instance_type,
                instance_count=1,
                data_s3_path=self.data_s3,
                output_s3_path=self.output_s3,
                overrides={},
            )

            recipe_path = builder._generate_recipe_path(None)
            path = recipe_path.path

            self.assertIn("sagemaker_hyperpod_recipes", path)
            self.assertIn("fine-tuning", path)
            self.assertTrue(path.endswith(".yaml"))

    def test_build_writes_yaml_file(self):
        builder = TestRecipeBuilder(
            job_name=self.job_name,
            platform=Platform.SMTJ,
            method=self.method,
            model_type=self.model_type,
            model_path=self.model_path,
            instance_type=self.instance_type,
            instance_count=1,
            data_s3_path=self.data_s3,
            output_s3_path=self.output_s3,
            overrides={},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "recipe.yaml")
            builder.build(file_path)

            self.assertTrue(os.path.exists(file_path))

            with open(file_path, "r") as f:
                config = yaml.safe_load(f)

            self.assertIn("run", config)
            self.assertEqual(config["run"]["name"], self.job_name)
            self.assertEqual(config["run"]["model_type"], self.model_type)
            self.assertEqual(config["run"]["replicas"], 1)

    def test_build_invalid_input(self):
        builder = TestRecipeBuilder(
            job_name=self.job_name,
            platform=Platform.SMTJ,
            method=self.method,
            model_type=self.model_type,
            model_path=self.model_path,
            instance_type=self.instance_type,
            instance_count=0,  # invalid
            data_s3_path=self.data_s3,
            output_s3_path=self.output_s3,
            overrides={},
        )

        with self.assertRaises(ValueError):
            builder.build()


if __name__ == "__main__":
    unittest.main()
