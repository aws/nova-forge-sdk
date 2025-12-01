import os
import tempfile
import unittest

import yaml

import amzn_nova_customization_sdk.recipe_config.v_two.rft_config_smhp as smhp
import amzn_nova_customization_sdk.recipe_config.v_two.rft_config_smtj as smtj
from amzn_nova_customization_sdk.model.model_enums import (
    Model,
    Platform,
    TrainingMethod,
)
from amzn_nova_customization_sdk.recipe_builder.rft_recipe_builder import (
    RFTRecipeBuilder,
)


class TestRFTRecipeBuilder(unittest.TestCase):
    def setUp(self):
        self.job_name = "test-rft-job"
        self.data_s3_path = "s3://test-bucket/data"
        self.output_s3_path = "s3://test-bucket/output"
        self.lambda_arn = "arn:aws:lambda:us-east-1:123456789012:function:test-fn"

    def test_rft_builder_initialization_smhp(self):
        builder = RFTRecipeBuilder(
            job_name=self.job_name,
            platform=Platform.SMHP,
            model=Model.NOVA_LITE_2,
            method=TrainingMethod.RFT_LORA,
            instance_type="ml.p5.48xlarge",
            instance_count=2,
            data_s3_path=self.data_s3_path,
            output_s3_path=self.output_s3_path,
            rft_lambda_arn=self.lambda_arn,
            overrides={},
        )

        self.assertEqual(builder.job_name, self.job_name)
        self.assertEqual(builder.platform, Platform.SMHP)
        self.assertEqual(builder.model, Model.NOVA_LITE_2)
        self.assertEqual(builder.method, TrainingMethod.RFT_LORA)
        self.assertEqual(builder.instance_type, "ml.p5.48xlarge")
        self.assertEqual(builder.instance_count, 2)
        self.assertEqual(builder.data_s3_path, self.data_s3_path)
        self.assertEqual(builder.output_s3_path, self.output_s3_path)
        self.assertEqual(builder.rft_lambda_arn, self.lambda_arn)
        self.assertEqual(builder.overrides, {})

    def test_rft_lora_config_build_smhp(self):
        builder = RFTRecipeBuilder(
            job_name=self.job_name,
            platform=Platform.SMHP,
            model=Model.NOVA_LITE_2,
            method=TrainingMethod.RFT_LORA,
            instance_type="ml.p5.48xlarge",
            instance_count=2,
            data_s3_path=self.data_s3_path,
            output_s3_path=self.output_s3_path,
            rft_lambda_arn=self.lambda_arn,
            overrides={"alpha": 32},
        )

        recipe = builder._build_recipe_config()

        self.assertIsInstance(recipe, smhp.RFTRecipeConfig)

        self.assertEqual(recipe.run.reward_lambda_arn, self.lambda_arn)
        self.assertEqual(recipe.run.replicas, 2)

        self.assertIsNotNone(recipe.training_config.trainer.peft)
        self.assertEqual(
            recipe.training_config.trainer.peft.peft_scheme, smhp.PeftScheme.LORA
        )
        self.assertEqual(recipe.training_config.trainer.peft.lora_tuning.alpha, 32)

    def test_rft_fullrank_config_build_smhp(self):
        builder = RFTRecipeBuilder(
            job_name=self.job_name,
            platform=Platform.SMHP,
            model=Model.NOVA_LITE_2,
            method=TrainingMethod.RFT,
            instance_type="ml.p5.48xlarge",
            instance_count=3,
            data_s3_path=self.data_s3_path,
            output_s3_path=self.output_s3_path,
            rft_lambda_arn=self.lambda_arn,
            overrides={},
        )

        recipe = builder._build_recipe_config()

        self.assertIsInstance(recipe, smhp.RFTRecipeConfig)
        self.assertIsNone(recipe.training_config.trainer.peft)

    def test_rft_overrides_apply_smhp(self):
        overrides = {
            "name": "overridden",
            "lr": 1e-4,
            "temperature": 2,
            "number_generation": 10,
        }

        builder = RFTRecipeBuilder(
            job_name=self.job_name,
            platform=Platform.SMHP,
            model=Model.NOVA_LITE_2,
            method=TrainingMethod.RFT,
            instance_type="ml.p5.48xlarge",
            instance_count=2,
            data_s3_path=self.data_s3_path,
            output_s3_path=self.output_s3_path,
            rft_lambda_arn=self.lambda_arn,
            overrides=overrides,
        )

        recipe = builder._build_recipe_config()

        self.assertIsInstance(recipe, smhp.RFTRecipeConfig)
        self.assertEqual(recipe.run.name, "overridden")
        self.assertEqual(recipe.training_config.trainer.optim_config.lr, 1e-4)
        self.assertEqual(recipe.training_config.rollout.generator.temperature, 2)
        self.assertEqual(
            recipe.training_config.rollout.advantage_strategy.number_generation, 10
        )

    def test_rft_yaml_output_smhp(self):
        builder = RFTRecipeBuilder(
            job_name=self.job_name,
            platform=Platform.SMHP,
            model=Model.NOVA_LITE_2,
            method=TrainingMethod.RFT_LORA,
            instance_type="ml.p5.48xlarge",
            instance_count=4,
            data_s3_path=self.data_s3_path,
            output_s3_path=self.output_s3_path,
            rft_lambda_arn=self.lambda_arn,
            overrides={},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test_rft_recipe_smhp.yaml")
            output_path = builder.build(
                file_path, validation_config={"infra": False, "iam": False}
            )

            self.assertEqual(output_path.path, file_path)
            self.assertTrue(os.path.exists(file_path))

            with open(file_path, "r") as f:
                config = yaml.safe_load(f)

            self.assertIn("run", config)
            self.assertIn("training_config", config)
            self.assertEqual(config["run"]["name"], self.job_name)
            self.assertEqual(config["run"]["reward_lambda_arn"], self.lambda_arn)
            self.assertEqual(config["run"]["replicas"], 4)

    def test_rft_builder_initialization_smtj(self):
        builder = RFTRecipeBuilder(
            job_name=self.job_name,
            platform=Platform.SMTJ,
            model=Model.NOVA_LITE_2,
            method=TrainingMethod.RFT_LORA,
            instance_type="ml.p5.48xlarge",
            instance_count=2,
            data_s3_path=self.data_s3_path,
            output_s3_path=self.output_s3_path,
            rft_lambda_arn=self.lambda_arn,
            overrides={},
        )

        self.assertEqual(builder.job_name, self.job_name)
        self.assertEqual(builder.platform, Platform.SMTJ)
        self.assertEqual(builder.model, Model.NOVA_LITE_2)
        self.assertEqual(builder.method, TrainingMethod.RFT_LORA)
        self.assertEqual(builder.instance_type, "ml.p5.48xlarge")
        self.assertEqual(builder.instance_count, 2)
        self.assertEqual(builder.data_s3_path, self.data_s3_path)
        self.assertEqual(builder.output_s3_path, self.output_s3_path)
        self.assertEqual(builder.rft_lambda_arn, self.lambda_arn)
        self.assertEqual(builder.overrides, {})

    def test_rft_lora_config_build_smtj(self):
        builder = RFTRecipeBuilder(
            job_name=self.job_name,
            platform=Platform.SMTJ,
            model=Model.NOVA_LITE_2,
            method=TrainingMethod.RFT_LORA,
            instance_type="ml.p5.48xlarge",
            instance_count=2,
            data_s3_path=self.data_s3_path,
            output_s3_path=self.output_s3_path,
            rft_lambda_arn=self.lambda_arn,
            overrides={"alpha": 32},
        )

        recipe = builder._build_recipe_config()

        self.assertIsInstance(recipe, smtj.RFTRecipeConfig)

        self.assertEqual(recipe.run.reward_lambda_arn, self.lambda_arn)
        self.assertEqual(recipe.run.replicas, 2)

        self.assertIsNotNone(recipe.training_config.model.peft)
        self.assertEqual(
            recipe.training_config.model.peft.peft_scheme, smtj.PeftScheme.LORA
        )
        self.assertEqual(recipe.training_config.model.peft.lora_tuning.alpha, 32)

    def test_rft_fullrank_config_build_smtj(self):
        builder = RFTRecipeBuilder(
            job_name=self.job_name,
            platform=Platform.SMTJ,
            model=Model.NOVA_LITE_2,
            method=TrainingMethod.RFT,
            instance_type="ml.p5.48xlarge",
            instance_count=3,
            data_s3_path=self.data_s3_path,
            output_s3_path=self.output_s3_path,
            rft_lambda_arn=self.lambda_arn,
            overrides={},
        )

        recipe = builder._build_recipe_config()

        self.assertIsInstance(recipe, smtj.RFTRecipeConfig)
        self.assertEqual(
            recipe.training_config.model.peft.peft_scheme, smtj.PeftScheme.NULL
        )

    def test_rft_overrides_apply_smtj(self):
        overrides = {
            "name": "overridden",
            "lr": 1e-4,
            "temperature": 2,
            "number_generation": 10,
        }

        builder = RFTRecipeBuilder(
            job_name=self.job_name,
            platform=Platform.SMTJ,
            model=Model.NOVA_LITE_2,
            method=TrainingMethod.RFT,
            instance_type="ml.p5.48xlarge",
            instance_count=2,
            data_s3_path=self.data_s3_path,
            output_s3_path=self.output_s3_path,
            rft_lambda_arn=self.lambda_arn,
            overrides=overrides,
        )

        recipe = builder._build_recipe_config()

        self.assertIsInstance(recipe, smtj.RFTRecipeConfig)
        self.assertEqual(recipe.run.name, "overridden")
        self.assertEqual(recipe.training_config.trainer.optim.lr, 1e-4)
        self.assertEqual(recipe.training_config.rollout.generator.temperature, 2)
        self.assertEqual(
            recipe.training_config.rollout.advantage_strategy.number_generation, 10
        )

    def test_rft_yaml_output_smtj(self):
        builder = RFTRecipeBuilder(
            job_name=self.job_name,
            platform=Platform.SMTJ,
            model=Model.NOVA_LITE_2,
            method=TrainingMethod.RFT_LORA,
            instance_type="ml.p5.48xlarge",
            instance_count=4,
            data_s3_path=self.data_s3_path,
            output_s3_path=self.output_s3_path,
            rft_lambda_arn=self.lambda_arn,
            overrides={},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test_rft_recipe_smtj.yaml")
            output_path = builder.build(file_path)

            self.assertEqual(output_path.path, file_path)
            self.assertTrue(os.path.exists(file_path))

            with open(file_path, "r") as f:
                config = yaml.safe_load(f)

            self.assertIn("run", config)
            self.assertIn("training_config", config)
            self.assertEqual(config["run"]["name"], self.job_name)
            self.assertEqual(config["run"]["reward_lambda_arn"], self.lambda_arn)
            self.assertEqual(config["run"]["replicas"], 4)

    def test_invalid_instance_type_raises_smhp(self):
        builder = RFTRecipeBuilder(
            job_name=self.job_name,
            platform=Platform.SMHP,
            model=Model.NOVA_LITE_2,
            method=TrainingMethod.RFT,
            instance_type="ml.invalid.instance",
            instance_count=2,
            data_s3_path=self.data_s3_path,
            output_s3_path=self.output_s3_path,
            rft_lambda_arn=self.lambda_arn,
            overrides={},
        )

        with self.assertRaises(ValueError):
            builder.build(validation_config={"infra": False, "iam": False})

    def test_invalid_instance_count_raises_smhp(self):
        builder = RFTRecipeBuilder(
            job_name=self.job_name,
            platform=Platform.SMHP,
            model=Model.NOVA_LITE_2,
            method=TrainingMethod.RFT,
            instance_type="ml.p5.48xlarge",
            instance_count=999,
            data_s3_path=self.data_s3_path,
            output_s3_path=self.output_s3_path,
            rft_lambda_arn=self.lambda_arn,
            overrides={},
        )

        with self.assertRaises(ValueError):
            builder.build()

    def test_invalid_instance_type_raises_smtj(self):
        builder = RFTRecipeBuilder(
            job_name=self.job_name,
            platform=Platform.SMTJ,
            model=Model.NOVA_LITE_2,
            method=TrainingMethod.RFT,
            instance_type="ml.invalid.instance",
            instance_count=2,
            data_s3_path=self.data_s3_path,
            output_s3_path=self.output_s3_path,
            rft_lambda_arn=self.lambda_arn,
            overrides={},
        )

        with self.assertRaises(ValueError):
            builder.build()

    def test_invalid_instance_count_raises_smtj(self):
        builder = RFTRecipeBuilder(
            job_name=self.job_name,
            platform=Platform.SMTJ,
            model=Model.NOVA_LITE_2,
            method=TrainingMethod.RFT,
            instance_type="ml.p5.48xlarge",
            instance_count=999,
            data_s3_path=self.data_s3_path,
            output_s3_path=self.output_s3_path,
            rft_lambda_arn=self.lambda_arn,
            overrides={},
        )

        with self.assertRaises(ValueError):
            builder.build(validation_config={"infra": False, "iam": False})


if __name__ == "__main__":
    unittest.main()
