import os
import tempfile
import unittest

import yaml

import amzn_nova_customization_sdk.recipe_config.v_one.sft_config as v1_sft
import amzn_nova_customization_sdk.recipe_config.v_two.sft_config as v2_sft
from amzn_nova_customization_sdk.model.model_enums import (
    Model,
    Platform,
    TrainingMethod,
    Version,
)
from amzn_nova_customization_sdk.recipe_builder.sft_recipe_builder import (
    SFTRecipeBuilder,
)


class TestSFTRecipeBuilder(unittest.TestCase):
    def setUp(self):
        self.job_name = "test-job"
        self.data_s3_path = "s3://test-bucket/data"
        self.output_s3_path = "s3://test-bucket/output"

    def test_training_recipe_builder_initialization(self):
        builder = SFTRecipeBuilder(
            platform=Platform.SMTJ,
            model=Model.NOVA_LITE,
            method=TrainingMethod.SFT_LORA,
            instance_type="ml.p5.48xlarge",
            instance_count=4,
            job_name=self.job_name,
            data_s3_path=self.data_s3_path,
            output_s3_path=self.output_s3_path,
            overrides={},
        )

        self.assertEqual(builder.platform, Platform.SMTJ)
        self.assertEqual(builder.model, Model.NOVA_LITE)
        self.assertEqual(builder.method, TrainingMethod.SFT_LORA)
        self.assertEqual(builder.version, Version.ONE)
        self.assertEqual(builder.instance_type, "ml.p5.48xlarge")
        self.assertEqual(builder.instance_count, 4)
        self.assertEqual(builder.model_type, "amazon.nova-lite-v1:0:300k")
        self.assertEqual(builder.model_path, "nova-lite/prod")
        self.assertEqual(builder.overrides, {})

    def test_training_recipe_builder_initialization_v2(self):
        builder = SFTRecipeBuilder(
            platform=Platform.SMTJ,
            model=Model.NOVA_LITE_2,
            method=TrainingMethod.SFT_LORA,
            instance_type="ml.p5.48xlarge",
            instance_count=4,
            job_name=self.job_name,
            data_s3_path=self.data_s3_path,
            output_s3_path=self.output_s3_path,
            overrides={},
        )

        self.assertEqual(builder.platform, Platform.SMTJ)
        self.assertEqual(builder.model, Model.NOVA_LITE_2)
        self.assertEqual(builder.method, TrainingMethod.SFT_LORA)
        self.assertEqual(builder.version, Version.TWO)
        self.assertEqual(builder.instance_type, "ml.p5.48xlarge")
        self.assertEqual(builder.instance_count, 4)
        self.assertEqual(builder.model_type, "amazon.nova-2-lite-v1:0:256k")
        self.assertEqual(builder.model_path, "nova-lite-2/prod")
        self.assertEqual(builder.overrides, {})

    def test_create_training_run_config(self):
        builder = SFTRecipeBuilder(
            platform=Platform.SMTJ,
            model=Model.NOVA_LITE,
            method=TrainingMethod.SFT_LORA,
            instance_type="ml.p5.48xlarge",
            instance_count=4,
            job_name=self.job_name,
            data_s3_path=self.data_s3_path,
            output_s3_path=self.output_s3_path,
            overrides={},
        )

        run_config = builder._create_base_run_config()

        self.assertEqual(run_config.name, self.job_name)
        self.assertEqual(run_config.model_type, "amazon.nova-lite-v1:0:300k")
        self.assertEqual(run_config.data_s3_path, self.data_s3_path)
        self.assertEqual(run_config.output_s3_path, self.output_s3_path)
        self.assertEqual(run_config.replicas, 4)

    def test_generate_training_recipe_path_format(self):
        builder = SFTRecipeBuilder(
            platform=Platform.SMTJ,
            model=Model.NOVA_LITE,
            method=TrainingMethod.SFT_LORA,
            instance_type="ml.g5.12xlarge",
            instance_count=1,
            job_name=self.job_name,
            data_s3_path=self.data_s3_path,
            output_s3_path=self.output_s3_path,
            overrides={},
        )

        # Set generated_recipe_dir to test the functionality
        builder.generated_recipe_dir = "generated-recipes"

        recipe_path = builder._generate_recipe_path(None)
        path = recipe_path.path

        self.assertTrue(path.endswith(".yaml"))
        self.assertTrue(path.startswith(f"generated-recipes/{builder.job_name}"))

    def test_create_sft_v1_lora_config(self):
        builder = SFTRecipeBuilder(
            platform=Platform.SMTJ,
            model=Model.NOVA_LITE,
            method=TrainingMethod.SFT_LORA,
            instance_type="ml.g5.12xlarge",
            instance_count=1,
            job_name=self.job_name,
            data_s3_path=self.data_s3_path,
            output_s3_path=self.output_s3_path,
            overrides={},
        )

        recipe_config = builder._build_recipe_config()

        self.assertIsInstance(recipe_config, v1_sft.SFTRecipeConfig)
        self.assertEqual(recipe_config.training_config.max_length, 8192)
        self.assertEqual(recipe_config.training_config.global_batch_size, 64)
        self.assertIsNotNone(recipe_config.training_config.model.peft)
        self.assertEqual(
            recipe_config.training_config.model.peft.peft_scheme, v1_sft.PeftScheme.LORA
        )

    def test_create_sft_v1_fullrank_config(self):
        builder = SFTRecipeBuilder(
            platform=Platform.SMTJ,
            model=Model.NOVA_LITE,
            method=TrainingMethod.SFT_FULLRANK,
            instance_type="ml.p5.48xlarge",
            instance_count=4,
            job_name=self.job_name,
            data_s3_path=self.data_s3_path,
            output_s3_path=self.output_s3_path,
            overrides={},
        )

        recipe_config = builder._build_recipe_config()

        self.assertIsInstance(recipe_config, v1_sft.SFTRecipeConfig)
        self.assertEqual(
            recipe_config.training_config.model.peft.peft_scheme, v1_sft.PeftScheme.NULL
        )
        self.assertEqual(recipe_config.training_config.model.optim.weight_decay, 0.0)

    def test_create_sft_v2_lora_config(self):
        builder = SFTRecipeBuilder(
            platform=Platform.SMTJ,
            model=Model.NOVA_LITE_2,
            method=TrainingMethod.SFT_LORA,
            instance_type="ml.p5.48xlarge",
            instance_count=4,
            job_name=self.job_name,
            data_s3_path=self.data_s3_path,
            output_s3_path=self.output_s3_path,
            overrides={},
        )

        recipe_config = builder._build_recipe_config()

        self.assertIsInstance(recipe_config, v2_sft.SFTRecipeConfig)
        self.assertEqual(recipe_config.training_config.max_steps, 100)
        self.assertEqual(recipe_config.training_config.save_steps, 100)
        self.assertEqual(recipe_config.training_config.save_top_k, 5)
        self.assertEqual(recipe_config.training_config.max_length, 32768)
        self.assertEqual(recipe_config.training_config.global_batch_size, 32)
        self.assertEqual(recipe_config.training_config.reasoning_enabled, True)
        self.assertEqual(recipe_config.training_config.lr_scheduler.warmup_steps, 10)
        self.assertEqual(recipe_config.training_config.lr_scheduler.min_lr, 1e-6)
        self.assertEqual(recipe_config.training_config.optim_config.lr, 1e-5)
        self.assertEqual(recipe_config.training_config.optim_config.weight_decay, 0.0)
        self.assertEqual(recipe_config.training_config.optim_config.adam_beta1, 0.9)
        self.assertEqual(recipe_config.training_config.optim_config.adam_beta2, 0.95)
        self.assertIsNotNone(recipe_config.training_config.peft)
        self.assertEqual(
            recipe_config.training_config.peft.peft_scheme, v2_sft.PeftScheme.LORA
        )
        self.assertIsNotNone(recipe_config.training_config.peft.lora_tuning)
        self.assertEqual(recipe_config.training_config.peft.lora_tuning.alpha, 64)
        self.assertEqual(
            recipe_config.training_config.peft.lora_tuning.lora_plus_lr_ratio, 64.0
        )

    def test_create_sft_v2_fullrank_config(self):
        builder = SFTRecipeBuilder(
            platform=Platform.SMTJ,
            model=Model.NOVA_LITE_2,
            method=TrainingMethod.SFT_FULLRANK,
            instance_type="ml.p5.48xlarge",
            instance_count=4,
            job_name=self.job_name,
            data_s3_path=self.data_s3_path,
            output_s3_path=self.output_s3_path,
            overrides={},
        )

        recipe_config = builder._build_recipe_config()

        self.assertIsInstance(recipe_config, v2_sft.SFTRecipeConfig)
        self.assertEqual(recipe_config.training_config.max_steps, 100)
        self.assertEqual(recipe_config.training_config.save_steps, 100)
        self.assertEqual(recipe_config.training_config.save_top_k, 5)
        self.assertEqual(recipe_config.training_config.max_length, 32768)
        self.assertEqual(recipe_config.training_config.global_batch_size, 32)
        self.assertEqual(recipe_config.training_config.reasoning_enabled, True)
        self.assertEqual(recipe_config.training_config.lr_scheduler.warmup_steps, 10)
        self.assertEqual(recipe_config.training_config.lr_scheduler.min_lr, 1e-6)
        self.assertEqual(recipe_config.training_config.optim_config.lr, 5e-6)
        self.assertEqual(recipe_config.training_config.optim_config.weight_decay, 0.0)
        self.assertEqual(recipe_config.training_config.optim_config.adam_beta1, 0.9)
        self.assertEqual(recipe_config.training_config.optim_config.adam_beta2, 0.95)
        self.assertIsNotNone(recipe_config.training_config.peft)
        self.assertEqual(
            recipe_config.training_config.peft.peft_scheme, v2_sft.PeftScheme.NULL
        )
        self.assertIsNone(recipe_config.training_config.peft.lora_tuning)

    def test_build_sft_v1_lora_yaml(self):
        builder = SFTRecipeBuilder(
            platform=Platform.SMTJ,
            model=Model.NOVA_LITE,
            method=TrainingMethod.SFT_LORA,
            instance_type="ml.g5.12xlarge",
            instance_count=1,
            job_name=self.job_name,
            data_s3_path=self.data_s3_path,
            output_s3_path=self.output_s3_path,
            overrides={},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test_recipe.yaml")
            result_path = builder.build(
                file_path, validation_config={"infra": False, "iam": False}
            )

            self.assertEqual(result_path.path, file_path)
            self.assertTrue(os.path.exists(file_path))

            with open(file_path, "r") as f:
                config = yaml.safe_load(f)

            self.assertIn("run", config)
            self.assertIn("training_config", config)
            self.assertEqual(config["run"]["name"], self.job_name)
            self.assertEqual(config["run"]["replicas"], 1)
            self.assertEqual(config["training_config"]["max_length"], 8192)

    def test_build_sft_v1_fullrank_yaml(self):
        builder = SFTRecipeBuilder(
            platform=Platform.SMTJ,
            model=Model.NOVA_LITE,
            method=TrainingMethod.SFT_FULLRANK,
            instance_type="ml.p5.48xlarge",
            instance_count=4,
            job_name=self.job_name,
            data_s3_path=self.data_s3_path,
            output_s3_path=self.output_s3_path,
            overrides={},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test_recipe.yaml")
            builder.build(file_path, validation_config={"infra": False, "iam": False})

            self.assertTrue(os.path.exists(file_path))

            with open(file_path, "r") as f:
                config = yaml.safe_load(f)

            self.assertIn("run", config)
            self.assertEqual(config["run"]["replicas"], 4)
            self.assertEqual(config["training_config"]["max_length"], 32768)
            self.assertEqual(
                config["training_config"]["model"]["peft"]["peft_scheme"],
                v1_sft.PeftScheme.NULL.value,
            )

    def test_build_sft_v2_lora_yaml(self):
        builder = SFTRecipeBuilder(
            platform=Platform.SMTJ,
            model=Model.NOVA_LITE_2,
            method=TrainingMethod.SFT_LORA,
            instance_type="ml.p5.48xlarge",
            instance_count=4,
            job_name=self.job_name,
            data_s3_path=self.data_s3_path,
            output_s3_path=self.output_s3_path,
            overrides={},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test_recipe.yaml")
            result_path = builder.build(
                file_path, validation_config={"infra": False, "iam": False}
            )

            self.assertEqual(result_path.path, file_path)
            self.assertTrue(os.path.exists(file_path))

            with open(file_path, "r") as f:
                config = yaml.safe_load(f)

            self.assertIn("run", config)
            self.assertIn("training_config", config)
            self.assertEqual(config["run"]["name"], self.job_name)
            self.assertEqual(
                config["run"]["model_type"], "amazon.nova-2-lite-v1:0:256k"
            )
            self.assertEqual(config["run"]["model_name_or_path"], "nova-lite-2/prod")
            self.assertEqual(config["run"]["replicas"], 4)
            self.assertEqual(config["run"]["data_s3_path"], self.data_s3_path)
            self.assertEqual(config["run"]["output_s3_path"], self.output_s3_path)
            self.assertEqual(config["training_config"]["max_steps"], 100)
            self.assertEqual(config["training_config"]["save_steps"], 100)
            self.assertEqual(config["training_config"]["save_top_k"], 5)
            self.assertEqual(config["training_config"]["max_length"], 32768)
            self.assertEqual(config["training_config"]["global_batch_size"], 32)
            self.assertEqual(config["training_config"]["reasoning_enabled"], True)
            self.assertEqual(
                config["training_config"]["lr_scheduler"]["warmup_steps"], 10
            )
            self.assertEqual(config["training_config"]["lr_scheduler"]["min_lr"], 1e-6)
            self.assertEqual(config["training_config"]["optim_config"]["lr"], 1e-5)
            self.assertEqual(
                config["training_config"]["optim_config"]["weight_decay"], 0.0
            )
            self.assertEqual(
                config["training_config"]["optim_config"]["adam_beta1"], 0.9
            )
            self.assertEqual(
                config["training_config"]["optim_config"]["adam_beta2"], 0.95
            )
            self.assertEqual(
                config["training_config"]["peft"]["peft_scheme"],
                v2_sft.PeftScheme.LORA.value,
            )
            self.assertEqual(
                config["training_config"]["peft"]["lora_tuning"]["alpha"], 64
            )
            self.assertEqual(
                config["training_config"]["peft"]["lora_tuning"]["lora_plus_lr_ratio"],
                64.0,
            )

    def test_build_sft_v2_fullrank_yaml(self):
        builder = SFTRecipeBuilder(
            platform=Platform.SMTJ,
            model=Model.NOVA_LITE_2,
            method=TrainingMethod.SFT_FULLRANK,
            instance_type="ml.p5.48xlarge",
            instance_count=4,
            job_name=self.job_name,
            data_s3_path=self.data_s3_path,
            output_s3_path=self.output_s3_path,
            overrides={},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test_recipe.yaml")
            builder.build(file_path, validation_config={"infra": False, "iam": False})

            self.assertTrue(os.path.exists(file_path))

            with open(file_path, "r") as f:
                config = yaml.safe_load(f)

            self.assertIn("run", config)
            self.assertIn("training_config", config)
            self.assertEqual(config["run"]["name"], self.job_name)
            self.assertEqual(
                config["run"]["model_type"], "amazon.nova-2-lite-v1:0:256k"
            )
            self.assertEqual(config["run"]["model_name_or_path"], "nova-lite-2/prod")
            self.assertEqual(config["run"]["replicas"], 4)
            self.assertEqual(config["run"]["data_s3_path"], self.data_s3_path)
            self.assertEqual(config["run"]["output_s3_path"], self.output_s3_path)
            self.assertEqual(config["training_config"]["max_steps"], 100)
            self.assertEqual(config["training_config"]["save_steps"], 100)
            self.assertEqual(config["training_config"]["save_top_k"], 5)
            self.assertEqual(config["training_config"]["max_length"], 32768)
            self.assertEqual(config["training_config"]["global_batch_size"], 32)
            self.assertEqual(config["training_config"]["reasoning_enabled"], True)
            self.assertEqual(
                config["training_config"]["lr_scheduler"]["warmup_steps"], 10
            )
            self.assertEqual(config["training_config"]["lr_scheduler"]["min_lr"], 1e-6)
            self.assertEqual(config["training_config"]["optim_config"]["lr"], 5e-6)
            self.assertEqual(
                config["training_config"]["optim_config"]["weight_decay"], 0.0
            )
            self.assertEqual(
                config["training_config"]["optim_config"]["adam_beta1"], 0.9
            )
            self.assertEqual(
                config["training_config"]["optim_config"]["adam_beta2"], 0.95
            )
            self.assertEqual(
                config["training_config"]["peft"]["peft_scheme"],
                v2_sft.PeftScheme.NULL.value,
            )
            self.assertNotIn("lora_tuning", config["training_config"]["peft"])

    def test_sft_v1_recipe_does_not_have_top_k(self):
        builder = SFTRecipeBuilder(
            platform=Platform.SMTJ,
            model=Model.NOVA_LITE,
            method=TrainingMethod.SFT_LORA,
            instance_type="ml.g5.12xlarge",
            instance_count=1,
            job_name=self.job_name,
            data_s3_path=self.data_s3_path,
            output_s3_path=self.output_s3_path,
            overrides={},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test_recipe.yaml")
            builder.build(file_path, validation_config={"infra": False, "iam": False})

            with open(file_path, "r") as f:
                config = yaml.safe_load(f)

            # SFT recipes should not have top_k anywhere
            self.assertNotIn("top_k", str(config))

    def test_sft_v2_with_overrides(self):
        overrides = {
            "name": "override",
            "max_steps": 200,
            "reasoning_enabled": False,
            "alpha": 128,
            "lora_plus_lr_ratio": 99.0,
            "lr": 2e-5,
        }

        builder = SFTRecipeBuilder(
            platform=Platform.SMTJ,
            model=Model.NOVA_LITE_2,
            method=TrainingMethod.SFT_LORA,
            instance_type="ml.p5.48xlarge",
            instance_count=4,
            job_name=self.job_name,
            data_s3_path=self.data_s3_path,
            output_s3_path=self.output_s3_path,
            overrides=overrides,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test_recipe.yaml")
            builder.build(file_path, validation_config={"infra": False, "iam": False})

            with open(file_path, "r") as f:
                config = yaml.safe_load(f)

            self.assertEqual(config["run"]["name"], "override")
            self.assertEqual(config["training_config"]["max_steps"], 200)
            self.assertEqual(config["training_config"]["reasoning_enabled"], False)
            self.assertEqual(
                config["training_config"]["peft"]["lora_tuning"]["alpha"], 128
            )
            self.assertEqual(
                config["training_config"]["peft"]["lora_tuning"]["lora_plus_lr_ratio"],
                99.0,
            )
            self.assertEqual(config["training_config"]["optim_config"]["lr"], 2e-5)

    def test_different_models_have_correct_max_lengths(self):
        # Nova Micro
        builder_micro = SFTRecipeBuilder(
            platform=Platform.SMTJ,
            model=Model.NOVA_MICRO,
            method=TrainingMethod.SFT_LORA,
            instance_type="ml.g5.12xlarge",
            instance_count=1,
            job_name=self.job_name,
            data_s3_path=self.data_s3_path,
            output_s3_path=self.output_s3_path,
            overrides={},
        )

        config_micro: v1_sft.SFTRecipeConfig = builder_micro._build_recipe_config()
        self.assertEqual(config_micro.training_config.max_length, 8192)

        # Nova Lite
        builder_lite = SFTRecipeBuilder(
            platform=Platform.SMTJ,
            model=Model.NOVA_LITE,
            method=TrainingMethod.SFT_LORA,
            instance_type="ml.g5.12xlarge",
            instance_count=1,
            job_name=self.job_name,
            data_s3_path=self.data_s3_path,
            output_s3_path=self.output_s3_path,
            overrides={},
        )

        config_lite: v1_sft.SFTRecipeConfig = builder_lite._build_recipe_config()
        self.assertEqual(config_lite.training_config.max_length, 8192)

        # Nova Lite 2
        builder_lite_2 = SFTRecipeBuilder(
            platform=Platform.SMTJ,
            model=Model.NOVA_LITE_2,
            method=TrainingMethod.SFT_LORA,
            instance_type="ml.p5.48xlarge",
            instance_count=4,
            job_name=self.job_name,
            data_s3_path=self.data_s3_path,
            output_s3_path=self.output_s3_path,
            overrides={},
        )

        config_lite_2: v2_sft.SFTRecipeConfig = builder_lite_2._build_recipe_config()
        self.assertEqual(config_lite_2.training_config.max_length, 32768)

        # Nova Pro
        builder_pro = SFTRecipeBuilder(
            platform=Platform.SMTJ,
            model=Model.NOVA_PRO,
            method=TrainingMethod.SFT_LORA,
            instance_type="ml.p5.48xlarge",
            instance_count=6,
            job_name=self.job_name,
            data_s3_path=self.data_s3_path,
            output_s3_path=self.output_s3_path,
            overrides={},
        )
        config_pro: v1_sft.SFTRecipeConfig = builder_pro._build_recipe_config()
        self.assertEqual(config_pro.training_config.max_length, 32768)

    def test_build_with_invalid_instance_type(self):
        builder = SFTRecipeBuilder(
            platform=Platform.SMTJ,
            model=Model.NOVA_LITE,
            method=TrainingMethod.SFT_LORA,
            instance_type="ml.invalid.instance",
            instance_count=4,
            job_name=self.job_name,
            data_s3_path=self.data_s3_path,
            output_s3_path=self.output_s3_path,
            overrides={},
        )

        with self.assertRaises(ValueError) as context:
            builder.build(validation_config={"infra": False, "iam": False})

        self.assertIn("not supported", str(context.exception))

    def test_build_with_invalid_instance_count(self):
        builder = SFTRecipeBuilder(
            platform=Platform.SMTJ,
            model=Model.NOVA_LITE,
            method=TrainingMethod.SFT_LORA,
            instance_type="ml.g5.12xlarge",
            instance_count=1000,
            job_name=self.job_name,
            data_s3_path=self.data_s3_path,
            output_s3_path=self.output_s3_path,
            overrides={},
        )

        with self.assertRaises(ValueError) as context:
            builder.build(validation_config={"infra": False, "iam": False})

        self.assertIn("not supported", str(context.exception))


if __name__ == "__main__":
    unittest.main()
