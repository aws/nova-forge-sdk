import unittest

from amzn_nova_customization_sdk.recipe_config.base_recipe_config import BaseRunConfig
from amzn_nova_customization_sdk.recipe_config.v_two.sft_config import (
    LoraTuningConfig,
    LrScheduler,
    OptimConfig,
    Peft,
    PeftScheme,
    SFTRecipeConfig,
    SFTTrainingConfig,
)


class TestLrScheduler(unittest.TestCase):
    def test_lr_scheduler_creation(self):
        scheduler = LrScheduler(warmup_steps=100, min_lr=1e-7)
        self.assertEqual(scheduler.warmup_steps, 100)
        self.assertEqual(scheduler.min_lr, 1e-7)

    def test_lr_scheduler_custom_values(self):
        scheduler = LrScheduler(warmup_steps=200, min_lr=5e-6)
        self.assertEqual(scheduler.warmup_steps, 200)
        self.assertEqual(scheduler.min_lr, 5e-6)

    def test_lr_scheduler_defaults(self):
        scheduler = LrScheduler()
        self.assertEqual(scheduler.warmup_steps, 10)
        self.assertEqual(scheduler.min_lr, 1e-6)


class TestOptimConfig(unittest.TestCase):
    def test_optim_config_creation(self):
        config = OptimConfig(lr=1e-5, weight_decay=0.0, adam_beta1=0.9, adam_beta2=0.95)
        self.assertEqual(config.lr, 1e-5)
        self.assertEqual(config.weight_decay, 0.0)
        self.assertEqual(config.adam_beta1, 0.9)
        self.assertEqual(config.adam_beta2, 0.95)

    def test_optim_config_defaults(self):
        config = OptimConfig(lr=1e-5)
        self.assertEqual(config.lr, 1e-5)
        self.assertEqual(config.weight_decay, 0.0)
        self.assertEqual(config.adam_beta1, 0.9)
        self.assertEqual(config.adam_beta2, 0.95)

    def test_optim_config_to_dict(self):
        config = OptimConfig(lr=1e-5, weight_decay=0.1)
        config_dict = config.to_dict()
        self.assertEqual(config_dict["lr"], 1e-5)
        self.assertEqual(config_dict["weight_decay"], 0.1)
        self.assertEqual(config_dict["adam_beta1"], 0.9)
        self.assertEqual(config_dict["adam_beta2"], 0.95)


class TestLoraTuningConfig(unittest.TestCase):
    def test_lora_tuning_config_defaults(self):
        config = LoraTuningConfig()
        self.assertEqual(config.alpha, 64)
        self.assertEqual(config.lora_plus_lr_ratio, 64.0)

    def test_lora_tuning_config_custom_values(self):
        config = LoraTuningConfig(alpha=32, lora_plus_lr_ratio=16.0)
        self.assertEqual(config.alpha, 32)
        self.assertEqual(config.lora_plus_lr_ratio, 16.0)

    def test_lora_tuning_config_to_dict(self):
        config = LoraTuningConfig(alpha=32, lora_plus_lr_ratio=16.0)
        config_dict = config.to_dict()
        self.assertEqual(config_dict["alpha"], 32)
        self.assertEqual(config_dict["lora_plus_lr_ratio"], 16.0)


class TestPeft(unittest.TestCase):
    def test_peft_lora_with_config(self):
        lora_tuning = LoraTuningConfig(alpha=32, lora_plus_lr_ratio=16.0)
        peft = Peft(peft_scheme=PeftScheme.LORA, lora_tuning=lora_tuning)
        self.assertEqual(peft.peft_scheme, PeftScheme.LORA)
        self.assertIsInstance(peft.lora_tuning, LoraTuningConfig)
        self.assertEqual(peft.lora_tuning.alpha, 32)
        self.assertEqual(peft.lora_tuning.lora_plus_lr_ratio, 16.0)

    def test_peft_lora_without_config_creates_default(self):
        peft = Peft(peft_scheme=PeftScheme.LORA)
        self.assertEqual(peft.peft_scheme, PeftScheme.LORA)
        self.assertIsInstance(peft.lora_tuning, LoraTuningConfig)
        self.assertEqual(peft.lora_tuning.alpha, 64)
        self.assertEqual(peft.lora_tuning.lora_plus_lr_ratio, 64.0)

    def test_peft_null_scheme(self):
        peft = Peft(peft_scheme=PeftScheme.NULL)
        self.assertEqual(peft.peft_scheme, PeftScheme.NULL)
        self.assertIsNone(peft.lora_tuning)

    def test_peft_none_scheme_converts_to_null(self):
        peft = Peft(peft_scheme=None)
        self.assertEqual(peft.peft_scheme, PeftScheme.NULL)
        self.assertIsNone(peft.lora_tuning)

    def test_peft_to_dict_with_lora(self):
        lora_tuning = LoraTuningConfig(alpha=32, lora_plus_lr_ratio=16.0)
        peft = Peft(peft_scheme=PeftScheme.LORA, lora_tuning=lora_tuning)
        peft_dict = peft.to_dict()
        self.assertEqual(peft_dict["peft_scheme"], "lora")
        self.assertIn("lora_tuning", peft_dict)
        self.assertEqual(peft_dict["lora_tuning"]["alpha"], 32)
        self.assertEqual(peft_dict["lora_tuning"]["lora_plus_lr_ratio"], 16.0)

    def test_peft_to_dict_null_scheme(self):
        peft = Peft(peft_scheme=PeftScheme.NULL)
        peft_dict = peft.to_dict()
        self.assertEqual(peft_dict["peft_scheme"], "null")
        self.assertNotIn("lora_tuning", peft_dict)


class TestSFTTrainingConfig(unittest.TestCase):
    def _create_default_lr_scheduler(self):
        return LrScheduler(warmup_steps=10, min_lr=1e-6)

    def _create_default_optim_config(self):
        return OptimConfig(lr=1e-5, weight_decay=0.0, adam_beta1=0.9, adam_beta2=0.95)

    def _create_default_peft_config(self):
        lora_tuning = LoraTuningConfig(alpha=32, lora_plus_lr_ratio=64.0)
        return Peft(peft_scheme=PeftScheme.LORA, lora_tuning=lora_tuning)

    def test_sft_training_config_creation(self):
        config = SFTTrainingConfig(
            max_steps=100,
            max_length=8192,
            global_batch_size=32,
            reasoning_enabled=True,
            lr_scheduler=self._create_default_lr_scheduler(),
            optim_config=self._create_default_optim_config(),
            peft=self._create_default_peft_config(),
        )
        self.assertEqual(config.max_steps, 100)
        self.assertEqual(config.max_length, 8192)
        self.assertEqual(config.global_batch_size, 32)
        self.assertEqual(config.reasoning_enabled, True)
        self.assertEqual(config.save_steps, 100)
        self.assertEqual(config.save_top_k, 5)
        self.assertIsInstance(config.lr_scheduler, LrScheduler)
        self.assertEqual(config.lr_scheduler.warmup_steps, 10)
        self.assertEqual(config.lr_scheduler.min_lr, 1e-6)
        self.assertIsInstance(config.optim_config, OptimConfig)
        self.assertEqual(config.optim_config.lr, 1e-5)
        self.assertEqual(config.optim_config.weight_decay, 0.0)
        self.assertEqual(config.optim_config.adam_beta1, 0.9)
        self.assertEqual(config.optim_config.adam_beta2, 0.95)
        self.assertIsInstance(config.peft, Peft)
        self.assertEqual(config.peft.peft_scheme, PeftScheme.LORA)
        self.assertIsInstance(config.peft.lora_tuning, LoraTuningConfig)
        self.assertEqual(config.peft.lora_tuning.alpha, 32)
        self.assertEqual(config.peft.lora_tuning.lora_plus_lr_ratio, 64.0)

    def test_sft_training_config_custom_values(self):
        config = SFTTrainingConfig(
            max_steps=2000,
            save_steps=200,
            save_top_k=10,
            max_length=16384,
            global_batch_size=64,
            reasoning_enabled=False,
            lr_scheduler=self._create_default_lr_scheduler(),
            optim_config=self._create_default_optim_config(),
            peft=self._create_default_peft_config(),
        )
        self.assertEqual(config.max_steps, 2000)
        self.assertEqual(config.save_steps, 200)
        self.assertEqual(config.save_top_k, 10)
        self.assertEqual(config.max_length, 16384)
        self.assertEqual(config.global_batch_size, 64)
        self.assertEqual(config.reasoning_enabled, False)

    def test_sft_training_config_to_dict(self):
        config = SFTTrainingConfig(
            max_steps=100,
            max_length=8192,
            global_batch_size=32,
            reasoning_enabled=True,
            lr_scheduler=self._create_default_lr_scheduler(),
            optim_config=self._create_default_optim_config(),
            peft=self._create_default_peft_config(),
        )
        config_dict = config.to_dict()

        self.assertIn("max_steps", config_dict)
        self.assertIn("save_steps", config_dict)
        self.assertIn("save_top_k", config_dict)
        self.assertIn("max_length", config_dict)
        self.assertIn("global_batch_size", config_dict)
        self.assertIn("reasoning_enabled", config_dict)
        self.assertIn("lr_scheduler", config_dict)
        self.assertIn("optim_config", config_dict)
        self.assertIn("peft", config_dict)

        self.assertEqual(config_dict["max_steps"], 100)
        self.assertEqual(config_dict["save_steps"], 100)
        self.assertEqual(config_dict["save_top_k"], 5)
        self.assertEqual(config_dict["max_length"], 8192)
        self.assertEqual(config_dict["global_batch_size"], 32)
        self.assertEqual(config_dict["reasoning_enabled"], True)

        self.assertIsInstance(config_dict["lr_scheduler"], dict)
        self.assertEqual(config_dict["lr_scheduler"]["warmup_steps"], 10)
        self.assertEqual(config_dict["lr_scheduler"]["min_lr"], 1e-6)

        self.assertIsInstance(config_dict["optim_config"], dict)
        self.assertEqual(config_dict["optim_config"]["lr"], 1e-5)
        self.assertEqual(config_dict["optim_config"]["weight_decay"], 0.0)
        self.assertEqual(config_dict["optim_config"]["adam_beta1"], 0.9)
        self.assertEqual(config_dict["optim_config"]["adam_beta2"], 0.95)

        self.assertIsInstance(config_dict["peft"], dict)
        self.assertEqual(config_dict["peft"]["peft_scheme"], "lora")
        self.assertIn("lora_tuning", config_dict["peft"])
        self.assertEqual(config_dict["peft"]["lora_tuning"]["alpha"], 32)
        self.assertEqual(config_dict["peft"]["lora_tuning"]["lora_plus_lr_ratio"], 64.0)


class TestSFTRecipeConfig(unittest.TestCase):
    def _create_default_lr_scheduler(self):
        return LrScheduler(warmup_steps=10, min_lr=1e-6)

    def _create_default_optim_config(self):
        return OptimConfig(lr=1e-5, weight_decay=0.0, adam_beta1=0.9, adam_beta2=0.95)

    def _create_default_peft_config(self):
        lora_tuning = LoraTuningConfig(alpha=32, lora_plus_lr_ratio=64.0)
        return Peft(peft_scheme=PeftScheme.LORA, lora_tuning=lora_tuning)

    def _create_default_training_config(self):
        return SFTTrainingConfig(
            max_steps=100,
            max_length=8192,
            global_batch_size=32,
            reasoning_enabled=True,
            lr_scheduler=self._create_default_lr_scheduler(),
            optim_config=self._create_default_optim_config(),
            peft=self._create_default_peft_config(),
        )

    def test_sft_recipe_config_creation(self):
        run_config = BaseRunConfig(
            name="test-sft-v2-job",
            model_type="amazon.nova-2-lite-v1:0:256k",
            model_name_or_path="nova-lite-2/prod",
            data_s3_path="s3://bucket/data",
            output_s3_path="s3://bucket/output",
            replicas=4,
        )
        recipe = SFTRecipeConfig(
            run=run_config, training_config=self._create_default_training_config()
        )

        self.assertEqual(recipe.run.name, "test-sft-v2-job")
        self.assertIsInstance(recipe.training_config, SFTTrainingConfig)
        self.assertEqual(recipe.training_config.max_length, 8192)
        self.assertEqual(recipe.training_config.max_steps, 100)
        self.assertEqual(recipe.training_config.reasoning_enabled, True)

    def test_sft_recipe_config_to_dict(self):
        run_config = BaseRunConfig(
            name="test-sft-v2-job",
            model_type="amazon.nova-2-lite-v1:0:256k",
            model_name_or_path="nova-lite-2/prod",
            data_s3_path="s3://bucket/data",
            output_s3_path="s3://bucket/output",
            replicas=4,
        )
        recipe = SFTRecipeConfig(
            run=run_config, training_config=self._create_default_training_config()
        )
        recipe_dict = recipe.to_dict()

        self.assertIn("run", recipe_dict)
        self.assertIn("training_config", recipe_dict)
        self.assertEqual(recipe_dict["run"]["name"], "test-sft-v2-job")
        self.assertEqual(
            recipe_dict["run"]["model_type"], "amazon.nova-2-lite-v1:0:256k"
        )
        self.assertEqual(recipe_dict["run"]["model_name_or_path"], "nova-lite-2/prod")
        self.assertEqual(recipe_dict["run"]["replicas"], 4)
        self.assertEqual(recipe_dict["run"]["data_s3_path"], "s3://bucket/data")
        self.assertEqual(recipe_dict["run"]["output_s3_path"], "s3://bucket/output")

        self.assertEqual(recipe_dict["training_config"]["max_steps"], 100)
        self.assertEqual(recipe_dict["training_config"]["save_steps"], 100)
        self.assertEqual(recipe_dict["training_config"]["save_top_k"], 5)
        self.assertEqual(recipe_dict["training_config"]["max_length"], 8192)
        self.assertEqual(recipe_dict["training_config"]["global_batch_size"], 32)
        self.assertEqual(recipe_dict["training_config"]["reasoning_enabled"], True)

        self.assertIsInstance(recipe_dict["training_config"]["lr_scheduler"], dict)
        self.assertEqual(
            recipe_dict["training_config"]["lr_scheduler"]["warmup_steps"], 10
        )
        self.assertEqual(recipe_dict["training_config"]["lr_scheduler"]["min_lr"], 1e-6)

        self.assertIsInstance(recipe_dict["training_config"]["optim_config"], dict)
        self.assertEqual(recipe_dict["training_config"]["optim_config"]["lr"], 1e-5)
        self.assertEqual(
            recipe_dict["training_config"]["optim_config"]["weight_decay"], 0.0
        )
        self.assertEqual(
            recipe_dict["training_config"]["optim_config"]["adam_beta1"], 0.9
        )
        self.assertEqual(
            recipe_dict["training_config"]["optim_config"]["adam_beta2"], 0.95
        )

        self.assertIsInstance(recipe_dict["training_config"]["peft"], dict)
        self.assertEqual(recipe_dict["training_config"]["peft"]["peft_scheme"], "lora")
        self.assertIn("lora_tuning", recipe_dict["training_config"]["peft"])
        self.assertEqual(
            recipe_dict["training_config"]["peft"]["lora_tuning"]["alpha"], 32
        )
        self.assertEqual(
            recipe_dict["training_config"]["peft"]["lora_tuning"]["lora_plus_lr_ratio"],
            64.0,
        )


if __name__ == "__main__":
    unittest.main()
