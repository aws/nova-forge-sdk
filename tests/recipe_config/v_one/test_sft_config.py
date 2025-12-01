import unittest

from amzn_nova_customization_sdk.recipe_config.base_recipe_config import BaseRunConfig
from amzn_nova_customization_sdk.recipe_config.v_one.sft_config import (
    LoraTuningConfig,
    ModelConfig,
    Name,
    OptimConfig,
    PeftConfig,
    PeftScheme,
    SchedConfig,
    SFTRecipeConfig,
    SFTTrainingConfig,
    TrainerConfig,
)


class TestSFTTrainingConfig(unittest.TestCase):
    def _create_default_trainer_config(self):
        return TrainerConfig(max_epochs=2)

    def _create_default_model_config(self):
        sched_config = SchedConfig(warmup_steps=10, constant_steps=0, min_lr=1e-6)
        optim_config = OptimConfig(
            lr=1e-5,
            name=Name.DISTRIBUTED_FUSED_ADAM,
            adam_w_mode=True,
            eps=1e-6,
            weight_decay=0.0,
            betas=[0.9, 0.999],
            sched=sched_config,
        )
        lora_tuning = LoraTuningConfig(
            loraplus_lr_ratio=8.0, alpha=32, adapter_dropout=0.01
        )
        return ModelConfig(
            hidden_dropout=0.0,
            attention_dropout=0.0,
            ffn_dropout=0.0,
            optim=optim_config,
            peft=PeftConfig(peft_scheme=PeftScheme.LORA, lora_tuning=lora_tuning),
        )

    def test_sft_training_config_creation(self):
        config = SFTTrainingConfig(
            max_length=8192,
            global_batch_size=32,
            trainer=self._create_default_trainer_config(),
            model=self._create_default_model_config(),
        )
        self.assertEqual(config.max_length, 8192)
        self.assertEqual(config.global_batch_size, 32)
        self.assertIsInstance(config.trainer, TrainerConfig)
        self.assertIsInstance(config.model, ModelConfig)

    def test_sft_training_config_custom_values(self):
        config = SFTTrainingConfig(
            max_length=16384,
            global_batch_size=64,
            trainer=self._create_default_trainer_config(),
            model=self._create_default_model_config(),
        )
        self.assertEqual(config.max_length, 16384)
        self.assertEqual(config.global_batch_size, 64)

    def test_sft_training_config_to_dict(self):
        config = SFTTrainingConfig(
            max_length=8192,
            global_batch_size=32,
            trainer=self._create_default_trainer_config(),
            model=self._create_default_model_config(),
        )
        config_dict = config.to_dict()

        self.assertIn("max_length", config_dict)
        self.assertIn("global_batch_size", config_dict)
        self.assertIn("trainer", config_dict)
        self.assertIn("model", config_dict)
        self.assertEqual(config_dict["max_length"], 8192)
        self.assertEqual(config_dict["global_batch_size"], 32)
        self.assertIsInstance(config_dict["trainer"], dict)
        self.assertEqual(config_dict["trainer"]["max_epochs"], 2)
        self.assertIsInstance(config_dict["model"], dict)
        self.assertEqual(config_dict["model"]["hidden_dropout"], 0.0)
        self.assertEqual(config_dict["model"]["attention_dropout"], 0.0)
        self.assertEqual(config_dict["model"]["ffn_dropout"], 0.0)
        self.assertIsInstance(config_dict["model"]["optim"], dict)
        self.assertEqual(config_dict["model"]["optim"]["lr"], 1e-5)
        self.assertEqual(
            config_dict["model"]["optim"]["name"], "distributed_fused_adam"
        )
        self.assertEqual(config_dict["model"]["optim"]["adam_w_mode"], True)
        self.assertEqual(config_dict["model"]["optim"]["eps"], 1e-6)
        self.assertEqual(config_dict["model"]["optim"]["weight_decay"], 0.0)
        self.assertEqual(config_dict["model"]["optim"]["betas"], [0.9, 0.999])
        self.assertIsInstance(config_dict["model"]["optim"]["sched"], dict)
        self.assertEqual(config_dict["model"]["optim"]["sched"]["warmup_steps"], 10)
        self.assertEqual(config_dict["model"]["optim"]["sched"]["constant_steps"], 0)
        self.assertEqual(config_dict["model"]["optim"]["sched"]["min_lr"], 1e-6)


class TestSFTRecipeConfig(unittest.TestCase):
    def _create_default_trainer_config(self):
        return TrainerConfig(max_epochs=2)

    def _create_default_model_config(self):
        sched_config = SchedConfig(warmup_steps=10, constant_steps=0, min_lr=1e-6)
        optim_config = OptimConfig(
            lr=1e-5,
            name=Name.DISTRIBUTED_FUSED_ADAM,
            adam_w_mode=True,
            eps=1e-6,
            weight_decay=0.0,
            betas=[0.9, 0.999],
            sched=sched_config,
        )
        lora_tuning = LoraTuningConfig(
            loraplus_lr_ratio=8.0, alpha=32, adapter_dropout=0.01
        )
        return ModelConfig(
            hidden_dropout=0.0,
            attention_dropout=0.0,
            ffn_dropout=0.0,
            optim=optim_config,
            peft=PeftConfig(peft_scheme=PeftScheme.LORA, lora_tuning=lora_tuning),
        )

    def _create_default_training_config(self):
        return SFTTrainingConfig(
            max_length=8192,
            global_batch_size=32,
            trainer=self._create_default_trainer_config(),
            model=self._create_default_model_config(),
        )

    def test_sft_recipe_config_creation(self):
        run_config = BaseRunConfig(
            name="test-sft-job",
            model_type="amazon.nova-micro-v1:0:128k",
            model_name_or_path="nova-micro/prod",
            data_s3_path="s3://bucket/data",
            output_s3_path="s3://bucket/output",
            replicas=2,
        )
        recipe = SFTRecipeConfig(
            run=run_config, training_config=self._create_default_training_config()
        )

        self.assertEqual(recipe.run.name, "test-sft-job")
        self.assertIsInstance(recipe.training_config, SFTTrainingConfig)
        self.assertEqual(recipe.training_config.max_length, 8192)

    def test_sft_recipe_config_to_dict(self):
        run_config = BaseRunConfig(
            name="test-sft-job",
            model_type="amazon.nova-micro-v1:0:128k",
            model_name_or_path="nova-micro/prod",
            data_s3_path="s3://bucket/data",
            output_s3_path="s3://bucket/output",
            replicas=2,
        )
        recipe = SFTRecipeConfig(
            run=run_config, training_config=self._create_default_training_config()
        )
        recipe_dict = recipe.to_dict()

        self.assertIn("run", recipe_dict)
        self.assertIn("training_config", recipe_dict)
        self.assertEqual(recipe_dict["run"]["name"], "test-sft-job")
        self.assertEqual(
            recipe_dict["run"]["model_type"], "amazon.nova-micro-v1:0:128k"
        )
        self.assertEqual(recipe_dict["run"]["model_name_or_path"], "nova-micro/prod")
        self.assertEqual(recipe_dict["run"]["replicas"], 2)
        self.assertEqual(recipe_dict["run"]["data_s3_path"], "s3://bucket/data")
        self.assertEqual(recipe_dict["run"]["output_s3_path"], "s3://bucket/output")
        self.assertEqual(recipe_dict["training_config"]["max_length"], 8192)
        self.assertEqual(recipe_dict["training_config"]["global_batch_size"], 32)
        self.assertEqual(recipe_dict["training_config"]["trainer"]["max_epochs"], 2)
        self.assertIsInstance(recipe_dict["training_config"]["model"], dict)
        self.assertEqual(recipe_dict["training_config"]["model"]["hidden_dropout"], 0.0)
        self.assertEqual(
            recipe_dict["training_config"]["model"]["attention_dropout"], 0.0
        )
        self.assertEqual(recipe_dict["training_config"]["model"]["ffn_dropout"], 0.0)
        self.assertIsInstance(recipe_dict["training_config"]["model"]["optim"], dict)
        self.assertEqual(recipe_dict["training_config"]["model"]["optim"]["lr"], 1e-5)
        self.assertEqual(
            recipe_dict["training_config"]["model"]["optim"]["name"],
            "distributed_fused_adam",
        )
        self.assertEqual(
            recipe_dict["training_config"]["model"]["optim"]["adam_w_mode"], True
        )
        self.assertEqual(recipe_dict["training_config"]["model"]["optim"]["eps"], 1e-6)
        self.assertEqual(
            recipe_dict["training_config"]["model"]["optim"]["weight_decay"], 0.0
        )
        self.assertEqual(
            recipe_dict["training_config"]["model"]["optim"]["betas"], [0.9, 0.999]
        )
        self.assertIsInstance(
            recipe_dict["training_config"]["model"]["optim"]["sched"], dict
        )
        self.assertEqual(
            recipe_dict["training_config"]["model"]["optim"]["sched"]["warmup_steps"],
            10,
        )
        self.assertEqual(
            recipe_dict["training_config"]["model"]["optim"]["sched"]["constant_steps"],
            0,
        )
        self.assertEqual(
            recipe_dict["training_config"]["model"]["optim"]["sched"]["min_lr"], 1e-6
        )


if __name__ == "__main__":
    unittest.main()
