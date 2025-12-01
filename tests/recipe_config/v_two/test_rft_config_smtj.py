import unittest

from amzn_nova_customization_sdk.recipe_config.v_two.rft_config_smtj import (
    AdvantageStrategy,
    ApiEndpoint,
    Generator,
    LoraTuning,
    Model,
    Optim,
    Optimizer,
    Peft,
    PeftScheme,
    ReasoningEffort,
    Rewards,
    RFTRecipeConfig,
    RFTRunConfig,
    RFTTrainingConfig,
    Rollout,
    Trainer,
)


class TestApiEndpoint(unittest.TestCase):
    def test_api_endpoint_config_creation(self):
        config = ApiEndpoint(
            lambda_arn="arn:aws:lambda:us-east-1:123456789:function:reward",
            lambda_concurrency_limit=10,
        )
        self.assertEqual(
            config.lambda_arn, "arn:aws:lambda:us-east-1:123456789:function:reward"
        )
        self.assertEqual(config.lambda_concurrency_limit, 10)

    def test_api_endpoint_config_default_concurrency_limit(self):
        config = ApiEndpoint(
            lambda_arn="arn:aws:lambda:us-east-1:123456789:function:reward"
        )
        self.assertEqual(config.lambda_concurrency_limit, 12)

    def test_api_endpoint_config_to_dict(self):
        config = ApiEndpoint(
            lambda_arn="arn:aws:lambda:us-east-1:123456789:function:reward",
            lambda_concurrency_limit=10,
        )
        config_dict = config.to_dict()

        self.assertIn("lambda_arn", config_dict)
        self.assertIn("lambda_concurrency_limit", config_dict)
        self.assertEqual(
            config_dict["lambda_arn"],
            "arn:aws:lambda:us-east-1:123456789:function:reward",
        )
        self.assertEqual(config_dict["lambda_concurrency_limit"], 10)


class TestRewards(unittest.TestCase):
    def _create_default_api_endpoint_config(self):
        return ApiEndpoint(
            lambda_arn="arn:aws:lambda:us-east-1:123456789:function:reward",
            lambda_concurrency_limit=10,
        )

    def test_rewards_config_creation(self):
        api_endpoint = self._create_default_api_endpoint_config()
        config = Rewards(api_endpoint=api_endpoint)

        self.assertIsInstance(config.api_endpoint, ApiEndpoint)
        self.assertEqual(
            config.api_endpoint.lambda_arn,
            "arn:aws:lambda:us-east-1:123456789:function:reward",
        )
        self.assertEqual(config.api_endpoint.lambda_concurrency_limit, 10)

    def test_rewards_config_to_dict(self):
        api_endpoint = self._create_default_api_endpoint_config()
        config = Rewards(api_endpoint=api_endpoint)
        config_dict = config.to_dict()

        self.assertIn("api_endpoint", config_dict)
        self.assertIsInstance(config_dict["api_endpoint"], dict)
        self.assertEqual(
            config_dict["api_endpoint"]["lambda_arn"],
            "arn:aws:lambda:us-east-1:123456789:function:reward",
        )
        self.assertEqual(config_dict["api_endpoint"]["lambda_concurrency_limit"], 10)


class TestGenerator(unittest.TestCase):
    def test_generator_config_creation_with_defaults(self):
        config = Generator()
        self.assertEqual(config.max_new_tokens, 6000)
        self.assertTrue(config.set_random_seed)
        self.assertEqual(config.temperature, 1)
        self.assertEqual(config.top_k, 1)

    def test_generator_config_creation_with_custom_values(self):
        config = Generator(
            max_new_tokens=2048, set_random_seed=False, temperature=2, top_k=50
        )
        self.assertEqual(config.max_new_tokens, 2048)
        self.assertFalse(config.set_random_seed)
        self.assertEqual(config.temperature, 2)
        self.assertEqual(config.top_k, 50)

    def test_generator_config_to_dict(self):
        config = Generator(
            max_new_tokens=2048, set_random_seed=True, temperature=1, top_k=50
        )
        config_dict = config.to_dict()

        self.assertIn("max_new_tokens", config_dict)
        self.assertIn("set_random_seed", config_dict)
        self.assertIn("temperature", config_dict)
        self.assertIn("top_k", config_dict)
        self.assertEqual(config_dict["max_new_tokens"], 2048)
        self.assertTrue(config_dict["set_random_seed"])
        self.assertEqual(config_dict["temperature"], 1)
        self.assertEqual(config_dict["top_k"], 50)


class TestAdvantageStrategy(unittest.TestCase):
    def test_advantage_strategy_config_creation_with_default(self):
        config = AdvantageStrategy()
        self.assertEqual(config.number_generation, 2)

    def test_advantage_strategy_config_creation_with_custom_value(self):
        config = AdvantageStrategy(number_generation=4)
        self.assertEqual(config.number_generation, 4)

    def test_advantage_strategy_config_to_dict(self):
        config = AdvantageStrategy(number_generation=4)
        config_dict = config.to_dict()

        self.assertIn("number_generation", config_dict)
        self.assertEqual(config_dict["number_generation"], 4)


class TestOptim(unittest.TestCase):
    def test_optim_config_creation_with_defaults(self):
        config = Optim()
        self.assertEqual(config.optimizer, Optimizer.ADAM)
        self.assertEqual(config.lr, 1e-6)
        self.assertEqual(config.min_lr, 0.0)

    def test_optim_config_creation_with_custom_values(self):
        config = Optim(optimizer=Optimizer.ADAM, lr=5e-7, min_lr=1e-8)
        self.assertEqual(config.optimizer, Optimizer.ADAM)
        self.assertEqual(config.lr, 5e-7)
        self.assertEqual(config.min_lr, 1e-8)

    def test_optim_config_to_dict(self):
        config = Optim(lr=5e-7, min_lr=1e-8)
        config_dict = config.to_dict()

        self.assertIn("optimizer", config_dict)
        self.assertIn("lr", config_dict)
        self.assertIn("min_lr", config_dict)
        self.assertEqual(config_dict["optimizer"], "adam")
        self.assertEqual(config_dict["lr"], 5e-7)
        self.assertEqual(config_dict["min_lr"], 1e-8)


class TestLoraTuning(unittest.TestCase):
    def test_lora_tuning_config_creation_with_defaults(self):
        config = LoraTuning()
        self.assertEqual(config.loraplus_lr_ratio, 16.0)
        self.assertEqual(config.alpha, 128)
        self.assertEqual(config.adapter_dropout, 0.0)

    def test_lora_tuning_config_creation_with_custom_values(self):
        config = LoraTuning(loraplus_lr_ratio=32.0, alpha=64, adapter_dropout=0.1)
        self.assertEqual(config.loraplus_lr_ratio, 32.0)
        self.assertEqual(config.alpha, 64)
        self.assertEqual(config.adapter_dropout, 0.1)

    def test_lora_tuning_config_to_dict(self):
        config = LoraTuning(loraplus_lr_ratio=32.0, alpha=64, adapter_dropout=0.1)
        config_dict = config.to_dict()

        self.assertIn("loraplus_lr_ratio", config_dict)
        self.assertIn("alpha", config_dict)
        self.assertIn("adapter_dropout", config_dict)
        self.assertEqual(config_dict["loraplus_lr_ratio"], 32.0)
        self.assertEqual(config_dict["alpha"], 64)
        self.assertEqual(config_dict["adapter_dropout"], 0.1)


class TestPeft(unittest.TestCase):
    def test_peft_config_creation_with_lora(self):
        config = Peft(peft_scheme=PeftScheme.LORA)
        self.assertEqual(config.peft_scheme, PeftScheme.LORA)
        self.assertIsInstance(config.lora_tuning, LoraTuning)
        self.assertEqual(config.lora_tuning.loraplus_lr_ratio, 16.0)
        self.assertEqual(config.lora_tuning.alpha, 128)

    def test_peft_config_creation_with_custom_lora_tuning(self):
        custom_lora = LoraTuning(loraplus_lr_ratio=32.0, alpha=64)
        config = Peft(peft_scheme=PeftScheme.LORA, lora_tuning=custom_lora)
        self.assertEqual(config.lora_tuning.loraplus_lr_ratio, 32.0)
        self.assertEqual(config.lora_tuning.alpha, 64)

    def test_peft_config_creation_with_null(self):
        config = Peft(peft_scheme=PeftScheme.NULL)
        self.assertEqual(config.peft_scheme, PeftScheme.NULL)
        self.assertIsNone(config.lora_tuning)

    def test_peft_config_to_dict(self):
        config = Peft(peft_scheme=PeftScheme.LORA)
        config_dict = config.to_dict()

        self.assertIn("peft_scheme", config_dict)
        self.assertIn("lora_tuning", config_dict)
        self.assertEqual(config_dict["peft_scheme"], "lora")
        self.assertIsInstance(config_dict["lora_tuning"], dict)


class TestModel(unittest.TestCase):
    def _create_default_peft_config(self):
        return Peft(peft_scheme=PeftScheme.LORA)

    def test_model_config_creation(self):
        peft = self._create_default_peft_config()
        config = Model(peft=peft)
        self.assertIsInstance(config.peft, Peft)
        self.assertEqual(config.peft.peft_scheme, PeftScheme.LORA)

    def test_model_config_to_dict(self):
        peft = self._create_default_peft_config()
        config = Model(peft=peft)
        config_dict = config.to_dict()

        self.assertIn("peft", config_dict)
        self.assertIsInstance(config_dict["peft"], dict)


class TestTrainer(unittest.TestCase):
    def _create_default_optim_config(self):
        return Optim(lr=5e-7)

    def test_trainer_config_creation_with_defaults(self):
        config = Trainer(optim=self._create_default_optim_config())
        self.assertIsInstance(config.optim, Optim)
        self.assertEqual(config.entropy_coeff, 0.0)
        self.assertEqual(config.kl_loss_coef, 0.001)

    def test_trainer_config_creation_with_custom_values(self):
        config = Trainer(
            optim=self._create_default_optim_config(),
            entropy_coeff=0.01,
            kl_loss_coef=0.005,
        )
        self.assertEqual(config.entropy_coeff, 0.01)
        self.assertEqual(config.kl_loss_coef, 0.005)

    def test_trainer_config_to_dict(self):
        config = Trainer(
            optim=self._create_default_optim_config(),
            entropy_coeff=0.01,
            kl_loss_coef=0.005,
        )
        config_dict = config.to_dict()

        self.assertIn("optim", config_dict)
        self.assertIn("entropy_coeff", config_dict)
        self.assertIn("kl_loss_coef", config_dict)
        self.assertIsInstance(config_dict["optim"], dict)
        self.assertEqual(config_dict["entropy_coeff"], 0.01)
        self.assertEqual(config_dict["kl_loss_coef"], 0.005)


class TestRollout(unittest.TestCase):
    def _create_default_advantage_strategy_config(self):
        return AdvantageStrategy(number_generation=4)

    def _create_default_generator_config(self):
        return Generator(
            max_new_tokens=2048, set_random_seed=True, temperature=1, top_k=50
        )

    def _create_default_rewards_config(self):
        api_endpoint = ApiEndpoint(
            lambda_arn="arn:aws:lambda:us-east-1:123456789:function:reward",
            lambda_concurrency_limit=10,
        )
        return Rewards(api_endpoint=api_endpoint)

    def test_rollout_config_creation(self):
        config = Rollout(
            advantage_strategy=self._create_default_advantage_strategy_config(),
            generator=self._create_default_generator_config(),
            rewards=self._create_default_rewards_config(),
        )
        self.assertIsInstance(config.advantage_strategy, AdvantageStrategy)
        self.assertIsInstance(config.generator, Generator)
        self.assertIsInstance(config.rewards, Rewards)

    def test_rollout_config_to_dict(self):
        config = Rollout(
            advantage_strategy=self._create_default_advantage_strategy_config(),
            generator=self._create_default_generator_config(),
            rewards=self._create_default_rewards_config(),
        )
        config_dict = config.to_dict()

        self.assertIn("advantage_strategy", config_dict)
        self.assertIn("generator", config_dict)
        self.assertIn("rewards", config_dict)
        self.assertIsInstance(config_dict["advantage_strategy"], dict)
        self.assertIsInstance(config_dict["generator"], dict)
        self.assertIsInstance(config_dict["rewards"], dict)


class TestRFTTrainingConfig(unittest.TestCase):
    def _create_default_trainer_config(self):
        return Trainer(optim=Optim())

    def _create_default_model_config(self):
        return Model(peft=Peft(peft_scheme=PeftScheme.LORA))

    def _create_default_rollout_config(self):
        return Rollout(
            advantage_strategy=AdvantageStrategy(),
            generator=Generator(),
            rewards=Rewards(
                api_endpoint=ApiEndpoint(
                    lambda_arn="arn:aws:lambda:us-east-1:123:function:r",
                )
            ),
        )

    def test_rft_training_config_creation_with_defaults(self):
        config = RFTTrainingConfig(
            trainer=self._create_default_trainer_config(),
            model=self._create_default_model_config(),
            rollout=self._create_default_rollout_config(),
        )
        self.assertEqual(config.max_epochs, 2)
        self.assertEqual(config.max_length, 8192)
        self.assertEqual(config.global_batch_size, 16)
        self.assertEqual(config.reasoning_effort, ReasoningEffort.HIGH)

    def test_rft_training_config_creation_with_custom_values(self):
        config = RFTTrainingConfig(
            trainer=self._create_default_trainer_config(),
            model=self._create_default_model_config(),
            rollout=self._create_default_rollout_config(),
            max_epochs=5,
            max_length=4096,
            global_batch_size=32,
            reasoning_effort=ReasoningEffort.HIGH,
        )
        self.assertEqual(config.max_epochs, 5)
        self.assertEqual(config.max_length, 4096)
        self.assertEqual(config.global_batch_size, 32)
        self.assertIsInstance(config.trainer, Trainer)
        self.assertIsInstance(config.model, Model)
        self.assertIsInstance(config.rollout, Rollout)

    def test_rft_training_config_to_dict(self):
        config = RFTTrainingConfig(
            trainer=self._create_default_trainer_config(),
            model=self._create_default_model_config(),
            rollout=self._create_default_rollout_config(),
            max_epochs=3,
            max_length=4096,
            global_batch_size=32,
        )
        config_dict = config.to_dict()

        self.assertIn("trainer", config_dict)
        self.assertIn("model", config_dict)
        self.assertIn("rollout", config_dict)
        self.assertIn("max_epochs", config_dict)
        self.assertIn("max_length", config_dict)
        self.assertIn("global_batch_size", config_dict)
        self.assertIn("reasoning_effort", config_dict)
        self.assertIsInstance(config_dict["trainer"], dict)
        self.assertIsInstance(config_dict["model"], dict)
        self.assertIsInstance(config_dict["rollout"], dict)
        self.assertEqual(config_dict["max_epochs"], 3)
        self.assertEqual(config_dict["max_length"], 4096)
        self.assertEqual(config_dict["global_batch_size"], 32)
        self.assertEqual(config_dict["reasoning_effort"], "high")


class TestRFTRunConfig(unittest.TestCase):
    def test_rft_run_config_creation(self):
        config = RFTRunConfig(
            name="test-rft-job",
            model_type="amazon.nova-2-lite-v1:0:256k",
            model_name_or_path="nova-lite-2/prod",
            data_s3_path="s3://bucket/data",
            output_s3_path="s3://bucket/output",
            replicas=4,
            reward_lambda_arn="arn:aws:lambda:us-east-1:123456789:function:reward",
        )
        self.assertEqual(config.name, "test-rft-job")
        self.assertEqual(config.model_type, "amazon.nova-2-lite-v1:0:256k")
        self.assertEqual(config.model_name_or_path, "nova-lite-2/prod")
        self.assertEqual(config.data_s3_path, "s3://bucket/data")
        self.assertEqual(config.output_s3_path, "s3://bucket/output")
        self.assertEqual(config.replicas, 4)
        self.assertEqual(
            config.reward_lambda_arn,
            "arn:aws:lambda:us-east-1:123456789:function:reward",
        )

    def test_rft_run_config_to_dict(self):
        config = RFTRunConfig(
            name="test-rft-job",
            model_type="amazon.nova-2-lite-v1:0:256k",
            model_name_or_path="nova-lite-2/prod",
            data_s3_path="s3://bucket/data",
            output_s3_path="s3://bucket/output",
            replicas=4,
            reward_lambda_arn="arn:aws:lambda:us-east-1:123456789:function:reward",
        )
        config_dict = config.to_dict()

        self.assertIn("name", config_dict)
        self.assertIn("model_type", config_dict)
        self.assertIn("model_name_or_path", config_dict)
        self.assertIn("data_s3_path", config_dict)
        self.assertIn("output_s3_path", config_dict)
        self.assertIn("replicas", config_dict)
        self.assertIn("reward_lambda_arn", config_dict)
        self.assertEqual(config_dict["name"], "test-rft-job")
        self.assertEqual(config_dict["model_type"], "amazon.nova-2-lite-v1:0:256k")
        self.assertEqual(config_dict["model_name_or_path"], "nova-lite-2/prod")
        self.assertEqual(config_dict["data_s3_path"], "s3://bucket/data")
        self.assertEqual(config_dict["output_s3_path"], "s3://bucket/output")
        self.assertEqual(config_dict["replicas"], 4)
        self.assertEqual(
            config_dict["reward_lambda_arn"],
            "arn:aws:lambda:us-east-1:123456789:function:reward",
        )


class TestRFTRecipeConfig(unittest.TestCase):
    def test_rft_recipe_config_creation(self):
        run_config = RFTRunConfig(
            name="test-rft-job",
            model_type="amazon.nova-2-lite-v1:0:256k",
            model_name_or_path="nova-lite-2/prod",
            data_s3_path="s3://bucket/data",
            output_s3_path="s3://bucket/output",
            replicas=4,
            reward_lambda_arn="arn:aws:lambda:us-east-1:123456789:function:reward",
        )
        training = RFTTrainingConfig(
            trainer=Trainer(optim=Optim()),
            model=Model(peft=Peft(peft_scheme=PeftScheme.LORA)),
            rollout=Rollout(
                advantage_strategy=AdvantageStrategy(),
                generator=Generator(),
                rewards=Rewards(
                    api_endpoint=ApiEndpoint(
                        lambda_arn="arn:aws:lambda:us-east-1:123:function:r"
                    )
                ),
            ),
        )
        recipe = RFTRecipeConfig(run=run_config, training_config=training)

        self.assertEqual(recipe.run.name, "test-rft-job")
        self.assertIsInstance(recipe.training_config, RFTTrainingConfig)
        self.assertEqual(recipe.training_config.max_epochs, 2)
        self.assertEqual(recipe.training_config.max_length, 8192)
        self.assertEqual(recipe.training_config.global_batch_size, 16)

    def test_rft_recipe_config_to_dict(self):
        run_config = RFTRunConfig(
            name="test-rft-job",
            model_type="amazon.nova-2-lite-v1:0:256k",
            model_name_or_path="nova-lite-2/prod",
            data_s3_path="s3://bucket/data",
            output_s3_path="s3://bucket/output",
            replicas=4,
            reward_lambda_arn="arn:aws:lambda:us-east-1:123456789:function:reward",
        )
        training = RFTTrainingConfig(
            trainer=Trainer(optim=Optim()),
            model=Model(peft=Peft(peft_scheme=PeftScheme.LORA)),
            rollout=Rollout(
                advantage_strategy=AdvantageStrategy(),
                generator=Generator(),
                rewards=Rewards(
                    api_endpoint=ApiEndpoint(
                        lambda_arn="arn:aws:lambda:us-east-1:123:function:r"
                    )
                ),
            ),
        )
        recipe = RFTRecipeConfig(run=run_config, training_config=training)
        recipe_dict = recipe.to_dict()

        self.assertIn("run", recipe_dict)
        self.assertIn("training_config", recipe_dict)
        self.assertEqual(recipe_dict["run"]["name"], "test-rft-job")
        self.assertEqual(recipe_dict["training_config"]["max_epochs"], 2)
        self.assertEqual(recipe_dict["training_config"]["global_batch_size"], 16)


if __name__ == "__main__":
    unittest.main()
