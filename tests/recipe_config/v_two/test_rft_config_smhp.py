import unittest

from amzn_nova_customization_sdk.recipe_config.v_two.rft_config_smhp import (
    AdvantageStrategy,
    ApiEndpoint,
    Data,
    Generator,
    LoraTuning,
    OptimConfig,
    Peft,
    PeftScheme,
    ReasoningEffort,
    Rewards,
    RFTRecipeConfig,
    RFTRunConfig,
    RFTTrainingConfig,
    Rollout,
    RolloutStrategy,
    Trainer,
    Type,
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
    def test_generator_config_creation(self):
        config = Generator(
            max_new_tokens=2048, set_random_seed=True, temperature=1, top_k=50
        )
        self.assertEqual(config.max_new_tokens, 2048)
        self.assertTrue(config.set_random_seed)
        self.assertEqual(config.temperature, 1)
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
    def test_advantage_strategy_config_creation(self):
        config = AdvantageStrategy(number_generation=4)
        self.assertEqual(config.number_generation, 4)

    def test_advantage_strategy_config_to_dict(self):
        config = AdvantageStrategy(number_generation=4)
        config_dict = config.to_dict()

        self.assertIn("number_generation", config_dict)
        self.assertEqual(config_dict["number_generation"], 4)


class TestRolloutStrategy(unittest.TestCase):
    def test_rollout_strategy_config_creation(self):
        config = RolloutStrategy(type=Type.OFF_POLICY_SYNC, age_tolerance=5)
        self.assertEqual(config.type, Type.OFF_POLICY_SYNC)
        self.assertEqual(config.age_tolerance, 5)

    def test_rollout_strategy_config_to_dict(self):
        config = RolloutStrategy(type=Type.OFF_POLICY_SYNC, age_tolerance=5)
        config_dict = config.to_dict()

        self.assertIn("type", config_dict)
        self.assertIn("age_tolerance", config_dict)
        self.assertEqual(config_dict["type"], "off_policy_async")
        self.assertEqual(config_dict["age_tolerance"], 5)


class TestTrainer(unittest.TestCase):
    def _create_default_optim_config(self):
        return OptimConfig(
            lr=7e-7,
            weight_decay=0.01,
            adam_beta1=0.9,
            adam_beta2=0.95,
        )

    def _create_default_peft_config(self):
        return Peft(peft_scheme=PeftScheme.LORA)

    def test_trainer_config_creation_with_peft(self):
        config = Trainer(
            max_steps=100,
            save_steps=20,
            save_top_k=5,
            refit_freq=4,
            clip_ratio_high=0.2,
            ent_coeff=0.001,
            loss_scale=1,
            optim_config=self._create_default_optim_config(),
            peft=self._create_default_peft_config(),
        )
        self.assertEqual(config.max_steps, 100)
        self.assertEqual(config.save_steps, 20)
        self.assertEqual(config.save_top_k, 5)
        self.assertEqual(config.clip_ratio_high, 0.2)
        self.assertEqual(config.ent_coeff, 0.001)
        self.assertEqual(config.loss_scale, 1)
        self.assertEqual(config.refit_freq, 4)
        self.assertIsInstance(config.optim_config, OptimConfig)
        self.assertEqual(config.optim_config.lr, 7e-7)
        self.assertEqual(config.optim_config.weight_decay, 0.01)
        self.assertEqual(config.optim_config.adam_beta1, 0.9)
        self.assertEqual(config.optim_config.adam_beta2, 0.95)
        self.assertIsInstance(config.peft, Peft)
        self.assertIsInstance(config.peft.lora_tuning, LoraTuning)
        self.assertEqual(config.peft.peft_scheme, PeftScheme.LORA)
        self.assertEqual(config.peft.lora_tuning.loraplus_lr_ratio, 64.0)
        self.assertEqual(config.peft.lora_tuning.alpha, 32)

    def test_trainer_config_creation_without_peft(self):
        config = Trainer(
            max_steps=100,
            save_steps=100,
            save_top_k=50,
            clip_ratio_high=0.2,
            ent_coeff=0.01,
            loss_scale=1,
            refit_freq=10,
            optim_config=self._create_default_optim_config(),
            peft=None,
        )
        self.assertIsNone(config.peft)

    def test_trainer_config_to_dict_with_peft(self):
        config = Trainer(
            max_steps=100,
            save_steps=100,
            save_top_k=50,
            clip_ratio_high=0.2,
            ent_coeff=0.01,
            loss_scale=1,
            refit_freq=10,
            optim_config=self._create_default_optim_config(),
            peft=self._create_default_peft_config(),
        )
        config_dict = config.to_dict()
        self.assertIn("max_steps", config_dict)
        self.assertIn("save_steps", config_dict)
        self.assertIn("optim_config", config_dict)
        self.assertIn("peft", config_dict)
        self.assertIsInstance(config_dict["peft"], dict)

    def test_trainer_config_to_dict_without_peft(self):
        config = Trainer(
            max_steps=100,
            save_steps=100,
            save_top_k=50,
            clip_ratio_high=0.2,
            ent_coeff=0.01,
            loss_scale=1,
            refit_freq=10,
            optim_config=self._create_default_optim_config(),
            peft=None,
        )
        config_dict = config.to_dict()
        self.assertIn("optim_config", config_dict)
        if "peft" in config_dict:
            self.assertIsNone(config_dict["peft"])


class TestRollout(unittest.TestCase):
    def _create_default_rollout_strategy_config(self):
        return RolloutStrategy(type=Type.OFF_POLICY_SYNC, age_tolerance=5)

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
            rollout_strategy=self._create_default_rollout_strategy_config(),
            advantage_strategy=self._create_default_advantage_strategy_config(),
            generator=self._create_default_generator_config(),
            rewards=self._create_default_rewards_config(),
        )
        self.assertIsInstance(config.rollout_strategy, RolloutStrategy)
        self.assertIsInstance(config.advantage_strategy, AdvantageStrategy)
        self.assertIsInstance(config.generator, Generator)
        self.assertIsInstance(config.rewards, Rewards)

    def test_rollout_config_to_dict(self):
        config = Rollout(
            rollout_strategy=self._create_default_rollout_strategy_config(),
            advantage_strategy=self._create_default_advantage_strategy_config(),
            generator=self._create_default_generator_config(),
            rewards=self._create_default_rewards_config(),
        )
        config_dict = config.to_dict()

        self.assertIn("rollout_strategy", config_dict)
        self.assertIn("advantage_strategy", config_dict)
        self.assertIn("generator", config_dict)
        self.assertIn("rewards", config_dict)
        self.assertIsInstance(config_dict["rollout_strategy"], dict)
        self.assertIsInstance(config_dict["advantage_strategy"], dict)
        self.assertIsInstance(config_dict["generator"], dict)
        self.assertIsInstance(config_dict["rewards"], dict)


class TestData(unittest.TestCase):
    def test_data_config_creation(self):
        config = Data(shuffle=True)
        self.assertTrue(config.shuffle)

    def test_data_config_to_dict(self):
        config = Data(shuffle=False)
        config_dict = config.to_dict()

        self.assertIn("shuffle", config_dict)
        self.assertFalse(config_dict["shuffle"])


class TestRFTTrainingConfig(unittest.TestCase):
    def _create_default_trainer_config(self):
        return Trainer(
            optim_config=OptimConfig(),
            peft=Peft(peft_scheme=PeftScheme.LORA),
        )

    def _create_default_rollout_config(self):
        return Rollout(
            rollout_strategy=RolloutStrategy(),
            advantage_strategy=AdvantageStrategy(),
            generator=Generator(),
            rewards=Rewards(
                api_endpoint=ApiEndpoint(
                    lambda_arn="arn:aws:lambda:us-east-1:123:function:r",
                )
            ),
        )

    def test_rft_training_config_creation(self):
        config = RFTTrainingConfig(
            global_batch_size=32,
            max_length=8192,
            reasoning_effort=ReasoningEffort.HIGH,
            data=Data(shuffle=True),
            rollout=self._create_default_rollout_config(),
            trainer=self._create_default_trainer_config(),
        )
        self.assertEqual(config.global_batch_size, 32)
        self.assertEqual(config.max_length, 8192)
        self.assertIsInstance(config.trainer, Trainer)

    def test_rft_training_config_to_dict(self):
        config = RFTTrainingConfig(
            global_batch_size=32,
            max_length=8192,
            reasoning_effort=ReasoningEffort.HIGH,
            data=Data(shuffle=True),
            rollout=self._create_default_rollout_config(),
            trainer=self._create_default_trainer_config(),
        )
        config_dict = config.to_dict()
        self.assertIn("global_batch_size", config_dict)
        self.assertIn("max_length", config_dict)
        self.assertIn("reasoning_effort", config_dict)
        self.assertIn("data", config_dict)
        self.assertIn("rollout", config_dict)
        self.assertIn("trainer", config_dict)
        self.assertEqual(config_dict["global_batch_size"], 32)
        self.assertEqual(config_dict["max_length"], 8192)
        self.assertEqual(config_dict["reasoning_effort"], "high")
        self.assertIsInstance(config_dict["data"], dict)
        self.assertIsInstance(config_dict["rollout"], dict)
        self.assertIsInstance(config_dict["trainer"], dict)


class TestRFTRunConfig(unittest.TestCase):
    def test_rft_run_config_creation(self):
        config = RFTRunConfig(
            name="test-rft-job",
            model_type="amazon.nova-2-lite-v1:0:256k",
            model_name_or_path="nova-lite-2/prod",
            data_s3_path="s3://bucket/data",
            output_s3_path="s3://bucket/output",
            replicas=2,
            generation_replicas=4,
            rollout_worker_replicas=8,
            reward_lambda_arn="arn:aws:lambda:us-east-1:123456789:function:reward",
        )
        self.assertEqual(config.name, "test-rft-job")
        self.assertEqual(config.model_type, "amazon.nova-2-lite-v1:0:256k")
        self.assertEqual(config.model_name_or_path, "nova-lite-2/prod")
        self.assertEqual(config.replicas, 2)
        self.assertEqual(config.generation_replicas, 4)
        self.assertEqual(config.rollout_worker_replicas, 8)
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
            replicas=2,
            generation_replicas=4,
            rollout_worker_replicas=8,
            reward_lambda_arn="arn:aws:lambda:us-east-1:123456789:function:reward",
        )
        config_dict = config.to_dict()

        self.assertIn("name", config_dict)
        self.assertIn("generation_replicas", config_dict)
        self.assertIn("rollout_worker_replicas", config_dict)
        self.assertIn("reward_lambda_arn", config_dict)
        self.assertEqual(config_dict["name"], "test-rft-job")
        self.assertEqual(config_dict["generation_replicas"], 4)


class TestRFTRecipeConfig(unittest.TestCase):
    def test_rft_recipe_config_creation(self):
        run_config = RFTRunConfig(
            name="test-rft-job",
            model_type="amazon.nova-2-lite-v1:0:256k",
            model_name_or_path="nova-lite-2/prod",
            data_s3_path="s3://bucket/data",
            output_s3_path="s3://bucket/output",
            replicas=2,
            generation_replicas=4,
            rollout_worker_replicas=8,
            reward_lambda_arn="arn:aws:lambda:us-east-1:123456789:function:reward",
        )
        training = RFTTrainingConfig(
            data=Data(),
            rollout=Rollout(
                rollout_strategy=RolloutStrategy(),
                advantage_strategy=AdvantageStrategy(),
                generator=Generator(),
                rewards=Rewards(
                    api_endpoint=ApiEndpoint(
                        lambda_arn="arn:aws:lambda:us-east-1:123:function:r"
                    )
                ),
            ),
            trainer=Trainer(optim_config=OptimConfig(), peft=None),
        )
        recipe = RFTRecipeConfig(run=run_config, training_config=training)

        self.assertEqual(recipe.run.name, "test-rft-job")
        self.assertIsInstance(recipe.training_config, RFTTrainingConfig)
        self.assertEqual(recipe.training_config.trainer.max_steps, 100)
        self.assertEqual(recipe.training_config.max_length, 10240)

    def test_rft_recipe_config_to_dict(self):
        run_config = RFTRunConfig(
            name="test-rft-job",
            model_type="amazon.nova-2-lite-v1:0:256k",
            model_name_or_path="nova-lite-2/prod",
            data_s3_path="s3://bucket/data",
            output_s3_path="s3://bucket/output",
            replicas=2,
            generation_replicas=4,
            rollout_worker_replicas=8,
            reward_lambda_arn="arn:aws:lambda:us-east-1:123456789:function:reward",
        )
        training = RFTTrainingConfig(
            data=Data(),
            rollout=Rollout(
                rollout_strategy=RolloutStrategy(),
                advantage_strategy=AdvantageStrategy(),
                generator=Generator(),
                rewards=Rewards(
                    api_endpoint=ApiEndpoint(
                        lambda_arn="arn:aws:lambda:us-east-1:123:function:r"
                    )
                ),
            ),
            trainer=Trainer(optim_config=OptimConfig(), peft=None),
        )
        recipe = RFTRecipeConfig(run=run_config, training_config=training)
        recipe_dict = recipe.to_dict()

        self.assertIn("run", recipe_dict)
        self.assertIn("training_config", recipe_dict)
        self.assertEqual(recipe_dict["run"]["name"], "test-rft-job")
        self.assertEqual(recipe_dict["training_config"]["trainer"]["max_steps"], 100)
        self.assertEqual(recipe_dict["training_config"]["global_batch_size"], 256)


if __name__ == "__main__":
    unittest.main()
