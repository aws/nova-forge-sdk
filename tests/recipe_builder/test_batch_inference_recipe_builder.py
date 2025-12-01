import unittest

from amzn_nova_customization_sdk.model.model_enums import Model, Platform
from amzn_nova_customization_sdk.recipe_builder.batch_inference_recipe_builder import (
    BatchInferenceRecipeBuilder,
)
from amzn_nova_customization_sdk.recipe_config.eval_config import (
    EvaluationMetric,
    EvaluationStrategy,
    EvaluationTask,
)


class TestBatchInferenceRecipeBuilder(unittest.TestCase):
    def setUp(self):
        self.job_name = "test-job"
        self.platform = Platform.SMTJ
        self.data_s3_path = "s3://test-bucket/data"
        self.output_s3_path = "s3://test-bucket/output"

    def test_batch_inference_recipe_builder_initialization(self):
        builder = BatchInferenceRecipeBuilder(
            job_name=self.job_name,
            platform=self.platform,
            model=Model.NOVA_LITE,
            model_path="nova-lite/prod",
            instance_type="ml.g5.48xlarge",
            instance_count=1,
            data_s3_path=self.data_s3_path,
            output_s3_path=self.output_s3_path,
            overrides={},
        )

        self.assertEqual(builder.model, Model.NOVA_LITE)
        self.assertEqual(builder.instance_type, "ml.g5.48xlarge")
        self.assertEqual(builder.instance_count, 1)
        self.assertEqual(builder.model_type, "amazon.nova-lite-v1:0:300k")
        self.assertEqual(builder.model_path, "nova-lite/prod")

    def test_create_batch_inference_run_config(self):
        builder = BatchInferenceRecipeBuilder(
            job_name=self.job_name,
            platform=self.platform,
            model=Model.NOVA_LITE,
            model_path="nova-lite/prod",
            instance_type="ml.g5.48xlarge",
            instance_count=1,
            data_s3_path=self.data_s3_path,
            output_s3_path=self.output_s3_path,
            overrides={},
        )

        run_config = builder._create_base_run_config()

        self.assertEqual(run_config.name, self.job_name)
        self.assertEqual(run_config.model_type, "amazon.nova-lite-v1:0:300k")
        self.assertEqual(run_config.data_s3_path, self.data_s3_path)
        self.assertEqual(run_config.output_s3_path, self.output_s3_path)
        self.assertEqual(run_config.replicas, 1)

    def test_batch_inference_with_full_overrides_config(self):
        builder = BatchInferenceRecipeBuilder(
            job_name=self.job_name,
            platform=self.platform,
            model=Model.NOVA_LITE,
            model_path="nova-lite/prod",
            instance_type="ml.g5.48xlarge",
            instance_count=1,
            data_s3_path=self.data_s3_path,
            output_s3_path=self.output_s3_path,
            overrides={
                "max_new_tokens": 2048,
                "top_k": -1,
                "top_p": 1.0,
                "temperature": 10.0,
            },
        )

        inference_config = builder._build_recipe_config()
        inference_config_dict = inference_config.to_dict()

        self.assertEqual(inference_config.run.name, self.job_name)
        self.assertEqual(inference_config.run.model_type, "amazon.nova-lite-v1:0:300k")
        self.assertEqual(inference_config.run.data_s3_path, "s3://test-bucket/data")
        self.assertEqual(inference_config.run.output_s3_path, self.output_s3_path)
        self.assertEqual(inference_config.run.replicas, 1)
        # Task has to be GEN_QA, Strategy has to be GEN_QA, and Metric has to be ALL for inference.
        self.assertEqual(inference_config.evaluation.task, EvaluationTask.GEN_QA)
        self.assertEqual(
            inference_config.evaluation.strategy, EvaluationStrategy.GEN_QA
        )
        self.assertEqual(inference_config.evaluation.metric, EvaluationMetric.ALL)
        self.assertIsNone(inference_config.evaluation.subtask)
        self.assertEqual(inference_config.inference.max_new_tokens, 2048)
        self.assertEqual(inference_config.inference.top_k, -1)
        self.assertEqual(inference_config.inference.top_p, 1.0)
        self.assertEqual(inference_config.inference.temperature, 10.0)
        self.assertEqual(
            inference_config_dict,
            {
                "run": {
                    "name": "test-job",
                    "model_type": "amazon.nova-lite-v1:0:300k",
                    "model_name_or_path": "nova-lite/prod",
                    "data_s3_path": "s3://test-bucket/data",
                    "output_s3_path": "s3://test-bucket/output",
                    "replicas": 1,
                },
                "evaluation": {
                    "task": "gen_qa",
                    "strategy": "gen_qa",
                    "metric": "all",
                },
                "inference": {
                    "max_new_tokens": 2048,
                    "top_k": -1,
                    "top_p": 1.0,
                    "temperature": 10.0,
                },
            },
        )

    def test_batch_inference_with_invalid_override_config(self):
        builder = BatchInferenceRecipeBuilder(
            job_name=self.job_name,
            platform=self.platform,
            model=Model.NOVA_LITE,
            model_path="nova-lite/prod",
            instance_type="ml.g5.48xlarge",
            instance_count=1,
            data_s3_path=self.data_s3_path,
            output_s3_path=self.output_s3_path,
            overrides={
                "max_new_tokens": 2048,
                "top_k": -1,
                "top_p": 1.0,
                "temperatures": 90.0,  # Incorrect name, should be temperature
            },
        )

        # Expected: It will ignore the incorrect override and will build without it.
        inference_config = builder._build_recipe_config()
        inference_config_dict = inference_config.to_dict()

        self.assertEqual(inference_config.run.name, self.job_name)
        self.assertEqual(inference_config.run.model_type, "amazon.nova-lite-v1:0:300k")
        self.assertEqual(inference_config.run.data_s3_path, "s3://test-bucket/data")
        self.assertEqual(inference_config.run.output_s3_path, self.output_s3_path)
        self.assertEqual(inference_config.run.replicas, 1)
        # Task has to be GEN_QA, Strategy has to be GEN_QA, and Metric has to be ALL for inference.
        self.assertEqual(inference_config.evaluation.task, EvaluationTask.GEN_QA)
        self.assertEqual(
            inference_config.evaluation.strategy, EvaluationStrategy.GEN_QA
        )
        self.assertEqual(inference_config.evaluation.metric, EvaluationMetric.ALL)
        self.assertIsNone(inference_config.evaluation.subtask)
        self.assertEqual(inference_config.inference.max_new_tokens, 2048)
        self.assertEqual(inference_config.inference.top_k, -1)
        self.assertEqual(inference_config.inference.top_p, 1.0)
        self.assertEqual(inference_config.inference.temperature, 0.0)
        self.assertEqual(
            inference_config_dict,
            {
                "run": {
                    "name": "test-job",
                    "model_type": "amazon.nova-lite-v1:0:300k",
                    "model_name_or_path": "nova-lite/prod",
                    "data_s3_path": "s3://test-bucket/data",
                    "output_s3_path": "s3://test-bucket/output",
                    "replicas": 1,
                },
                "evaluation": {
                    "task": "gen_qa",
                    "strategy": "gen_qa",
                    "metric": "all",
                },
                "inference": {
                    "max_new_tokens": 2048,
                    "top_k": -1,
                    "top_p": 1.0,
                    "temperature": 0.0,
                },
            },
        )


if __name__ == "__main__":
    unittest.main()
