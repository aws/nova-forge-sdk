import unittest

from amzn_nova_customization_sdk.recipe_config.eval_config import (
    Aggregation,
    ProcessingConfig,
    ProcessorConfig,
)


class TestV1EvalConfig(unittest.TestCase):
    def test_processor_config_from_dict_with_default_value(self):
        """Test ProcessorConfig with dict preprocessing"""
        config = ProcessorConfig.from_dict(
            {
                "lambda_arn": "arn:aws:lambda:us-east-1:123456789012:function:test",
            }
        )

        self.assertIsInstance(config.preprocessing, ProcessingConfig)
        self.assertTrue(config.preprocessing.enabled)
        self.assertTrue(config.postprocessing.enabled)
        self.assertEqual(config.aggregation, Aggregation.AVERAGE)

    def test_processor_config_with_dict_preprocessing(self):
        """Test ProcessorConfig with dict preprocessing"""
        config = ProcessorConfig.from_dict(
            {
                "lambda_arn": "arn:aws:lambda:us-east-1:123456789012:function:test",
                "preprocessing": {"enabled": "False"},
            }
        )

        self.assertIsInstance(config.preprocessing, ProcessingConfig)
        self.assertFalse(config.preprocessing.enabled)

    def test_processor_config_with_dict_postprocessing(self):
        """Test ProcessorConfig with dict postprocessing"""
        config = ProcessorConfig.from_dict(
            {
                "lambda_arn": "arn:aws:lambda:us-east-1:123456789012:function:test",
                "postprocessing": {"enabled": "False"},
            }
        )

        self.assertIsInstance(config.postprocessing, ProcessingConfig)
        self.assertFalse(config.postprocessing.enabled)

    def test_processor_config_with_both_dict_configs(self):
        """Test ProcessorConfig with both preprocessing and postprocessing as dicts"""
        config = ProcessorConfig.from_dict(
            {
                "lambda_arn": "arn:aws:lambda:us-east-1:123456789012:function:test",
                "preprocessing": {"enabled": "False"},
                "postprocessing": {"enabled": "True"},
                "aggregation": "average",
            }
        )

        self.assertIsInstance(config.preprocessing, ProcessingConfig)
        self.assertIsInstance(config.postprocessing, ProcessingConfig)
        self.assertFalse(config.preprocessing.enabled)
        self.assertTrue(config.postprocessing.enabled)
        self.assertEqual(config.aggregation, Aggregation.AVERAGE)

    def test_processor_config_with_object_configs(self):
        """Test ProcessorConfig with ProcessingConfig objects (no conversion needed)"""
        preprocessing = ProcessingConfig(enabled=False)
        postprocessing = ProcessingConfig(enabled=True)

        config = ProcessorConfig(
            lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:test",
            preprocessing=preprocessing,
            postprocessing=postprocessing,
        )

        self.assertIs(config.preprocessing, preprocessing)
        self.assertIs(config.postprocessing, postprocessing)


if __name__ == "__main__":
    unittest.main()
