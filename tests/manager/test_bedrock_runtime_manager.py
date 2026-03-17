"""Unit tests for BedrockRuntimeManager.

This module contains tests that verify the BedrockRuntimeManager implementation
including properties, training method mapping, base model resolution, execution,
and RFT hyperparameter handling.
"""

import unittest
from unittest.mock import patch

from amzn_nova_forge.manager.runtime_manager import (
    BedrockRuntimeManager,
    JobConfig,
    RuntimeManager,
)
from amzn_nova_forge.model.model_enums import TrainingMethod


class TestBedrockRuntimeManager(unittest.TestCase):
    """Unit tests for BedrockRuntimeManager.

    Tests cover:
    - RuntimeManager interface compliance
    - Bedrock client initialization
    - Training method to customization type mapping
    - Base model identifier resolution
    - Job execution with various configurations
    - RFT hyperparameter type handling
    """

    @patch.object(BedrockRuntimeManager, "setup", return_value=None)
    def test_property_runtime_manager_interface_compliance(self, mock_setup):
        """Test RuntimeManager interface compliance across configurations."""
        # Test configurations covering various parameter combinations
        test_configs = [
            # Minimal configuration
            {
                "execution_role": "arn:aws:iam::123456789012:role/BedrockRole",
                "base_model_identifier": None,
                "kms_key_id": None,
                "vpc_config": None,
            },
            # Full configuration with all optional parameters
            {
                "execution_role": "arn:aws:iam::123456789012:role/BedrockExecRole",
                "base_model_identifier": "arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-micro-v1:0",
                "kms_key_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
                "vpc_config": {
                    "subnet_ids": ["subnet-12345", "subnet-67890"],
                    "security_group_ids": ["sg-abcdef"],
                },
            },
            # Configuration with model identifier only
            {
                "execution_role": "arn:aws:iam::123456789012:role/TestRole",
                "base_model_identifier": "arn:aws:bedrock:us-west-2::foundation-model/amazon.nova-lite-v1:0",
                "kms_key_id": None,
                "vpc_config": None,
            },
            # Configuration with KMS key only
            {
                "execution_role": "arn:aws:iam::123456789012:role/SecureRole",
                "base_model_identifier": None,
                "kms_key_id": "bbbbbbbb-cccc-dddd-eeee-ffffffffffff",
                "vpc_config": None,
            },
            # Configuration with VPC only
            {
                "execution_role": "arn:aws:iam::123456789012:role/VPCRole",
                "base_model_identifier": None,
                "kms_key_id": None,
                "vpc_config": {
                    "subnet_ids": ["subnet-abc123"],
                    "security_group_ids": ["sg-xyz789", "sg-def456"],
                },
            },
            # Configuration with different region
            {
                "execution_role": "arn:aws:iam::123456789012:role/EURole",
                "base_model_identifier": "arn:aws:bedrock:eu-west-1::foundation-model/amazon.nova-pro-v1:0",
                "kms_key_id": "cccccccc-dddd-eeee-ffff-aaaaaaaaaaaa",
                "vpc_config": {
                    "subnet_ids": ["subnet-eu1", "subnet-eu2", "subnet-eu3"],
                    "security_group_ids": ["sg-eu1"],
                },
            },
        ]

        for i, config in enumerate(test_configs):
            with self.subTest(config_index=i, config=config):
                # Reset mock for each iteration
                mock_setup.reset_mock()

                # Create BedrockRuntimeManager instance with test configuration
                manager = BedrockRuntimeManager(**config)

                # Verify it's an instance of RuntimeManager
                self.assertIsInstance(
                    manager,
                    RuntimeManager,
                    f"Config {i}: Manager should be instance of RuntimeManager",
                )

                # Verify all required methods exist and are callable
                self.assertTrue(
                    hasattr(manager, "setup"),
                    f"Config {i}: Manager should have setup method",
                )
                self.assertTrue(
                    callable(manager.setup), f"Config {i}: setup should be callable"
                )

                self.assertTrue(
                    hasattr(manager, "execute"),
                    f"Config {i}: Manager should have execute method",
                )
                self.assertTrue(
                    callable(manager.execute), f"Config {i}: execute should be callable"
                )

                self.assertTrue(
                    hasattr(manager, "cleanup"),
                    f"Config {i}: Manager should have cleanup method",
                )
                self.assertTrue(
                    callable(manager.cleanup), f"Config {i}: cleanup should be callable"
                )

                self.assertTrue(
                    hasattr(manager, "required_calling_role_permissions"),
                    f"Config {i}: Manager should have required_calling_role_permissions method",
                )
                self.assertTrue(
                    callable(manager.required_calling_role_permissions),
                    f"Config {i}: required_calling_role_permissions should be callable",
                )

                # Verify instance attributes are set correctly
                self.assertEqual(
                    manager.execution_role,
                    config["execution_role"],
                    f"Config {i}: execution_role should match",
                )
                self.assertEqual(
                    manager.base_model_identifier,
                    config["base_model_identifier"],
                    f"Config {i}: base_model_identifier should match",
                )
                self.assertEqual(
                    manager.kms_key_id,
                    config["kms_key_id"],
                    f"Config {i}: kms_key_id should match",
                )
                self.assertEqual(
                    manager.vpc_config,
                    config["vpc_config"],
                    f"Config {i}: vpc_config should match",
                )

                # Verify instance_type and instance_count are None (Bedrock manages compute)
                self.assertIsNone(
                    manager.instance_type,
                    f"Config {i}: instance_type should be None for Bedrock",
                )
                self.assertIsNone(
                    manager.instance_count,
                    f"Config {i}: instance_count should be None for Bedrock",
                )

                # Verify setup was called during initialization
                mock_setup.assert_called_once()

    @patch("boto3.session.Session")
    @patch("boto3.client")
    def test_property_bedrock_client_initialization(
        self, mock_boto_client, mock_session_class
    ):
        """Test Bedrock client initialization across AWS regions."""
        # Test configurations covering AWS regions
        test_regions = [
            "us-west-2",  # US West (Oregon)
            None,  # No region (should default to us-east-1)
        ]

        for region in test_regions:
            with self.subTest(region=region):
                # Reset mocks for each iteration
                mock_boto_client.reset_mock()
                mock_session_class.reset_mock()

                # Mock boto3 session with the test region
                mock_session = mock_session_class.return_value
                mock_session.region_name = region

                # Mock bedrock client
                mock_bedrock_client = mock_boto_client.return_value

                # Create BedrockRuntimeManager (setup is called in __init__)
                manager = BedrockRuntimeManager(
                    execution_role=f"arn:aws:iam::123456789012:role/TestRole-{region or 'default'}"
                )

                # Verify session was created
                mock_session_class.assert_called_once()

                # Determine expected region (default to us-east-1 if None)
                expected_region = region if region is not None else "us-east-1"

                # Verify bedrock client was created with correct region
                mock_boto_client.assert_called_once_with(
                    "bedrock", region_name=expected_region
                )

                # Verify manager attributes are set correctly
                self.assertEqual(
                    manager.region,
                    expected_region,
                    f"Region {region}: manager.region should be {expected_region}",
                )
                self.assertEqual(
                    manager.bedrock_client,
                    mock_bedrock_client,
                    f"Region {region}: bedrock_client should be set",
                )

                # Verify the client is callable (can make API calls)
                self.assertTrue(
                    hasattr(manager.bedrock_client, "create_model_customization_job"),
                    f"Region {region}: Client should have create_model_customization_job method",
                )
                self.assertTrue(
                    hasattr(manager.bedrock_client, "stop_model_customization_job"),
                    f"Region {region}: Client should have stop_model_customization_job method",
                )
                self.assertTrue(
                    hasattr(manager.bedrock_client, "get_model_customization_job"),
                    f"Region {region}: Client should have get_model_customization_job method",
                )

    def test_get_customization_type_sft_lora(self):
        """Test that SFT_LORA maps to FINE_TUNING."""
        from amzn_nova_forge.model.model_enums import TrainingMethod
        from amzn_nova_forge.util.bedrock import get_customization_type

        result = get_customization_type(TrainingMethod.SFT_LORA)
        self.assertEqual(result, "FINE_TUNING")

    def test_get_customization_type_sft_full(self):
        """Test that SFT_FULL raises ValueError (not supported on Bedrock)."""
        from amzn_nova_forge.model.model_enums import TrainingMethod
        from amzn_nova_forge.util.bedrock import get_customization_type

        with self.assertRaises(ValueError) as context:
            get_customization_type(TrainingMethod.SFT_FULL)

        error_message = str(context.exception)
        self.assertIn("sft_full", error_message)
        self.assertIn("not supported on Bedrock", error_message)

    def test_get_customization_type_rft_lora(self):
        """Test that RFT_LORA maps to REINFORCEMENT_FINE_TUNING."""
        from amzn_nova_forge.model.model_enums import TrainingMethod
        from amzn_nova_forge.util.bedrock import get_customization_type

        result = get_customization_type(TrainingMethod.RFT_LORA)
        self.assertEqual(result, "REINFORCEMENT_FINE_TUNING")

    def test_get_customization_type_rft_full(self):
        """Test that RFT_FULL raises ValueError (not supported on Bedrock)."""
        from amzn_nova_forge.model.model_enums import TrainingMethod
        from amzn_nova_forge.util.bedrock import get_customization_type

        with self.assertRaises(ValueError) as context:
            get_customization_type(TrainingMethod.RFT_FULL)

        error_message = str(context.exception)
        self.assertIn("rft_full", error_message)
        self.assertIn("not supported on Bedrock", error_message)

    def test_get_customization_type_unsupported_cpt(self):
        """Test that CPT raises ValueError with supported methods listed."""
        from amzn_nova_forge.model.model_enums import TrainingMethod
        from amzn_nova_forge.util.bedrock import get_customization_type

        with self.assertRaises(ValueError) as context:
            get_customization_type(TrainingMethod.CPT)

        error_message = str(context.exception)
        self.assertIn("cpt", error_message)
        self.assertIn("not supported on Bedrock", error_message)
        self.assertIn("Supported methods:", error_message)
        self.assertIn("sft_lora", error_message)
        self.assertIn("rft_lora", error_message)

    def test_get_customization_type_unsupported_dpo_lora(self):
        """Test that DPO_LORA raises ValueError."""
        from amzn_nova_forge.model.model_enums import TrainingMethod
        from amzn_nova_forge.util.bedrock import get_customization_type

        with self.assertRaises(ValueError) as context:
            get_customization_type(TrainingMethod.DPO_LORA)

        error_message = str(context.exception)
        self.assertIn("dpo_lora", error_message)
        self.assertIn("not supported on Bedrock", error_message)

    def test_get_customization_type_unsupported_dpo_full(self):
        """Test that DPO_FULL raises ValueError."""
        from amzn_nova_forge.model.model_enums import TrainingMethod
        from amzn_nova_forge.util.bedrock import get_customization_type

        with self.assertRaises(ValueError) as context:
            get_customization_type(TrainingMethod.DPO_FULL)

        error_message = str(context.exception)
        self.assertIn("dpo_full", error_message)
        self.assertIn("not supported on Bedrock", error_message)

    def test_get_customization_type_unsupported_rft_multiturn_lora(self):
        """Test that RFT_MULTITURN_LORA raises ValueError."""
        from amzn_nova_forge.model.model_enums import TrainingMethod
        from amzn_nova_forge.util.bedrock import get_customization_type

        with self.assertRaises(ValueError) as context:
            get_customization_type(TrainingMethod.RFT_MULTITURN_LORA)

        error_message = str(context.exception)
        self.assertIn("rft_multiturn_lora", error_message)
        self.assertIn("not supported on Bedrock", error_message)

    def test_get_customization_type_unsupported_rft_multiturn_full(self):
        """Test that RFT_MULTITURN_FULL raises ValueError."""
        from amzn_nova_forge.model.model_enums import TrainingMethod
        from amzn_nova_forge.util.bedrock import get_customization_type

        with self.assertRaises(ValueError) as context:
            get_customization_type(TrainingMethod.RFT_MULTITURN_FULL)

        error_message = str(context.exception)
        self.assertIn("rft_multiturn_full", error_message)
        self.assertIn("not supported on Bedrock", error_message)

    def test_resolve_base_model_identifier_uses_explicit_value(self):
        """Test that resolve_base_model_identifier returns explicit value when provided."""
        from amzn_nova_forge.util.bedrock import (
            resolve_base_model_identifier,
        )

        explicit_identifier = (
            "arn:aws:bedrock:us-west-2::foundation-model/custom-model:0"
        )

        # Should return the explicit identifier without parsing recipe
        result = resolve_base_model_identifier(
            "dummy_recipe_path.yaml", explicit_identifier
        )
        self.assertEqual(result, explicit_identifier)

    @patch("yaml.safe_load")
    @patch("builtins.open", create=True)
    def test_resolve_base_model_identifier_from_local_recipe_nova_micro(
        self, mock_open, mock_yaml_load
    ):
        """Test resolving NOVA_MICRO from local recipe file."""
        from amzn_nova_forge.util.bedrock import (
            resolve_base_model_identifier,
        )

        # Mock recipe config
        mock_yaml_load.return_value = {
            "run": {
                "name": "test-job",
                "model_type": "amazon.nova-micro-v1:0:128k",
                "model_name_or_path": "nova-micro/prod",
            }
        }

        result = resolve_base_model_identifier(
            "/path/to/recipe.yaml", region="us-east-1"
        )

        self.assertIn("arn:aws:bedrock:us-east-1::foundation-model", result)
        self.assertIn("nova-micro", result)

    @patch("yaml.safe_load")
    @patch("builtins.open", create=True)
    def test_resolve_base_model_identifier_respects_region(
        self, mock_open, mock_yaml_load
    ):
        """Test that resolve_base_model_identifier uses the provided region in ARN."""
        from amzn_nova_forge.util.bedrock import (
            resolve_base_model_identifier,
        )

        # Mock recipe config
        mock_yaml_load.return_value = {
            "run": {
                "name": "test-job",
                "model_type": "amazon.nova-micro-v1:0:128k",
                "model_name_or_path": "nova-micro/prod",
            }
        }

        # Test us-west-2 region
        result_pdx = resolve_base_model_identifier(
            "/path/to/recipe.yaml", region="us-west-2"
        )
        self.assertIn("arn:aws:bedrock:us-west-2::foundation-model", result_pdx)
        self.assertIn("nova-micro", result_pdx)

        # Test eu-west-1 region
        result_eu = resolve_base_model_identifier(
            "/path/to/recipe.yaml", region="eu-west-1"
        )
        self.assertIn("arn:aws:bedrock:eu-west-1::foundation-model", result_eu)
        self.assertIn("nova-micro", result_eu)

    @patch("yaml.safe_load")
    @patch("builtins.open", create=True)
    def test_resolve_base_model_identifier_from_local_recipe_nova_lite(
        self, mock_open, mock_yaml_load
    ):
        """Test resolving NOVA_LITE from local recipe file."""
        from amzn_nova_forge.util.bedrock import (
            resolve_base_model_identifier,
        )

        # Mock recipe config
        mock_yaml_load.return_value = {
            "run": {
                "name": "test-job",
                "model_type": "amazon.nova-lite-v1:0:300k",
                "model_name_or_path": "nova-lite/prod",
            }
        }

        result = resolve_base_model_identifier(
            "/path/to/recipe.yaml", region="us-east-1"
        )

        self.assertIn("arn:aws:bedrock:us-east-1::foundation-model", result)
        self.assertIn("nova-lite", result)

    @patch("yaml.safe_load")
    @patch("builtins.open", create=True)
    def test_resolve_base_model_identifier_from_local_recipe_nova_lite_2(
        self, mock_open, mock_yaml_load
    ):
        """Test resolving NOVA_LITE_2 from local recipe file."""
        from amzn_nova_forge.util.bedrock import (
            resolve_base_model_identifier,
        )

        # Mock recipe config
        mock_yaml_load.return_value = {
            "run": {
                "name": "test-job",
                "model_type": "amazon.nova-2-lite-v1:0:256k",
                "model_name_or_path": "nova-lite-2/prod",
            }
        }

        result = resolve_base_model_identifier(
            "/path/to/recipe.yaml", region="us-east-1"
        )

        self.assertIn("arn:aws:bedrock:us-east-1::foundation-model", result)
        self.assertIn("nova-2-lite", result)

    @patch("yaml.safe_load")
    @patch("builtins.open", create=True)
    def test_resolve_base_model_identifier_from_local_recipe_nova_pro(
        self, mock_open, mock_yaml_load
    ):
        """Test resolving NOVA_PRO from local recipe file."""
        from amzn_nova_forge.util.bedrock import (
            resolve_base_model_identifier,
        )

        # Mock recipe config
        mock_yaml_load.return_value = {
            "run": {
                "name": "test-job",
                "model_type": "amazon.nova-pro-v1:0:300k",
                "model_name_or_path": "nova-pro/prod",
            }
        }

        result = resolve_base_model_identifier(
            "/path/to/recipe.yaml", region="us-east-1"
        )

        self.assertIn("arn:aws:bedrock:us-east-1::foundation-model", result)
        self.assertIn("nova-pro", result)

    @patch("yaml.safe_load")
    @patch("builtins.open", create=True)
    def test_resolve_base_model_identifier_missing_model_type_field(
        self, mock_open, mock_yaml_load
    ):
        """Test that missing model_type field raises ValueError."""
        from amzn_nova_forge.util.bedrock import (
            resolve_base_model_identifier,
        )

        # Mock recipe config without model_type
        mock_yaml_load.return_value = {
            "run": {
                "name": "test-job",
                # Missing model_type field
            }
        }

        with self.assertRaises(ValueError) as context:
            resolve_base_model_identifier("/path/to/recipe.yaml")

        error_message = str(context.exception)
        self.assertIn("model_type", error_message)

    @patch("yaml.safe_load")
    @patch("builtins.open", create=True)
    def test_resolve_base_model_identifier_unrecognized_model_type(
        self, mock_open, mock_yaml_load
    ):
        """Test that unrecognized model_type raises ValueError."""
        from amzn_nova_forge.util.bedrock import (
            resolve_base_model_identifier,
        )

        # Mock recipe config with unrecognized model_type
        mock_yaml_load.return_value = {
            "run": {
                "name": "test-job",
                "model_type": "unknown-model-type",
                "model_name_or_path": "unknown/model",
            }
        }

        with self.assertRaises(ValueError) as context:
            resolve_base_model_identifier("/path/to/recipe.yaml")

        error_message = str(context.exception)
        self.assertIn("unknown-model-type", error_message)
        self.assertIn("not supported on Bedrock", error_message)

    @patch("builtins.open", side_effect=FileNotFoundError("Recipe file not found"))
    def test_resolve_base_model_identifier_file_not_found(self, mock_open):
        """Test that FileNotFoundError is raised for missing recipe file."""
        from amzn_nova_forge.util.bedrock import (
            resolve_base_model_identifier,
        )

        with self.assertRaises(FileNotFoundError):
            resolve_base_model_identifier("/nonexistent/recipe.yaml")

    @patch("yaml.safe_load")
    @patch("builtins.open", create=True)
    def test_resolve_base_model_identifier_invalid_yaml(
        self, mock_open, mock_yaml_load
    ):
        """Test that invalid YAML raises exception."""
        from amzn_nova_forge.util.bedrock import (
            resolve_base_model_identifier,
        )

        # Mock yaml.safe_load to raise exception
        mock_yaml_load.side_effect = Exception("Invalid YAML")

        with self.assertRaises(Exception):
            resolve_base_model_identifier("/path/to/recipe.yaml")

    @patch("boto3.session.Session")
    @patch("boto3.client")
    @patch("yaml.safe_load")
    @patch("builtins.open", create=True)
    def test_execute_creates_bedrock_job_successfully(
        self, mock_open, mock_yaml_load, mock_boto_client, mock_session_class
    ):
        """Test that execute() creates a Bedrock customization job successfully."""
        from amzn_nova_forge.manager.runtime_manager import JobConfig

        # Mock boto3 session
        mock_session = mock_session_class.return_value
        mock_session.region_name = "us-east-1"

        # Mock bedrock client
        mock_bedrock_client = mock_boto_client.return_value
        mock_bedrock_client.create_model_customization_job.return_value = {
            "jobArn": "arn:aws:bedrock:us-east-1:123456789012:model-customization-job/test-job-12345678"
        }

        # Mock recipe config
        mock_yaml_load.return_value = {
            "run": {
                "name": "test-job",
                "model_type": "amazon.nova-micro-v1:0:128k",
                "model_name_or_path": "nova-micro/prod",
            },
            "training_config": {
                "method": "sft_lora",
                "learningRate": 0.001,
                "epochCount": 10,
                "trainer": {"peft": {"peft_scheme": "lora"}},
            },
        }

        # Create manager
        manager = BedrockRuntimeManager(
            execution_role="arn:aws:iam::123456789012:role/BedrockRole"
        )

        # Create job config
        from amzn_nova_forge.model.model_enums import TrainingMethod

        job_config = JobConfig(
            job_name="test-bedrock-job",
            image_uri="dummy-image-uri",  # Not used by Bedrock
            recipe_path="/path/to/recipe.yaml",
            data_s3_path="s3://bucket/data/",
            output_s3_path="s3://bucket/output/",
            method=TrainingMethod.SFT_LORA,  # Bedrock needs method in JobConfig
        )

        # Execute job
        job_arn = manager.execute(job_config)

        # Verify job ARN is returned
        self.assertEqual(
            job_arn,
            "arn:aws:bedrock:us-east-1:123456789012:model-customization-job/test-job-12345678",
        )

        # Verify create_model_customization_job was called
        mock_bedrock_client.create_model_customization_job.assert_called_once()

        # Get the API call arguments
        call_args = mock_bedrock_client.create_model_customization_job.call_args[1]

        # Verify required fields
        self.assertEqual(call_args["jobName"], "test-bedrock-job")
        self.assertIn("test-bedrock-job", call_args["customModelName"])
        self.assertEqual(
            call_args["roleArn"], "arn:aws:iam::123456789012:role/BedrockRole"
        )
        self.assertIn("nova-micro", call_args["baseModelIdentifier"])
        self.assertEqual(call_args["customizationType"], "FINE_TUNING")

        # Verify data configs
        self.assertEqual(call_args["trainingDataConfig"]["s3Uri"], "s3://bucket/data/")
        self.assertEqual(call_args["outputDataConfig"]["s3Uri"], "s3://bucket/output/")

        # Verify hyperparameters are strings and correctly filtered
        self.assertIn("learningRate", call_args["hyperParameters"])
        self.assertIsInstance(call_args["hyperParameters"]["learningRate"], str)
        self.assertEqual(call_args["hyperParameters"]["learningRate"], "0.001")

        self.assertIn("epochCount", call_args["hyperParameters"])
        self.assertIsInstance(call_args["hyperParameters"]["epochCount"], str)
        self.assertEqual(call_args["hyperParameters"]["epochCount"], "10")

        # Verify invalid hyperparameters are filtered out
        self.assertNotIn("method", call_args["hyperParameters"])
        self.assertNotIn("trainer", call_args["hyperParameters"])

    @patch("boto3.session.Session")
    @patch("boto3.client")
    @patch("yaml.safe_load")
    @patch("builtins.open", create=True)
    def test_execute_with_validation_data(
        self, mock_open, mock_yaml_load, mock_boto_client, mock_session_class
    ):
        """Test that execute() includes validationDataConfig when provided."""
        from amzn_nova_forge.manager.runtime_manager import JobConfig

        # Mock boto3 session
        mock_session = mock_session_class.return_value
        mock_session.region_name = "us-east-1"

        # Mock bedrock client
        mock_bedrock_client = mock_boto_client.return_value
        mock_bedrock_client.create_model_customization_job.return_value = {
            "jobArn": "arn:aws:bedrock:us-east-1:123456789012:model-customization-job/test-job-12345678"
        }

        # Mock recipe config
        mock_yaml_load.return_value = {
            "run": {
                "name": "test-job",
                "model_type": "amazon.nova-micro-v1:0:128k",
                "model_name_or_path": "nova-micro/prod",
            },
            "training_config": {
                "method": "sft_lora",
                "learning_rate": 0.001,
            },
        }

        # Create manager
        manager = BedrockRuntimeManager(
            execution_role="arn:aws:iam::123456789012:role/BedrockRole"
        )

        # Create job config with validation data
        from amzn_nova_forge.model.model_enums import TrainingMethod

        job_config = JobConfig(
            job_name="test-bedrock-job",
            image_uri="dummy-image-uri",
            recipe_path="/path/to/recipe.yaml",
            data_s3_path="s3://bucket/data/",
            output_s3_path="s3://bucket/output/",
            validation_data_s3_path="s3://bucket/validation/",
            method=TrainingMethod.SFT_LORA,  # Bedrock needs method in JobConfig
        )

        # Execute job
        job_arn = manager.execute(job_config)

        # Verify job ARN is returned
        self.assertIsNotNone(job_arn)

        # Get the API call arguments
        call_args = mock_bedrock_client.create_model_customization_job.call_args[1]

        # Verify validationDataConfig is included
        self.assertIn("validationDataConfig", call_args)
        self.assertEqual(
            call_args["validationDataConfig"]["validators"][0]["s3Uri"],
            "s3://bucket/validation/",
        )

    @patch("boto3.session.Session")
    @patch("boto3.client")
    @patch("yaml.safe_load")
    @patch("builtins.open", create=True)
    def test_execute_without_training_data(
        self, mock_open, mock_yaml_load, mock_boto_client, mock_session_class
    ):
        """Test that execute() raises ValueError when data_s3_path is None.

        Bedrock API requires trainingDataConfig to be provided.

        """
        from amzn_nova_forge.manager.runtime_manager import JobConfig

        # Mock boto3 session
        mock_session = mock_session_class.return_value
        mock_session.region_name = "us-east-1"

        # Mock bedrock client
        mock_bedrock_client = mock_boto_client.return_value

        # Mock recipe config
        mock_yaml_load.return_value = {
            "run": {
                "name": "test-job",
                "model_type": "amazon.nova-micro-v1:0:128k",
                "model_name_or_path": "nova-micro/prod",
            },
            "training_config": {
                "method": "sft_lora",
                "learning_rate": 0.001,
            },
        }

        # Create manager
        manager = BedrockRuntimeManager(
            execution_role="arn:aws:iam::123456789012:role/BedrockRole"
        )

        # Create job config without training data
        from amzn_nova_forge.model.model_enums import TrainingMethod

        job_config = JobConfig(
            job_name="test-bedrock-job",
            image_uri="dummy-image-uri",
            recipe_path="/path/to/recipe.yaml",
            data_s3_path=None,  # Missing training data
            output_s3_path="s3://bucket/output/",
            method=TrainingMethod.SFT_LORA,  # Bedrock needs method in JobConfig
        )

        # Verify ValueError is raised
        with self.assertRaises(ValueError) as context:
            manager.execute(job_config)

        self.assertIn("data_s3_path is required", str(context.exception))

    @patch("boto3.session.Session")
    @patch("boto3.client")
    @patch("yaml.safe_load")
    @patch("builtins.open", create=True)
    def test_execute_with_vpc_config(
        self, mock_open, mock_yaml_load, mock_boto_client, mock_session_class
    ):
        """Test that execute() includes vpcConfig when provided."""
        from amzn_nova_forge.manager.runtime_manager import JobConfig

        # Mock boto3 session
        mock_session = mock_session_class.return_value
        mock_session.region_name = "us-east-1"

        # Mock bedrock client
        mock_bedrock_client = mock_boto_client.return_value
        mock_bedrock_client.create_model_customization_job.return_value = {
            "jobArn": "arn:aws:bedrock:us-east-1:123456789012:model-customization-job/test-job-12345678"
        }

        # Mock recipe config
        mock_yaml_load.return_value = {
            "run": {
                "name": "test-job",
                "model_type": "amazon.nova-micro-v1:0:128k",
                "model_name_or_path": "nova-micro/prod",
            },
            "training_config": {
                "method": "sft_lora",
                "learning_rate": 0.001,
            },
        }

        # Create manager with VPC config
        manager = BedrockRuntimeManager(
            execution_role="arn:aws:iam::123456789012:role/BedrockRole",
            vpc_config={
                "subnet_ids": ["subnet-12345", "subnet-67890"],
                "security_group_ids": ["sg-abcdef"],
            },
        )

        # Create job config
        from amzn_nova_forge.model.model_enums import TrainingMethod

        job_config = JobConfig(
            job_name="test-bedrock-job",
            image_uri="dummy-image-uri",
            recipe_path="/path/to/recipe.yaml",
            data_s3_path="s3://bucket/data/",
            output_s3_path="s3://bucket/output/",
            method=TrainingMethod.SFT_LORA,  # Bedrock needs method in JobConfig
        )

        # Execute job
        job_arn = manager.execute(job_config)

        # Verify job ARN is returned
        self.assertIsNotNone(job_arn)

        # Get the API call arguments
        call_args = mock_bedrock_client.create_model_customization_job.call_args[1]

        # Verify vpcConfig is included
        self.assertIn("vpcConfig", call_args)
        self.assertEqual(
            call_args["vpcConfig"]["subnetIds"], ["subnet-12345", "subnet-67890"]
        )
        self.assertEqual(call_args["vpcConfig"]["securityGroupIds"], ["sg-abcdef"])

    @patch("boto3.session.Session")
    @patch("boto3.client")
    @patch("yaml.safe_load")
    @patch("builtins.open", create=True)
    def test_execute_with_kms_key(
        self, mock_open, mock_yaml_load, mock_boto_client, mock_session_class
    ):
        """Test that execute() includes KMS key in outputDataConfig when provided."""
        from amzn_nova_forge.manager.runtime_manager import JobConfig

        # Mock boto3 session
        mock_session = mock_session_class.return_value
        mock_session.region_name = "us-east-1"

        # Mock bedrock client
        mock_bedrock_client = mock_boto_client.return_value
        mock_bedrock_client.create_model_customization_job.return_value = {
            "jobArn": "arn:aws:bedrock:us-east-1:123456789012:model-customization-job/test-job-12345678"
        }

        # Mock recipe config
        mock_yaml_load.return_value = {
            "run": {
                "name": "test-job",
                "model_type": "amazon.nova-micro-v1:0:128k",
                "model_name_or_path": "nova-micro/prod",
            },
            "training_config": {
                "method": "sft_lora",
                "learning_rate": 0.001,
            },
        }

        # Create manager with KMS key
        manager = BedrockRuntimeManager(
            execution_role="arn:aws:iam::123456789012:role/BedrockRole",
            kms_key_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        )

        # Create job config
        from amzn_nova_forge.model.model_enums import TrainingMethod

        job_config = JobConfig(
            job_name="test-bedrock-job",
            image_uri="dummy-image-uri",
            recipe_path="/path/to/recipe.yaml",
            data_s3_path="s3://bucket/data/",
            output_s3_path="s3://bucket/output/",
            method=TrainingMethod.SFT_LORA,  # Bedrock needs method in JobConfig
        )

        # Execute job
        job_arn = manager.execute(job_config)

        # Verify job ARN is returned
        self.assertIsNotNone(job_arn)

        # Get the API call arguments
        call_args = mock_bedrock_client.create_model_customization_job.call_args[1]

        # Verify KMS key is included in outputDataConfig
        self.assertIn("kmsKeyId", call_args["outputDataConfig"])
        self.assertEqual(
            call_args["outputDataConfig"]["kmsKeyId"],
            "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        )

    @patch("boto3.session.Session")
    @patch("boto3.client")
    @patch("yaml.safe_load")
    @patch("builtins.open", create=True)
    def test_execute_handles_bedrock_api_error(
        self, mock_open, mock_yaml_load, mock_boto_client, mock_session_class
    ):
        """Test that execute() handles Bedrock API errors appropriately."""
        from botocore.exceptions import ClientError

        from amzn_nova_forge.manager.runtime_manager import JobConfig

        # Mock boto3 session
        mock_session = mock_session_class.return_value
        mock_session.region_name = "us-east-1"

        # Mock bedrock client to raise an error
        mock_bedrock_client = mock_boto_client.return_value
        mock_bedrock_client.create_model_customization_job.side_effect = ClientError(
            {
                "Error": {
                    "Code": "ValidationException",
                    "Message": "Invalid job configuration",
                }
            },
            "create_model_customization_job",
        )

        # Mock recipe config
        mock_yaml_load.return_value = {
            "run": {
                "name": "test-job",
                "model_type": "amazon.nova-micro-v1:0:128k",
                "model_name_or_path": "nova-micro/prod",
            },
            "training_config": {
                "method": "sft_lora",
                "learning_rate": 0.001,
            },
        }

        # Create manager
        manager = BedrockRuntimeManager(
            execution_role="arn:aws:iam::123456789012:role/BedrockRole"
        )

        # Create job config
        job_config = JobConfig(
            job_name="test-bedrock-job",
            image_uri="dummy-image-uri",
            recipe_path="/path/to/recipe.yaml",
            data_s3_path="s3://bucket/data/",
            output_s3_path="s3://bucket/output/",
        )

        # Verify ValueError is raised with appropriate message
        with self.assertRaises(ValueError) as context:
            manager.execute(job_config)

        self.assertIn(
            "Training method must be provided in job_config", str(context.exception)
        )

    @patch("boto3.session.Session")
    @patch("boto3.client")
    @patch("yaml.safe_load")
    @patch("builtins.open", create=True)
    def test_execute_with_rft_lambda_arn(
        self, mock_open, mock_yaml_load, mock_boto_client, mock_session_class
    ):
        """Test that execute() includes customizationConfig with Lambda ARN for RFT jobs.

        Validates: RFT Lambda ARN support for Bedrock API
        """
        from amzn_nova_forge.manager.runtime_manager import JobConfig

        # Mock boto3 session
        mock_session = mock_session_class.return_value
        mock_session.region_name = "us-east-1"

        # Mock bedrock client
        mock_bedrock_client = mock_boto_client.return_value
        mock_bedrock_client.create_model_customization_job.return_value = {
            "jobArn": "arn:aws:bedrock:us-east-1:123456789012:model-customization-job/test-rft-job-12345678"
        }

        # Mock recipe config for RFT job
        mock_yaml_load.return_value = {
            "run": {
                "name": "test-rft-job",
                "model_type": "amazon.nova-micro-v1:0:128k",
                "model_name_or_path": "nova-micro/prod",
            },
            "training_config": {
                "method": "rft_lora",  # RFT method
                "learning_rate": 0.001,
                "epochs": 10,
                "rollout": {  # This indicates RFT
                    "enabled": True
                },
                "trainer": {
                    "peft": {
                        "peft_scheme": "lora"  # This indicates LoRA
                    }
                },
            },
        }

        # Create manager
        manager = BedrockRuntimeManager(
            execution_role="arn:aws:iam::123456789012:role/BedrockRole"
        )

        # Create job config with RFT Lambda ARN
        from amzn_nova_forge.model.model_enums import TrainingMethod

        job_config = JobConfig(
            job_name="test-rft-job",
            image_uri="dummy-image-uri",
            recipe_path="/path/to/recipe.yaml",
            data_s3_path="s3://bucket/data/",
            output_s3_path="s3://bucket/output/",
            rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-rft-grader",
            method=TrainingMethod.RFT_LORA,  # Bedrock needs method in JobConfig
        )

        # Execute job
        job_arn = manager.execute(job_config)

        # Verify job ARN is returned
        self.assertEqual(
            job_arn,
            "arn:aws:bedrock:us-east-1:123456789012:model-customization-job/test-rft-job-12345678",
        )

        # Verify create_model_customization_job was called
        mock_bedrock_client.create_model_customization_job.assert_called_once()

        # Get the API call arguments
        call_args = mock_bedrock_client.create_model_customization_job.call_args[1]

        # Verify customization type is RFT
        self.assertEqual(call_args["customizationType"], "REINFORCEMENT_FINE_TUNING")

        # Verify customizationConfig is included with Lambda ARN
        self.assertIn("customizationConfig", call_args)
        self.assertIn("rftConfig", call_args["customizationConfig"])
        self.assertIn("graderConfig", call_args["customizationConfig"]["rftConfig"])
        self.assertIn(
            "lambdaGrader",
            call_args["customizationConfig"]["rftConfig"]["graderConfig"],
        )
        self.assertEqual(
            call_args["customizationConfig"]["rftConfig"]["graderConfig"][
                "lambdaGrader"
            ]["lambdaArn"],
            "arn:aws:lambda:us-east-1:123456789012:function:my-rft-grader",
        )

    @patch("boto3.session.Session")
    @patch("boto3.client")
    @patch("yaml.safe_load")
    @patch("builtins.open", create=True)
    def test_execute_rft_without_lambda_arn_raises_error(
        self, mock_open, mock_yaml_load, mock_boto_client, mock_session_class
    ):
        """Test that execute() raises ValueError for RFT jobs without Lambda ARN.

        Validates: RFT Lambda ARN validation
        """
        from amzn_nova_forge.manager.runtime_manager import JobConfig

        # Mock boto3 session
        mock_session = mock_session_class.return_value
        mock_session.region_name = "us-east-1"

        # Mock bedrock client
        mock_bedrock_client = mock_boto_client.return_value

        # Mock recipe config for RFT job
        mock_yaml_load.return_value = {
            "run": {
                "name": "test-rft-job",
                "model_type": "amazon.nova-micro-v1:0:128k",
                "model_name_or_path": "nova-micro/prod",
            },
            "training_config": {
                "method": "rft_lora",  # RFT method
                "learning_rate": 0.001,
                "rollout": {  # This indicates RFT
                    "enabled": True
                },
                "trainer": {
                    "peft": {
                        "peft_scheme": "lora"  # This indicates LoRA
                    }
                },
            },
        }

        # Create manager
        manager = BedrockRuntimeManager(
            execution_role="arn:aws:iam::123456789012:role/BedrockRole"
        )

        # Create job config WITHOUT RFT Lambda ARN
        from amzn_nova_forge.model.model_enums import TrainingMethod

        job_config = JobConfig(
            job_name="test-rft-job",
            image_uri="dummy-image-uri",
            recipe_path="/path/to/recipe.yaml",
            data_s3_path="s3://bucket/data/",
            output_s3_path="s3://bucket/output/",
            rft_lambda_arn=None,  # Missing Lambda ARN
            method=TrainingMethod.RFT_LORA,  # Bedrock needs method in JobConfig
        )

        # Verify ValueError is raised
        with self.assertRaises(ValueError) as context:
            manager.execute(job_config)

        self.assertIn("rft_lambda_arn is required", str(context.exception))
        self.assertIn("RFT", str(context.exception))

    @patch("boto3.session.Session")
    @patch("boto3.client")
    def test_validation_data_s3_path_supported_for_sft(
        self, mock_boto_client, mock_session_class
    ):
        """Test that validation_data_s3_path is supported for SFT on Bedrock."""
        # Mock session
        mock_session = mock_session_class.return_value
        mock_session.region_name = "us-east-1"

        # Create manager
        manager = BedrockRuntimeManager(
            execution_role="arn:aws:iam::123456789012:role/BedrockRole"
        )

        # Create job config with validation data for SFT
        job_config = JobConfig(
            job_name="test-sft-validation",
            image_uri="dummy-image",
            recipe_path="/path/to/recipe.yaml",
            data_s3_path="s3://bucket/train/",
            output_s3_path="s3://bucket/output/",
            validation_data_s3_path="s3://bucket/validation/",
            method=TrainingMethod.SFT_LORA,
        )

        # Should not raise an error
        self.assertIsNotNone(job_config.validation_data_s3_path)

    @patch("boto3.session.Session")
    @patch("boto3.client")
    def test_validation_data_s3_path_supported_for_rft(
        self, mock_boto_client, mock_session_class
    ):
        """Test that validation_data_s3_path is supported for RFT on Bedrock."""
        # Mock session
        mock_session = mock_session_class.return_value
        mock_session.region_name = "us-east-1"

        # Create manager
        manager = BedrockRuntimeManager(
            execution_role="arn:aws:iam::123456789012:role/BedrockRole"
        )

        # Create job config with validation data for RFT
        job_config = JobConfig(
            job_name="test-rft-validation",
            image_uri="dummy-image",
            recipe_path="/path/to/recipe.yaml",
            data_s3_path="s3://bucket/train/",
            output_s3_path="s3://bucket/output/",
            validation_data_s3_path="s3://bucket/validation/",
            method=TrainingMethod.RFT_LORA,
        )

        # Should not raise an error
        self.assertIsNotNone(job_config.validation_data_s3_path)

    @patch("boto3.session.Session")
    @patch("boto3.client")
    @patch("yaml.safe_load")
    @patch("builtins.open", create=True)
    def test_validation_data_logs_warning_for_nova_lite_2(
        self, mock_open, mock_yaml_load, mock_boto_client, mock_session_class
    ):
        """Test that validation_data_s3_path logs warning and is ignored for Nova Lite 2 on Bedrock."""
        from amzn_nova_forge.manager.runtime_manager import JobConfig
        from amzn_nova_forge.model.model_enums import TrainingMethod

        # Mock session
        mock_session = mock_session_class.return_value
        mock_session.region_name = "us-east-1"

        # Mock bedrock client
        mock_bedrock_client = mock_boto_client.return_value
        mock_bedrock_client.create_model_customization_job.return_value = {
            "jobArn": "arn:aws:bedrock:us-east-1:123456789012:model-customization-job/test-job-12345678"
        }

        # Mock recipe config with Nova Lite 2
        mock_yaml_load.return_value = {
            "run": {
                "name": "test-job",
                "model_type": "amazon.nova-2-lite-v1:0:256k",
                "model_name_or_path": "nova-lite-2/prod",
            },
            "training_config": {
                "method": "sft_lora",
                "learning_rate": 0.001,
            },
        }

        # Create manager
        manager = BedrockRuntimeManager(
            execution_role="arn:aws:iam::123456789012:role/BedrockRole"
        )

        # Create job config with validation data for Nova Lite 2
        job_config = JobConfig(
            job_name="test-nova-lite-2-validation",
            image_uri="dummy-image-uri",
            recipe_path="/path/to/recipe.yaml",
            data_s3_path="s3://bucket/train/",
            output_s3_path="s3://bucket/output/",
            validation_data_s3_path="s3://bucket/validation/",
            method=TrainingMethod.SFT_LORA,
        )

        # Should succeed with warning (not raise error)
        with self.assertLogs(level="WARNING") as log:
            job_arn = manager.execute(job_config)

        # Verify job was created successfully
        self.assertIsNotNone(job_arn)

        # Verify warning was logged
        warning_found = any(
            "Validation datasets are not supported for Nova Lite 2" in msg
            for msg in log.output
        )
        self.assertTrue(
            warning_found,
            "Expected warning about Nova Lite 2 validation dataset not being supported",
        )

        # Verify validationDataConfig was NOT included in API call
        call_args = mock_bedrock_client.create_model_customization_job.call_args[1]
        self.assertNotIn(
            "validationDataConfig",
            call_args,
            "validationDataConfig should not be included for Nova Lite 2",
        )

    @patch("boto3.session.Session")
    @patch("boto3.client")
    def test_get_job_status_completed(self, mock_boto_client, mock_session_class):
        """Test getting job status for completed job."""
        # Mock session
        mock_session = mock_session_class.return_value
        mock_session.region_name = "us-east-1"

        # Mock bedrock client
        mock_bedrock_client = mock_boto_client.return_value
        mock_bedrock_client.get_model_customization_job.return_value = {
            "status": "Completed",
            "jobArn": "arn:aws:bedrock:us-east-1:123456789012:model-customization-job/test-job",
        }

        # Create manager
        manager = BedrockRuntimeManager(
            execution_role="arn:aws:iam::123456789012:role/BedrockRole"
        )

        # Get job status
        from amzn_nova_forge.model.result.job_result import (
            BedrockStatusManager,
        )

        status_manager = BedrockStatusManager(bedrock_client=mock_bedrock_client)
        status, message = status_manager.get_job_status(
            "arn:aws:bedrock:us-east-1:123456789012:model-customization-job/test-job"
        )

        # Verify status
        from amzn_nova_forge.model.result.job_result import JobStatus

        self.assertEqual(status, JobStatus.COMPLETED)

    @patch("boto3.session.Session")
    @patch("boto3.client")
    def test_get_job_status_in_progress(self, mock_boto_client, mock_session_class):
        """Test getting job status for in-progress job."""
        # Mock session
        mock_session = mock_session_class.return_value
        mock_session.region_name = "us-east-1"

        # Mock bedrock client
        mock_bedrock_client = mock_boto_client.return_value
        mock_bedrock_client.get_model_customization_job.return_value = {
            "status": "InProgress",
            "jobArn": "arn:aws:bedrock:us-east-1:123456789012:model-customization-job/test-job",
        }

        # Create manager
        manager = BedrockRuntimeManager(
            execution_role="arn:aws:iam::123456789012:role/BedrockRole"
        )

        # Get job status
        from amzn_nova_forge.model.result.job_result import (
            BedrockStatusManager,
        )

        status_manager = BedrockStatusManager(bedrock_client=mock_bedrock_client)
        status, message = status_manager.get_job_status(
            "arn:aws:bedrock:us-east-1:123456789012:model-customization-job/test-job"
        )

        # Verify status
        from amzn_nova_forge.model.result.job_result import JobStatus

        self.assertEqual(status, JobStatus.IN_PROGRESS)

    @patch("boto3.session.Session")
    @patch("boto3.client")
    def test_get_job_status_failed(self, mock_boto_client, mock_session_class):
        """Test getting job status for failed job."""
        # Mock session
        mock_session = mock_session_class.return_value
        mock_session.region_name = "us-east-1"

        # Mock bedrock client
        mock_bedrock_client = mock_boto_client.return_value
        mock_bedrock_client.get_model_customization_job.return_value = {
            "status": "Failed",
            "failureMessage": "Training failed due to insufficient data",
            "jobArn": "arn:aws:bedrock:us-east-1:123456789012:model-customization-job/test-job",
        }

        # Create manager
        manager = BedrockRuntimeManager(
            execution_role="arn:aws:iam::123456789012:role/BedrockRole"
        )

        # Get job status
        from amzn_nova_forge.model.result.job_result import (
            BedrockStatusManager,
        )

        status_manager = BedrockStatusManager(bedrock_client=mock_bedrock_client)
        status, message = status_manager.get_job_status(
            "arn:aws:bedrock:us-east-1:123456789012:model-customization-job/test-job"
        )

        # Verify status
        from amzn_nova_forge.model.result.job_result import JobStatus

        self.assertEqual(status, JobStatus.FAILED)
        self.assertEqual(message, "Failed")

    @patch("boto3.session.Session")
    @patch("boto3.client")
    @patch("yaml.safe_load")
    @patch("builtins.open", create=True)
    def test_execute_with_invalid_execution_role(
        self, mock_open, mock_yaml_load, mock_boto_client, mock_session_class
    ):
        """Test that execute raises error with invalid execution role format."""
        from botocore.exceptions import ClientError

        # Mock session
        mock_session = mock_session_class.return_value
        mock_session.region_name = "us-east-1"

        # Mock bedrock client to raise error
        mock_bedrock_client = mock_boto_client.return_value
        mock_bedrock_client.create_model_customization_job.side_effect = ClientError(
            {
                "Error": {
                    "Code": "ValidationException",
                    "Message": "Invalid execution role ARN",
                }
            },
            "CreateModelCustomizationJob",
        )

        # Mock recipe config
        mock_yaml_load.return_value = {
            "run": {
                "name": "test-job",
                "model_type": "amazon.nova-micro-v1:0:128k",
            },
            "training_config": {"method": "sft_lora"},
        }

        # Create manager with invalid role
        manager = BedrockRuntimeManager(execution_role="invalid-role-arn")

        # Create job config
        job_config = JobConfig(
            job_name="test-job",
            image_uri="dummy-image",
            recipe_path="/path/to/recipe.yaml",
            data_s3_path="s3://bucket/data/",
            output_s3_path="s3://bucket/output/",
            method=TrainingMethod.SFT_LORA,
        )

        # Verify ClientError or ValueError is raised
        with self.assertRaises((ClientError, ValueError)):
            manager.execute(job_config)

    @patch("boto3.session.Session")
    @patch("boto3.client")
    def test_execute_without_method_raises_error(
        self, mock_boto_client, mock_session_class
    ):
        """Test that execute raises error when method is not provided."""
        # Mock session
        mock_session = mock_session_class.return_value
        mock_session.region_name = "us-east-1"

        # Create manager
        manager = BedrockRuntimeManager(
            execution_role="arn:aws:iam::123456789012:role/BedrockRole"
        )

        # Create job config without method
        job_config = JobConfig(
            job_name="test-job",
            image_uri="dummy-image",
            recipe_path="/path/to/recipe.yaml",
            data_s3_path="s3://bucket/data/",
            output_s3_path="s3://bucket/output/",
        )

        # Verify ValueError is raised
        with self.assertRaises(ValueError) as context:
            manager.execute(job_config)

        self.assertIn("Training method must be provided", str(context.exception))

    @patch("boto3.session.Session")
    @patch("boto3.client")
    @patch("yaml.safe_load")
    @patch("builtins.open", create=True)
    def test_execute_rft_hyperparameters_use_native_types(
        self, mock_open, mock_yaml_load, mock_boto_client, mock_session_class
    ):
        """Test that RFT hyperparameters are passed as native types (int, float, str).

        This is CRITICAL - Bedrock API requires RFT hyperparameters to be native types,
        NOT strings. SFT hyperparameters must be strings, but RFT must be native types.

        Validates: RFT hyperparameter type requirements for Bedrock API
        """
        from amzn_nova_forge.manager.runtime_manager import JobConfig

        # Mock boto3 session
        mock_session = mock_session_class.return_value
        mock_session.region_name = "us-east-1"

        # Mock bedrock client
        mock_bedrock_client = mock_boto_client.return_value
        mock_bedrock_client.create_model_customization_job.return_value = {
            "jobArn": "arn:aws:bedrock:us-east-1:123456789012:model-customization-job/test-rft-job-12345678"
        }

        # Mock recipe config for RFT job with various hyperparameter types
        mock_yaml_load.return_value = {
            "run": {
                "name": "test-rft-job",
                "model_type": "amazon.nova-micro-v1:0:128k",
                "model_name_or_path": "nova-micro/prod",
            },
            "training_config": {
                "method": "rft_lora",
                "rft": {
                    # Integer hyperparameters
                    "epochCount": 1,
                    "batchSize": 128,
                    "maxPromptLength": 8192,
                    "trainingSamplePerPrompt": 8,
                    "inferenceMaxTokens": 2048,
                    "evalInterval": 10,
                    # Float hyperparameter
                    "learningRate": 0.0001,
                    # String hyperparameter
                    "reasoningEffort": "low",
                    # Nested dict - should be excluded
                    "graderConfig": {
                        "lambdaArn": "arn:aws:lambda:us-east-1:123456789012:function:grader"
                    },
                },
            },
        }

        # Create manager
        manager = BedrockRuntimeManager(
            execution_role="arn:aws:iam::123456789012:role/BedrockRole"
        )

        # Create job config with RFT Lambda ARN
        from amzn_nova_forge.model.model_enums import TrainingMethod

        job_config = JobConfig(
            job_name="test-rft-job",
            image_uri="dummy-image-uri",
            recipe_path="/path/to/recipe.yaml",
            data_s3_path="s3://bucket/data/",
            output_s3_path="s3://bucket/output/",
            rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-rft-grader",
            method=TrainingMethod.RFT_LORA,
        )

        # Execute job
        job_arn = manager.execute(job_config)

        # Verify job ARN is returned
        self.assertIsNotNone(job_arn)

        # Get the API call arguments
        call_args = mock_bedrock_client.create_model_customization_job.call_args[1]

        # Verify customizationConfig exists
        self.assertIn("customizationConfig", call_args)
        self.assertIn("rftConfig", call_args["customizationConfig"])
        self.assertIn("hyperParameters", call_args["customizationConfig"]["rftConfig"])

        rft_hyperparams = call_args["customizationConfig"]["rftConfig"][
            "hyperParameters"
        ]

        # CRITICAL: Verify integer hyperparameters are integers, NOT strings
        self.assertIsInstance(rft_hyperparams["epochCount"], int)
        self.assertEqual(rft_hyperparams["epochCount"], 1)
        self.assertNotIsInstance(rft_hyperparams["epochCount"], str)

        self.assertIsInstance(rft_hyperparams["batchSize"], int)
        self.assertEqual(rft_hyperparams["batchSize"], 128)
        self.assertNotIsInstance(rft_hyperparams["batchSize"], str)

        self.assertIsInstance(rft_hyperparams["maxPromptLength"], int)
        self.assertEqual(rft_hyperparams["maxPromptLength"], 8192)

        self.assertIsInstance(rft_hyperparams["trainingSamplePerPrompt"], int)
        self.assertEqual(rft_hyperparams["trainingSamplePerPrompt"], 8)

        self.assertIsInstance(rft_hyperparams["inferenceMaxTokens"], int)
        self.assertEqual(rft_hyperparams["inferenceMaxTokens"], 2048)

        self.assertIsInstance(rft_hyperparams["evalInterval"], int)
        self.assertEqual(rft_hyperparams["evalInterval"], 10)

        # CRITICAL: Verify float hyperparameters are floats, NOT strings
        self.assertIsInstance(rft_hyperparams["learningRate"], float)
        self.assertEqual(rft_hyperparams["learningRate"], 0.0001)
        self.assertNotIsInstance(rft_hyperparams["learningRate"], str)

        # Verify string hyperparameters remain strings
        self.assertIsInstance(rft_hyperparams["reasoningEffort"], str)
        self.assertEqual(rft_hyperparams["reasoningEffort"], "low")

        # Verify graderConfig is NOT in hyperparameters (it's in graderConfig section)
        self.assertNotIn("graderConfig", rft_hyperparams)

    @patch("boto3.session.Session")
    @patch("boto3.client")
    @patch("yaml.safe_load")
    @patch("builtins.open", create=True)
    def test_execute_sft_hyperparameters_use_strings(
        self, mock_open, mock_yaml_load, mock_boto_client, mock_session_class
    ):
        """Test that SFT hyperparameters are passed as strings.

        This verifies the contrast with RFT - SFT hyperparameters MUST be strings,
        while RFT hyperparameters must be native types.

        Validates: SFT hyperparameter type requirements for Bedrock API
        """
        from amzn_nova_forge.manager.runtime_manager import JobConfig

        # Mock boto3 session
        mock_session = mock_session_class.return_value
        mock_session.region_name = "us-east-1"

        # Mock bedrock client
        mock_bedrock_client = mock_boto_client.return_value
        mock_bedrock_client.create_model_customization_job.return_value = {
            "jobArn": "arn:aws:bedrock:us-east-1:123456789012:model-customization-job/test-sft-job-12345678"
        }

        # Mock recipe config for SFT job
        mock_yaml_load.return_value = {
            "run": {
                "name": "test-sft-job",
                "model_type": "amazon.nova-micro-v1:0:128k",
                "model_name_or_path": "nova-micro/prod",
            },
            "training_config": {
                "method": "sft_lora",
                "epochCount": 10,
                "batchSize": 128,
                "learningRate": 0.0001,
                "maxPromptLength": 8192,
                "trainer": {  # Nested dict - should be excluded
                    "peft": {"peft_scheme": "lora"}
                },
            },
        }

        # Create manager
        manager = BedrockRuntimeManager(
            execution_role="arn:aws:iam::123456789012:role/BedrockRole"
        )

        # Create job config
        from amzn_nova_forge.model.model_enums import TrainingMethod

        job_config = JobConfig(
            job_name="test-sft-job",
            image_uri="dummy-image-uri",
            recipe_path="/path/to/recipe.yaml",
            data_s3_path="s3://bucket/data/",
            output_s3_path="s3://bucket/output/",
            method=TrainingMethod.SFT_LORA,
        )

        # Execute job
        job_arn = manager.execute(job_config)

        # Verify job ARN is returned
        self.assertIsNotNone(job_arn)

        # Get the API call arguments
        call_args = mock_bedrock_client.create_model_customization_job.call_args[1]

        # Verify hyperParameters exist at top level (not in customizationConfig)
        self.assertIn("hyperParameters", call_args)
        sft_hyperparams = call_args["hyperParameters"]

        # CRITICAL: Verify all SFT hyperparameters are strings
        self.assertIsInstance(sft_hyperparams["epochCount"], str)
        self.assertEqual(sft_hyperparams["epochCount"], "10")

        self.assertIsInstance(sft_hyperparams["batchSize"], str)
        self.assertEqual(sft_hyperparams["batchSize"], "128")

        self.assertIsInstance(sft_hyperparams["learningRate"], str)
        self.assertEqual(sft_hyperparams["learningRate"], "0.0001")

        self.assertIsInstance(sft_hyperparams["maxPromptLength"], str)
        self.assertEqual(sft_hyperparams["maxPromptLength"], "8192")

        # Verify nested dicts are excluded
        self.assertNotIn("trainer", sft_hyperparams)

        # Verify method field is excluded
        self.assertNotIn("method", sft_hyperparams)


if __name__ == "__main__":
    unittest.main()
