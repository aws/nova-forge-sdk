import json
import unittest
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, Mock, mock_open, patch

from botocore.exceptions import ClientError

from amzn_nova_customization_sdk.model.model_enums import (
    Model,
    Platform,
    TrainingMethod,
)
from amzn_nova_customization_sdk.recipe.recipe_config import EvaluationTask
from amzn_nova_customization_sdk.util.recipe import (
    FileLoadError,
    RecipePath,
    _get_hub_content_name,
    _parse_s3_uri,
    _validate_extension,
    download_templates_from_local,
    download_templates_from_s3,
    get_hub_recipe_metadata,
    load_file_content,
)


# Dummy enums and dataclasses for testing
class DummyEnum(Enum):
    A = "a"
    B = "b"


@dataclass
class Nested:
    field2: int
    enum_field: DummyEnum


@dataclass
class Root:
    field1: str
    nested: Nested


class TestRecipePath(unittest.TestCase):
    def test_recipe_path_init_non_temp(self):
        rp = RecipePath("/some/path", root="/some", temp=False)
        self.assertEqual(rp.path, "/some/path")
        self.assertEqual(rp.root, "/some")
        self.assertFalse(rp.temp)

    def test_recipe_path_init_temp(self):
        rp = RecipePath("/tmp/path", root="/tmp", temp=True)
        self.assertEqual(rp.path, "/tmp/path")
        self.assertEqual(rp.root, "/tmp")
        self.assertTrue(rp.temp)
        self.assertIn("/tmp", RecipePath.roots)
        RecipePath.roots.remove("/tmp")

    @patch("amzn_nova_customization_sdk.util.recipe.shutil.rmtree")
    def test_recipe_path_close_temp(self, mock_rmtree):
        rp = RecipePath("/tmp/path", root="/tmp", temp=True)
        rp.close()
        mock_rmtree.assert_called_once_with("/tmp")

    @patch("amzn_nova_customization_sdk.util.recipe.shutil.rmtree")
    def test_recipe_path_close_non_temp(self, mock_rmtree):
        rp = RecipePath("/some/path", root="/some", temp=False)
        rp.close()
        mock_rmtree.assert_not_called()

    @patch("amzn_nova_customization_sdk.util.recipe.shutil.rmtree")
    def test_recipe_path_delete_temp_dir_error(self, mock_rmtree):
        mock_rmtree.side_effect = OSError("Permission denied")
        RecipePath.delete_temp_dir("/tmp/test")
        mock_rmtree.assert_called_once_with("/tmp/test")

    @patch("amzn_nova_customization_sdk.util.recipe.shutil.rmtree")
    def test_recipe_path_close_all(self, mock_rmtree):
        RecipePath.roots = ["/tmp/1", "/tmp/2", "/tmp/3"]
        rp = RecipePath("/tmp/path")
        rp.close_all()
        self.assertEqual(mock_rmtree.call_count, 3)


class TestParseS3Uri(unittest.TestCase):
    def test_parse_s3_uri_valid(self):
        uri = "s3://my-bucket/path/to/file.yaml"
        bucket, key = _parse_s3_uri(uri)
        self.assertEqual(bucket, "my-bucket")
        self.assertEqual(key, "path/to/file.yaml")

    def test_parse_s3_uri_invalid(self):
        self.assertIsNone(_parse_s3_uri("not-an-s3-uri"))


class TestValidateExtension(unittest.TestCase):
    def test_validate_extension_no_exception_raised(self):
        _validate_extension("file.yaml", ".yaml")

    def test_validate_extension_case_insensitive(self):
        _validate_extension("file.YAML", ".yaml")
        _validate_extension("file.yaml", ".YAML")

    def test_validate_extension_failure(self):
        with self.assertRaises(FileLoadError):
            _validate_extension("file.txt", ".yaml")

    def test_validate_extension_failure_message(self):
        with self.assertRaises(FileLoadError) as context:
            _validate_extension("file.txt", ".yaml")
        self.assertIn("must have .yaml extension", str(context.exception))


class TestLoadFileContent(unittest.TestCase):
    @patch("amzn_nova_customization_sdk.util.recipe.boto3.client")
    def test_load_file_content_s3(self, mock_boto_client):
        content = "key: value"
        mock_s3 = MagicMock()
        mock_s3.get_object.return_value = {"Body": BytesIO(content.encode("utf-8"))}
        mock_boto_client.return_value = mock_s3

        with patch(
            "amzn_nova_customization_sdk.util.recipe._parse_s3_uri",
            return_value=("bucket", "key.yaml"),
        ):
            result = load_file_content("s3://bucket/key.yaml", ".yaml")
            self.assertEqual(result, content)

    @patch("amzn_nova_customization_sdk.util.recipe.Path.read_text")
    def test_load_file_content_local(self, mock_read_text):
        mock_read_text.return_value = "field: value"
        with patch(
            "amzn_nova_customization_sdk.util.recipe._parse_s3_uri",
            return_value=None,
        ):
            result = load_file_content("/tmp/file.yaml")
            self.assertEqual(result, "field: value")

    @patch("amzn_nova_customization_sdk.util.recipe.boto3.client")
    @patch("amzn_nova_customization_sdk.util.recipe._parse_s3_uri")
    def test_load_file_content_s3_client_error(
        self, mock_parse_s3_uri, mock_boto_client
    ):
        mock_parse_s3_uri.return_value = ("bucket", "key.yaml")
        mock_s3 = MagicMock()

        mock_s3.get_object.side_effect = ClientError(
            error_response={"Error": {"Code": "NoSuchKey", "Message": "Not found"}},
            operation_name="GetObject",
        )
        mock_boto_client.return_value = mock_s3

        with self.assertRaises(FileLoadError) as context:
            load_file_content("s3://bucket/key.yaml")

        self.assertIn("Failed to load S3 file", str(context.exception))

    @patch("amzn_nova_customization_sdk.util.recipe._parse_s3_uri", return_value=None)
    def test_load_file_content_file_not_found(self, mock_parse_s3_uri):
        with patch.object(Path, "read_text", side_effect=FileNotFoundError):
            with self.assertRaises(FileLoadError) as context:
                load_file_content("local_file.yaml")

            self.assertIn("File not found", str(context.exception))

    @patch("amzn_nova_customization_sdk.util.recipe._parse_s3_uri", return_value=None)
    def test_load_file_content_os_error(self, mock_parse_s3_uri):
        with patch.object(Path, "read_text", side_effect=OSError("disk error")):
            with self.assertRaises(FileLoadError) as context:
                load_file_content("local_file.yaml")

            self.assertIn("Failed to read file", str(context.exception))

    def test_load_file_content_wrong_extension(self):
        with self.assertRaises(FileLoadError) as context:
            load_file_content("file.txt", ".yaml")
        self.assertIn("must have .yaml extension", str(context.exception))


class TestGetHubContentName(unittest.TestCase):
    def test_get_hub_content_name_nova_micro(self):
        result = _get_hub_content_name(Model.NOVA_MICRO)
        self.assertEqual(result, "nova-textgeneration-micro")

    def test_get_hub_content_name_nova_lite(self):
        result = _get_hub_content_name(Model.NOVA_LITE)
        self.assertEqual(result, "nova-textgeneration-lite")

    def test_get_hub_content_name_nova_lite_2(self):
        result = _get_hub_content_name(Model.NOVA_LITE_2)
        self.assertEqual(result, "nova-textgeneration-lite-v2")

    def test_get_hub_content_name_nova_pro(self):
        result = _get_hub_content_name(Model.NOVA_PRO)
        self.assertEqual(result, "nova-textgeneration-pro")

    def test_get_hub_content_name_unsupported_model(self):
        mock_model = Mock(spec=Model)
        mock_model.value = "unsupported"
        with self.assertRaises(ValueError) as context:
            _get_hub_content_name(mock_model)
        self.assertIn("Unsupported model", str(context.exception))


class TestGetHubRecipeMetadata(unittest.TestCase):
    @patch("amzn_nova_customization_sdk.util.recipe._get_hub_content")
    def test_get_hub_recipe_metadata_sft_lora_smtj(self, mock_get_hub_content):
        mock_get_hub_content.return_value = {
            "HubContentDocument": {
                "RecipeCollection": [
                    {
                        "DisplayName": "Nova Lite LoRA SFT on GPU",
                        "Name": "nova_lite_1_0_g5_g6_12x_gpu_lora_sft",
                        "RecipeFilePath": "recipes/fine-tuning/nova/nova_1_0/nova_lite/SFT/nova_lite_1_0_g5_g6_12x_gpu_lora_sft.yaml",
                        "CustomizationTechnique": "SFT",
                        "InstanceCount": 1,
                        "ServerlessMeteringType": "Token-based",
                        "Type": "FineTuning",
                        "Versions": ["1.0.0"],
                        "Hardware": "GPU",
                        "SupportedInstanceTypes": ["ml.g5.12xlarge", "ml.g6.12xlarge"],
                        "Peft": "LORA",
                        "SequenceLength": "8K",
                        "SmtjRecipeTemplateS3Uri": "s3://jumpstart-cache-prod-us-east-1/recipes/nova_lite_1_0_g5_g6_12x_gpu_lora_sft_payload_template_sm_jobs_v1.0.20.yaml",
                        "SmtjOverrideParamsS3Uri": "s3://jumpstart-cache-prod-us-east-1/recipes/nova_lite_1_0_g5_g6_12x_gpu_lora_sft_override_params_sm_jobs_v1.0.20.json",
                        "SmtjImageUri": "708977205387.dkr.ecr.us-east-1.amazonaws.com/nova-fine-tune-repo:SM-TJ-SFT-latest",
                    }
                ]
            }
        }

        result = get_hub_recipe_metadata(
            model=Model.NOVA_LITE,
            method=TrainingMethod.SFT_LORA,
            platform=Platform.SMTJ,
            instance_type="ml.g5.12xlarge",
            region="us-east-1",
        )

        self.assertEqual(result["DisplayName"], "Nova Lite LoRA SFT on GPU")
        self.assertEqual(result["CustomizationTechnique"], "SFT")
        self.assertEqual(result["Peft"], "LORA")
        self.assertEqual(result["SequenceLength"], "8K")
        self.assertIn("SmtjRecipeTemplateS3Uri", result)
        self.assertIn("SmtjOverrideParamsS3Uri", result)

    @patch("amzn_nova_customization_sdk.util.recipe._get_hub_content")
    def test_get_hub_recipe_metadata_sft_full_smtj(self, mock_get_hub_content):
        mock_get_hub_content.return_value = {
            "HubContentDocument": {
                "RecipeCollection": [
                    {
                        "DisplayName": "Nova Lite SFT on GPU",
                        "Name": "nova_lite_1_0_p5_p4d_gpu_sft",
                        "RecipeFilePath": "recipes/fine-tuning/nova/nova_1_0/nova_lite/SFT/nova_lite_1_0_p5_p4d_gpu_sft.yaml",
                        "CustomizationTechnique": "SFT",
                        "InstanceCount": 4,
                        "Type": "FineTuning",
                        "Hardware": "GPU",
                        "SupportedInstanceTypes": ["ml.p5.48xlarge"],
                        "SequenceLength": "32K",
                        "SmtjRecipeTemplateS3Uri": "s3://jumpstart-cache-prod-us-east-1/recipes/nova_lite_1_0_p5_p4d_gpu_sft_payload_template_sm_jobs_v1.0.20.yaml",
                        "SmtjOverrideParamsS3Uri": "s3://jumpstart-cache-prod-us-east-1/recipes/nova_lite_1_0_p5_p4d_gpu_sft_override_params_sm_jobs_v1.0.20.json",
                    }
                ]
            }
        }

        result = get_hub_recipe_metadata(
            model=Model.NOVA_LITE,
            method=TrainingMethod.SFT_FULL,
            platform=Platform.SMTJ,
            instance_type="ml.p5.48xlarge",
            region="us-east-1",
        )

        self.assertEqual(result["DisplayName"], "Nova Lite SFT on GPU")
        self.assertNotIn("Peft", result)
        self.assertEqual(result["SequenceLength"], "32K")

    @patch("amzn_nova_customization_sdk.util.recipe._get_hub_content")
    def test_get_hub_recipe_metadata_smhp_platform(self, mock_get_hub_content):
        mock_get_hub_content.return_value = {
            "HubContentDocument": {
                "RecipeCollection": [
                    {
                        "DisplayName": "Nova Lite LoRA SFT on GPU",
                        "Name": "nova_lite_1_0_p5_p4d_gpu_lora_sft",
                        "CustomizationTechnique": "SFT",
                        "InstanceCount": 4,
                        "Type": "FineTuning",
                        "Hardware": "GPU",
                        "SupportedInstanceTypes": ["ml.p5.48xlarge"],
                        "Peft": "LORA",
                        "SequenceLength": "32K",
                        "HpEksPayloadTemplateS3Uri": "s3://jumpstart-cache-prod-us-east-1/recipes/nova_lite_1_0_p5_p4d_gpu_lora_sft_payload_template_k8s_v1.0.20.yaml",
                        "HpEksOverrideParamsS3Uri": "s3://jumpstart-cache-prod-us-east-1/recipes/nova_lite_1_0_p5_p4d_gpu_lora_sft_override_params_k8s_v1.0.20.json",
                        "SmtjRecipeTemplateS3Uri": "s3://jumpstart-cache-prod-us-east-1/recipes/nova_lite_1_0_p5_p4d_gpu_lora_sft_payload_template_sm_jobs_v1.0.20.yaml",
                        "SmtjOverrideParamsS3Uri": "s3://jumpstart-cache-prod-us-east-1/recipes/nova_lite_1_0_p5_p4d_gpu_lora_sft_override_params_sm_jobs_v1.0.20.json",
                    }
                ]
            }
        }

        result = get_hub_recipe_metadata(
            model=Model.NOVA_LITE,
            method=TrainingMethod.SFT_LORA,
            platform=Platform.SMHP,
            instance_type="ml.p5.48xlarge",
            region="us-east-1",
        )

        self.assertEqual(result["DisplayName"], "Nova Lite LoRA SFT on GPU")
        self.assertIn("HpEksPayloadTemplateS3Uri", result)
        self.assertIn("HpEksOverrideParamsS3Uri", result)

    @patch("amzn_nova_customization_sdk.util.recipe._get_hub_content")
    def test_get_hub_recipe_metadata_filters_forge_recipes(self, mock_get_hub_content):
        mock_get_hub_content.return_value = {
            "HubContentDocument": {
                "RecipeCollection": [
                    {
                        "DisplayName": "Nova Lite LoRA SFT on GPU",
                        "Name": "nova_lite_1_0_p5_gpu_lora_sft_text_with_datamix",
                        "CustomizationTechnique": "SFT",
                        "Peft": "LORA",
                        "IsSubscriptionModel": True,  # This is a Forge recipe
                        "SmtjRecipeTemplateS3Uri": "s3://arn:aws:s3:us-east-1:334772094012:accesspoint/recipe.yaml",
                        "SmtjOverrideParamsS3Uri": "s3://arn:aws:s3:us-east-1:334772094012:accesspoint/overrides.json",
                    },
                    {
                        "DisplayName": "Nova Lite LoRA SFT on GPU",
                        "Name": "nova_lite_1_0_p5_p4d_gpu_lora_sft",
                        "CustomizationTechnique": "SFT",
                        "Peft": "LORA",
                        "SmtjRecipeTemplateS3Uri": "s3://jumpstart-cache-prod-us-east-1/recipes/recipe.yaml",
                        "SmtjOverrideParamsS3Uri": "s3://jumpstart-cache-prod-us-east-1/recipes/overrides.json",
                    },
                ]
            }
        }

        result = get_hub_recipe_metadata(
            model=Model.NOVA_LITE,
            method=TrainingMethod.SFT_LORA,
            platform=Platform.SMTJ,
            instance_type="ml.g5.12xlarge",
            region="us-east-1",
        )

        self.assertEqual(result["Name"], "nova_lite_1_0_p5_p4d_gpu_lora_sft")
        self.assertNotIn("IsSubscriptionModel", result)

    @patch("amzn_nova_customization_sdk.util.recipe._get_hub_content")
    def test_get_hub_recipe_metadata_with_data_mixing_includes_forge_recipes(
        self, mock_get_hub_content
    ):
        mock_get_hub_content.return_value = {
            "HubContentDocument": {
                "RecipeCollection": [
                    {
                        "DisplayName": "Nova Lite LoRA SFT on GPU with Data Mixing",
                        "Name": "nova_lite_1_0_p5_gpu_lora_sft_text_with_datamix",
                        "CustomizationTechnique": "SFT",
                        "Peft": "LORA",
                        "IsSubscriptionModel": True,
                        "SupportedInstanceTypes": ["ml.p5.48xlarge"],
                        "HpEksPayloadTemplateS3Uri": "s3://arn:aws:s3:us-east-1:334772094012:accesspoint/payload_datamix.yaml",
                        "HpEksOverrideParamsS3Uri": "s3://arn:aws:s3:us-east-1:334772094012:accesspoint/overrides_datamix.json",
                    },
                    {
                        "DisplayName": "Nova Lite LoRA SFT on GPU",
                        "Name": "nova_lite_1_0_p5_p4d_gpu_lora_sft",
                        "CustomizationTechnique": "SFT",
                        "Peft": "LORA",
                        "SupportedInstanceTypes": ["ml.p5.48xlarge"],
                        "HpEksPayloadTemplateS3Uri": "s3://jumpstart-cache-prod-us-east-1/recipes/payload.yaml",
                        "HpEksOverrideParamsS3Uri": "s3://jumpstart-cache-prod-us-east-1/recipes/overrides.json",
                    },
                ]
            }
        }

        result = get_hub_recipe_metadata(
            model=Model.NOVA_LITE,
            method=TrainingMethod.SFT_LORA,
            platform=Platform.SMHP,
            instance_type="ml.p5.48xlarge",
            region="us-east-1",
            data_mixing=True,
        )

        self.assertEqual(
            result["Name"], "nova_lite_1_0_p5_gpu_lora_sft_text_with_datamix"
        )
        self.assertTrue(result.get("IsSubscriptionModel"))
        self.assertIn("text_with_datamix", result["Name"])

    @patch("amzn_nova_customization_sdk.util.recipe._get_hub_content")
    def test_get_hub_recipe_metadata_no_recipes_for_method(self, mock_get_hub_content):
        mock_get_hub_content.return_value = {
            "HubContentDocument": {"RecipeCollection": []}
        }

        with self.assertRaises(ValueError) as context:
            get_hub_recipe_metadata(
                model=Model.NOVA_LITE,
                method=TrainingMethod.SFT_LORA,
                platform=Platform.SMTJ,
                instance_type="ml.g5.12xlarge",
                region="us-east-1",
            )
        self.assertIn("is not supported", str(context.exception))

    @patch("amzn_nova_customization_sdk.util.recipe._get_hub_content")
    def test_get_hub_recipe_metadata_no_recipes_for_platform(
        self, mock_get_hub_content
    ):
        mock_get_hub_content.return_value = {
            "HubContentDocument": {
                "RecipeCollection": [
                    {
                        "CustomizationTechnique": "SFT",
                        "Peft": "LORA",
                        # Missing SmtjRecipeTemplateS3Uri and HpEksPayloadTemplateS3Uri
                    }
                ]
            }
        }

        with self.assertRaises(ValueError) as context:
            get_hub_recipe_metadata(
                model=Model.NOVA_LITE,
                method=TrainingMethod.SFT_LORA,
                platform=Platform.SMTJ,
                instance_type="ml.g5.12xlarge",
                region="us-east-1",
            )
        self.assertIn("is not supported on", str(context.exception))

    @patch("amzn_nova_customization_sdk.util.recipe._get_hub_content")
    def test_get_hub_recipe_metadata_no_matching_training_type(
        self, mock_get_hub_content
    ):
        mock_get_hub_content.return_value = {
            "HubContentDocument": {
                "RecipeCollection": [
                    {
                        "CustomizationTechnique": "SFT",
                        # No "Peft" key means full fine-tuning
                        "SmtjRecipeTemplateS3Uri": "s3://bucket/recipe.yaml",
                        "SmtjOverrideParamsS3Uri": "s3://bucket/overrides.json",
                    }
                ]
            }
        }

        with self.assertRaises(ValueError) as context:
            get_hub_recipe_metadata(
                model=Model.NOVA_LITE,
                method=TrainingMethod.SFT_LORA,
                platform=Platform.SMTJ,
                instance_type="ml.g5.12xlarge",
                region="us-east-1",
            )
        self.assertIn("is not supported on", str(context.exception))

    @patch("amzn_nova_customization_sdk.util.recipe._get_hub_content")
    def test_get_hub_recipe_metadata_multiple_recipes_selects_correct_one(
        self, mock_get_hub_content
    ):
        mock_get_hub_content.return_value = {
            "HubContentDocument": {
                "RecipeCollection": [
                    {
                        "DisplayName": "Nova Lite LoRA SFT on G5",
                        "Name": "nova_lite_1_0_g5_g6_12x_gpu_lora_sft",
                        "CustomizationTechnique": "SFT",
                        "Peft": "LORA",
                        "SupportedInstanceTypes": ["ml.g5.12xlarge"],
                        "SmtjRecipeTemplateS3Uri": "s3://bucket/recipe1.yaml",
                        "SmtjOverrideParamsS3Uri": "s3://bucket/overrides1.json",
                    },
                    {
                        "DisplayName": "Nova Lite SFT on P5",
                        "Name": "nova_lite_1_0_p5_p4d_gpu_sft",
                        "CustomizationTechnique": "SFT",
                        # No Peft - this is full fine-tuning
                        "SupportedInstanceTypes": ["ml.p5.48xlarge"],
                        "SmtjRecipeTemplateS3Uri": "s3://bucket/recipe2.yaml",
                        "SmtjOverrideParamsS3Uri": "s3://bucket/overrides2.json",
                    },
                ]
            }
        }

        result = get_hub_recipe_metadata(
            model=Model.NOVA_LITE,
            method=TrainingMethod.SFT_LORA,
            platform=Platform.SMTJ,
            instance_type="ml.g5.12xlarge",
            region="us-east-1",
        )

        self.assertEqual(result["Name"], "nova_lite_1_0_g5_g6_12x_gpu_lora_sft")
        self.assertEqual(result["Peft"], "LORA")

    @patch("amzn_nova_customization_sdk.util.recipe._get_hub_content")
    def test_get_hub_recipe_metadata_rft_converts_to_rlvr(self, mock_get_hub_content):
        mock_get_hub_content.return_value = {
            "HubContentDocument": {
                "RecipeCollection": [
                    {
                        "DisplayName": "Nova Lite RFT on GPU",
                        "Name": "nova_lite_rft",
                        "CustomizationTechnique": "RLVR",
                        "SmtjRecipeTemplateS3Uri": "s3://bucket/recipe.yaml",
                        "SmtjOverrideParamsS3Uri": "s3://bucket/overrides.json",
                    }
                ]
            }
        }

        result = get_hub_recipe_metadata(
            model=Model.NOVA_LITE,
            method=TrainingMethod.RFT_FULL,
            platform=Platform.SMTJ,
            instance_type="ml.g5.12xlarge",
            region="us-east-1",
        )

        self.assertEqual(result["CustomizationTechnique"], "RLVR")
        self.assertNotIn("Peft", result)

    @patch("amzn_nova_customization_sdk.util.recipe._get_hub_content")
    def test_get_hub_recipe_metadata_evaluation_missing_task(
        self, mock_get_hub_content
    ):
        mock_get_hub_content.return_value = {
            "HubContentDocument": {
                "RecipeCollection": [
                    {
                        "DisplayName": "Nova Lite Evaluation - Bring Your Own Dataset",
                        "Name": "nova_lite_eval_byo",
                        "Type": "Evaluation",
                        "InstanceCount": 1,
                        "SupportedInstanceTypes": ["ml.p5.48xlarge"],
                        "SmtjRecipeTemplateS3Uri": "s3://bucket/recipe.yaml",
                        "SmtjOverrideParamsS3Uri": "s3://bucket/overrides.json",
                    }
                ]
            }
        }

        with self.assertRaises(ValueError) as context:
            get_hub_recipe_metadata(
                model=Model.NOVA_LITE,
                method=TrainingMethod.EVALUATION,
                platform=Platform.SMTJ,
                region="us-east-1",
                instance_type="ml.p5.48xlarge",
                task=None,
            )

        self.assertEqual(
            str(context.exception),
            "'eval_task' is a required parameter when calling evaluate().",
        )

    @patch("amzn_nova_customization_sdk.util.recipe._get_hub_content")
    def test_get_hub_recipe_metadata_evaluation_gen_qa(self, mock_get_hub_content):
        mock_get_hub_content.return_value = {
            "HubContentDocument": {
                "RecipeCollection": [
                    {
                        "DisplayName": "Nova Lite Evaluation - Bring Your Own Dataset",
                        "Name": "nova_lite_eval_byo",
                        "Type": "Evaluation",
                        "InstanceCount": 1,
                        "SupportedInstanceTypes": ["ml.p5.48xlarge"],
                        "SmtjRecipeTemplateS3Uri": "s3://bucket/recipe.yaml",
                        "SmtjOverrideParamsS3Uri": "s3://bucket/overrides.json",
                    },
                    {
                        "DisplayName": "Nova Lite Evaluation - General Text Benchmark",
                        "Name": "nova_lite_eval_benchmark",
                        "Type": "Evaluation",
                        "InstanceCount": 1,
                        "SupportedInstanceTypes": ["ml.p5.48xlarge"],
                        "SmtjRecipeTemplateS3Uri": "s3://bucket/recipe2.yaml",
                        "SmtjOverrideParamsS3Uri": "s3://bucket/overrides2.json",
                    },
                ]
            }
        }

        result = get_hub_recipe_metadata(
            model=Model.NOVA_LITE,
            method=TrainingMethod.EVALUATION,
            platform=Platform.SMTJ,
            region="us-east-1",
            instance_type="ml.p5.48xlarge",
            task=EvaluationTask.GEN_QA,
        )

        self.assertEqual(result["Name"], "nova_lite_eval_byo")
        self.assertIn("bring your own dataset", result["DisplayName"].lower())

    @patch("amzn_nova_customization_sdk.util.recipe._get_hub_content")
    @patch("amzn_nova_customization_sdk.util.recipe.os.path.dirname")
    def test_get_hub_recipe_metadata_evaluation_llm_judge(
        self, mock_dirname, mock_get_hub_content
    ):
        mock_dirname.return_value = "/mock/sdk/path"
        mock_get_hub_content.return_value = {
            "HubContentDocument": {
                "RecipeCollection": [
                    {
                        "DisplayName": "Nova Lite Evaluation",
                        "Name": "nova_lite_eval",
                        "Type": "Evaluation",
                        "SmtjRecipeTemplateS3Uri": "s3://bucket/recipe.yaml",
                        "SmtjOverrideParamsS3Uri": "s3://bucket/overrides.json",
                    }
                ]
            }
        }

        result = get_hub_recipe_metadata(
            model=Model.NOVA_LITE,
            method=TrainingMethod.EVALUATION,
            platform=Platform.SMTJ,
            region="us-east-1",
            instance_type="ml.g5.12xlarge",
            task=EvaluationTask.LLM_JUDGE,
        )

        self.assertEqual(result["InstanceCount"], 1)
        self.assertEqual(result["SupportedInstanceTypes"], ["ml.p5.48xlarge"])
        self.assertIn("RecipeTemplatePath", result)
        self.assertIn("OverrideParamsPath", result)
        self.assertIn("llm_judge", result["RecipeTemplatePath"])
        self.assertIn("llm_judge", result["OverrideParamsPath"])

    @patch("amzn_nova_customization_sdk.util.recipe._get_hub_content")
    @patch("amzn_nova_customization_sdk.util.recipe.os.path.dirname")
    def test_get_hub_recipe_metadata_evaluation_rubric_llm_judge(
        self, mock_dirname, mock_get_hub_content
    ):
        mock_dirname.return_value = "/mock/sdk/path"
        mock_get_hub_content.return_value = {
            "HubContentDocument": {
                "RecipeCollection": [
                    {
                        "DisplayName": "Nova Lite Evaluation",
                        "Name": "nova_lite_eval",
                        "Type": "Evaluation",
                        "SmtjRecipeTemplateS3Uri": "s3://bucket/recipe.yaml",
                        "SmtjOverrideParamsS3Uri": "s3://bucket/overrides.json",
                    }
                ]
            }
        }

        result = get_hub_recipe_metadata(
            model=Model.NOVA_LITE,
            method=TrainingMethod.EVALUATION,
            platform=Platform.SMTJ,
            region="us-east-1",
            instance_type="ml.g5.12xlarge",
            task=EvaluationTask.RUBRIC_LLM_JUDGE,
        )

        self.assertEqual(result["InstanceCount"], 1)
        self.assertEqual(result["SupportedInstanceTypes"], ["ml.p5.48xlarge"])
        self.assertIn("RecipeTemplatePath", result)
        self.assertIn("OverrideParamsPath", result)
        self.assertIn("rubric_llm_judge", result["RecipeTemplatePath"])
        self.assertIn("rubric_llm_judge", result["OverrideParamsPath"])

    @patch("amzn_nova_customization_sdk.util.recipe._get_hub_content")
    @patch("amzn_nova_customization_sdk.util.recipe.os.path.dirname")
    def test_get_hub_recipe_metadata_evaluation_rft_eval(
        self, mock_dirname, mock_get_hub_content
    ):
        mock_dirname.return_value = "/mock/sdk/path"
        mock_get_hub_content.return_value = {
            "HubContentDocument": {
                "RecipeCollection": [
                    {
                        "DisplayName": "Nova Lite Evaluation",
                        "Name": "nova_lite_eval",
                        "Type": "Evaluation",
                        "SmtjRecipeTemplateS3Uri": "s3://bucket/recipe.yaml",
                        "SmtjOverrideParamsS3Uri": "s3://bucket/overrides.json",
                    }
                ]
            }
        }

        result = get_hub_recipe_metadata(
            model=Model.NOVA_LITE,
            method=TrainingMethod.EVALUATION,
            platform=Platform.SMTJ,
            region="us-east-1",
            instance_type="ml.g5.12xlarge",
            task=EvaluationTask.RFT_EVAL,
        )

        self.assertEqual(result["InstanceCount"], 1)
        self.assertEqual(result["SupportedInstanceTypes"], ["ml.p5.48xlarge"])
        self.assertIn("RecipeTemplatePath", result)
        self.assertIn("OverrideParamsPath", result)
        self.assertIn("rft_eval", result["RecipeTemplatePath"])
        self.assertIn("rft_eval", result["OverrideParamsPath"])

    @patch("amzn_nova_customization_sdk.util.recipe._get_hub_content")
    def test_get_hub_recipe_metadata_evaluation_general_benchmark(
        self, mock_get_hub_content
    ):
        mock_get_hub_content.return_value = {
            "HubContentDocument": {
                "RecipeCollection": [
                    {
                        "DisplayName": "Nova Lite Evaluation - General Text Benchmark",
                        "Name": "nova_lite_eval_benchmark",
                        "Type": "Evaluation",
                        "InstanceCount": 1,
                        "SupportedInstanceTypes": ["ml.p5.48xlarge"],
                        "SmtjRecipeTemplateS3Uri": "s3://bucket/recipe.yaml",
                        "SmtjOverrideParamsS3Uri": "s3://bucket/overrides.json",
                    },
                    {
                        "DisplayName": "Nova Lite Evaluation - Bring Your Own Dataset",
                        "Name": "nova_lite_eval_byo",
                        "Type": "Evaluation",
                        "InstanceCount": 1,
                        "SupportedInstanceTypes": ["ml.p5.48xlarge"],
                        "SmtjRecipeTemplateS3Uri": "s3://bucket/recipe2.yaml",
                        "SmtjOverrideParamsS3Uri": "s3://bucket/overrides2.json",
                    },
                ]
            }
        }

        result = get_hub_recipe_metadata(
            model=Model.NOVA_LITE,
            method=TrainingMethod.EVALUATION,
            platform=Platform.SMTJ,
            region="us-east-1",
            instance_type="ml.p5.48xlarge",
            task=EvaluationTask.MATH,
        )

        self.assertEqual(result["Name"], "nova_lite_eval_benchmark")
        self.assertIn("general text benchmark", result["DisplayName"].lower())

    @patch("amzn_nova_customization_sdk.util.recipe._get_hub_content")
    def test_get_hub_recipe_metadata_evaluation_no_matching_recipe(
        self, mock_get_hub_content
    ):
        mock_get_hub_content.return_value = {
            "HubContentDocument": {
                "RecipeCollection": [
                    {
                        "DisplayName": "Nova Lite Evaluation - Some Other Type",
                        "Name": "nova_lite_eval_other",
                        "Type": "Evaluation",
                        "SmtjRecipeTemplateS3Uri": "s3://bucket/recipe.yaml",
                        "SmtjOverrideParamsS3Uri": "s3://bucket/overrides.json",
                    }
                ]
            }
        }

        with self.assertRaises(ValueError) as context:
            get_hub_recipe_metadata(
                model=Model.NOVA_LITE,
                method=TrainingMethod.EVALUATION,
                platform=Platform.SMTJ,
                region="us-east-1",
                instance_type="ml.g5.12xlarge",
                task=EvaluationTask.GEN_QA,
            )

        self.assertIn("is not supported on", str(context.exception))

    @patch("amzn_nova_customization_sdk.util.recipe._get_hub_content")
    @patch("amzn_nova_customization_sdk.util.recipe.os.path.dirname")
    def test_get_hub_recipe_metadata_evaluation_nova_lite_2_model_version(
        self, mock_dirname, mock_get_hub_content
    ):
        mock_dirname.return_value = "/mock/sdk/path"
        mock_get_hub_content.return_value = {
            "HubContentDocument": {
                "RecipeCollection": [
                    {
                        "DisplayName": "Nova Lite V2 Evaluation",
                        "Name": "nova_lite_v2_eval",
                        "Type": "Evaluation",
                        "SmtjRecipeTemplateS3Uri": "s3://bucket/recipe.yaml",
                        "SmtjOverrideParamsS3Uri": "s3://bucket/overrides.json",
                    }
                ]
            }
        }

        result = get_hub_recipe_metadata(
            model=Model.NOVA_LITE_2,
            method=TrainingMethod.EVALUATION,
            platform=Platform.SMTJ,
            region="us-east-1",
            instance_type="ml.g5.12xlarge",
            task=EvaluationTask.LLM_JUDGE,
        )

        self.assertIn("RecipeTemplatePath", result)
        self.assertIn("llm_judge", result["RecipeTemplatePath"])

    @patch("amzn_nova_customization_sdk.util.recipe._get_hub_content")
    @patch("amzn_nova_customization_sdk.util.recipe.os.path.dirname")
    def test_get_hub_recipe_metadata_evaluation_path_construction(
        self, mock_dirname, mock_get_hub_content
    ):
        mock_dirname.return_value = "/mock/sdk/path"
        mock_get_hub_content.return_value = {
            "HubContentDocument": {
                "RecipeCollection": [
                    {
                        "DisplayName": "Nova Lite Evaluation",
                        "Name": "nova_lite_eval",
                        "Type": "Evaluation",
                        "SmtjRecipeTemplateS3Uri": "s3://bucket/recipe.yaml",
                        "SmtjOverrideParamsS3Uri": "s3://bucket/overrides.json",
                    }
                ]
            }
        }

        result = get_hub_recipe_metadata(
            model=Model.NOVA_LITE,
            method=TrainingMethod.EVALUATION,
            platform=Platform.SMTJ,
            region="us-east-1",
            instance_type="ml.g5.12xlarge",
            task=EvaluationTask.LLM_JUDGE,
        )

        expected_recipe_path = (
            "/mock/sdk/path/recipe/templates/recipe/llm_judge_one.yaml"
        )
        expected_override_path = (
            "/mock/sdk/path/recipe/templates/override/llm_judge_one.json"
        )

        self.assertEqual(result["RecipeTemplatePath"], expected_recipe_path)
        self.assertEqual(result["OverrideParamsPath"], expected_override_path)

    @patch("amzn_nova_customization_sdk.util.recipe._get_hub_content")
    @patch("amzn_nova_customization_sdk.util.recipe.os.path.dirname")
    @patch(
        "amzn_nova_customization_sdk.util.recipe.REGION_TO_ESCROW_ACCOUNT_MAPPING",
        {"us-east-1": "123456789012"},
    )
    def test_get_hub_recipe_metadata_evaluation_image_uri_smtj_v1(
        self, mock_dirname, mock_get_hub_content
    ):
        mock_dirname.return_value = "/mock/sdk/path"
        mock_get_hub_content.return_value = {
            "HubContentDocument": {
                "RecipeCollection": [
                    {
                        "DisplayName": "Nova Lite Evaluation",
                        "Name": "nova_lite_eval",
                        "Type": "Evaluation",
                        "SmtjRecipeTemplateS3Uri": "s3://bucket/recipe.yaml",
                        "SmtjOverrideParamsS3Uri": "s3://bucket/overrides.json",
                    }
                ]
            }
        }

        result = get_hub_recipe_metadata(
            model=Model.NOVA_LITE,
            method=TrainingMethod.EVALUATION,
            platform=Platform.SMTJ,
            region="us-east-1",
            instance_type="ml.g5.12xlarge",
            task=EvaluationTask.LLM_JUDGE,
        )

        expected_image_uri = "123456789012.dkr.ecr.us-east-1.amazonaws.com/nova-evaluation-repo:SM-TJ-Eval-latest"
        self.assertEqual(result["ImageUri"], expected_image_uri)

    @patch("amzn_nova_customization_sdk.util.recipe._get_hub_content")
    @patch("amzn_nova_customization_sdk.util.recipe.os.path.dirname")
    @patch(
        "amzn_nova_customization_sdk.util.recipe.REGION_TO_ESCROW_ACCOUNT_MAPPING",
        {"us-west-2": "987654321098"},
    )
    def test_get_hub_recipe_metadata_evaluation_image_uri_smhp_v1(
        self, mock_dirname, mock_get_hub_content
    ):
        mock_dirname.return_value = "/mock/sdk/path"
        mock_get_hub_content.return_value = {
            "HubContentDocument": {
                "RecipeCollection": [
                    {
                        "DisplayName": "Nova Lite Evaluation",
                        "Name": "nova_lite_eval",
                        "Type": "Evaluation",
                        "HpEksPayloadTemplateS3Uri": "s3://bucket/recipe.yaml",
                        "HpEksOverrideParamsS3Uri": "s3://bucket/overrides.json",
                    }
                ]
            }
        }

        result = get_hub_recipe_metadata(
            model=Model.NOVA_LITE,
            method=TrainingMethod.EVALUATION,
            platform=Platform.SMHP,
            region="us-west-2",
            instance_type="ml.g5.12xlarge",
            task=EvaluationTask.RUBRIC_LLM_JUDGE,
        )

        expected_image_uri = "987654321098.dkr.ecr.us-west-2.amazonaws.com/nova-evaluation-repo:SM-HP-Eval-latest"
        self.assertEqual(result["ImageUri"], expected_image_uri)

    @patch("amzn_nova_customization_sdk.util.recipe._get_hub_content")
    @patch("amzn_nova_customization_sdk.util.recipe.os.path.dirname")
    @patch(
        "amzn_nova_customization_sdk.util.recipe.REGION_TO_ESCROW_ACCOUNT_MAPPING",
        {"us-east-1": "123456789012"},
    )
    def test_get_hub_recipe_metadata_evaluation_image_uri_v2_model(
        self, mock_dirname, mock_get_hub_content
    ):
        mock_dirname.return_value = "/mock/sdk/path"
        mock_get_hub_content.return_value = {
            "HubContentDocument": {
                "RecipeCollection": [
                    {
                        "DisplayName": "Nova Lite V2 Evaluation",
                        "Name": "nova_lite_v2_eval",
                        "Type": "Evaluation",
                        "SmtjRecipeTemplateS3Uri": "s3://bucket/recipe.yaml",
                        "SmtjOverrideParamsS3Uri": "s3://bucket/overrides.json",
                    }
                ]
            }
        }

        result = get_hub_recipe_metadata(
            model=Model.NOVA_LITE_2,
            method=TrainingMethod.EVALUATION,
            platform=Platform.SMTJ,
            region="us-east-1",
            instance_type="ml.g5.12xlarge",
            task=EvaluationTask.RFT_EVAL,
        )

        expected_image_uri = "123456789012.dkr.ecr.us-east-1.amazonaws.com/nova-evaluation-repo:SM-TJ-Eval-V2-latest"
        self.assertEqual(result["ImageUri"], expected_image_uri)

    @patch("amzn_nova_customization_sdk.util.recipe._get_hub_content")
    @patch("amzn_nova_customization_sdk.util.recipe.os.path.dirname")
    @patch(
        "amzn_nova_customization_sdk.util.recipe.REGION_TO_ESCROW_ACCOUNT_MAPPING",
        {"eu-west-1": "111222333444"},
    )
    def test_get_hub_recipe_metadata_evaluation_image_uri_all_tasks(
        self, mock_dirname, mock_get_hub_content
    ):
        mock_dirname.return_value = "/mock/sdk/path"
        mock_get_hub_content.return_value = {
            "HubContentDocument": {
                "RecipeCollection": [
                    {
                        "DisplayName": "Nova Lite Evaluation",
                        "Name": "nova_lite_eval",
                        "Type": "Evaluation",
                        "SmtjRecipeTemplateS3Uri": "s3://bucket/recipe.yaml",
                        "SmtjOverrideParamsS3Uri": "s3://bucket/overrides.json",
                    }
                ]
            }
        }

        expected_image_uri = "111222333444.dkr.ecr.eu-west-1.amazonaws.com/nova-evaluation-repo:SM-TJ-Eval-latest"

        for task in [
            EvaluationTask.LLM_JUDGE,
            EvaluationTask.RUBRIC_LLM_JUDGE,
            EvaluationTask.RFT_EVAL,
        ]:
            result = get_hub_recipe_metadata(
                model=Model.NOVA_LITE,
                method=TrainingMethod.EVALUATION,
                platform=Platform.SMTJ,
                region="eu-west-1",
                instance_type="ml.g5.12xlarge",
                task=task,
            )

            self.assertEqual(
                result["ImageUri"],
                expected_image_uri,
                f"ImageUri mismatch for task {task.value}",
            )

    @patch("amzn_nova_customization_sdk.util.recipe._get_hub_content")
    def test_get_hub_recipe_metadata_cpt(self, mock_get_hub_content):
        mock_get_hub_content.return_value = {
            "HubContentDocument": {
                "RecipeCollection": [
                    {
                        "DisplayName": "Nova Lite Pretrain on P5 GPU",
                        "Name": "nova_lite_2_0_p5x8_gpu_pretrain",
                        "RecipeFilePath": "recipes/training/nova/nova_2_0/nova_lite/CPT/nova_lite_2_0_p5x8_gpu_pretrain.yaml",
                        "InstanceCount": 8,
                        "Type": "FineTuning",
                        "Versions": ["1.0.5"],
                        "CustomizationTechnique": "CPT",
                        "Hardware": "GPU",
                        "SupportedInstanceTypes": ["ml.p5.48xlarge"],
                        "SequenceLength": "8K",
                        "HpEksPayloadTemplateS3Uri": "s3://jumpstart-cache-prod-us-east-1/recipes/nova_lite_2_0_p5x8_gpu_pretrain_payload_template_k8s_v1.0.20.yaml",
                        "HpEksOverrideParamsS3Uri": "s3://jumpstart-cache-prod-us-east-1/recipes/nova_lite_2_0_p5x8_gpu_pretrain_override_params_k8s_v1.0.20.json",
                    }
                ]
            }
        }

        result = get_hub_recipe_metadata(
            model=Model.NOVA_LITE_2,
            method=TrainingMethod.CPT,
            platform=Platform.SMHP,
            instance_type="ml.p5.48xlarge",
            region="us-east-1",
        )

        self.assertEqual(result["CustomizationTechnique"], "CPT")
        self.assertIn("HpEksPayloadTemplateS3Uri", result)
        self.assertIn("HpEksOverrideParamsS3Uri", result)

    @patch("amzn_nova_customization_sdk.util.recipe._get_hub_content")
    def test_get_hub_recipe_metadata_cpt_not_supported_on_smtj(
        self, mock_get_hub_content
    ):
        mock_get_hub_content.return_value = {
            "HubContentDocument": {
                "RecipeCollection": [
                    {
                        "DisplayName": "Nova Lite Pretrain on P5 GPU",
                        "Name": "nova_lite_2_0_p5x8_gpu_pretrain",
                        "RecipeFilePath": "recipes/training/nova/nova_2_0/nova_lite/CPT/nova_lite_2_0_p5x8_gpu_pretrain.yaml",
                        "InstanceCount": 8,
                        "Type": "FineTuning",
                        "Versions": ["1.0.5"],
                        "CustomizationTechnique": "CPT",
                        "Hardware": "GPU",
                        "SupportedInstanceTypes": ["ml.p5.48xlarge"],
                        "SequenceLength": "8K",
                        "HpEksPayloadTemplateS3Uri": "s3://jumpstart-cache-prod-us-east-1/recipes/nova_lite_2_0_p5x8_gpu_pretrain_payload_template_k8s_v1.0.20.yaml",
                        "HpEksOverrideParamsS3Uri": "s3://jumpstart-cache-prod-us-east-1/recipes/nova_lite_2_0_p5x8_gpu_pretrain_override_params_k8s_v1.0.20.json",
                    }
                ]
            }
        }

        with self.assertRaises(ValueError) as context:
            get_hub_recipe_metadata(
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.CPT,
                platform=Platform.SMTJ,
                instance_type="ml.p5.48xlarge",
                region="us-east-1",
            )

        self.assertIn("CPT is not supported on SMTJ", str(context.exception))

    @patch("amzn_nova_customization_sdk.util.recipe._get_hub_content")
    def test_get_hub_recipe_metadata_checks_instance_type(self, mock_get_hub_content):
        mock_get_hub_content.return_value = {
            "HubContentDocument": {
                "RecipeCollection": [
                    {
                        "DisplayName": "Nova Lite LoRA SFT on GPU",
                        "Name": "nova_lite_1_0_g5_g6_12x_gpu_lora_sft",
                        "RecipeFilePath": "recipes/fine-tuning/nova/nova_1_0/nova_lite/SFT/nova_lite_1_0_g5_g6_12x_gpu_lora_sft.yaml",
                        "CustomizationTechnique": "SFT",
                        "InstanceCount": 1,
                        "ServerlessMeteringType": "Token-based",
                        "Type": "FineTuning",
                        "Versions": ["1.0.0"],
                        "Hardware": "GPU",
                        "SupportedInstanceTypes": ["ml.g5.12xlarge", "ml.g6.12xlarge"],
                        "Peft": "LORA",
                        "SequenceLength": "8K",
                        "SmtjRecipeTemplateS3Uri": "s3://jumpstart-cache-prod-us-east-1/recipes/nova_lite_1_0_g5_g6_12x_gpu_lora_sft_payload_template_sm_jobs_v1.0.20.yaml",
                        "SmtjOverrideParamsS3Uri": "s3://jumpstart-cache-prod-us-east-1/recipes/nova_lite_1_0_g5_g6_12x_gpu_lora_sft_override_params_sm_jobs_v1.0.20.json",
                        "SmtjImageUri": "708977205387.dkr.ecr.us-east-1.amazonaws.com/nova-fine-tune-repo:SM-TJ-SFT-latest",
                    },
                    {
                        "DisplayName": "Nova Lite LoRA SFT on GPU P5 P5en",
                        "Name": "nova_lite_1_0_p5_p5en_12x_gpu_lora_sft",
                        "RecipeFilePath": "recipes/fine-tuning/nova/nova_1_0/nova_lite/SFT/nova_lite_1_0_p5_p5en_12x_gpu_lora_sft.yaml",
                        "CustomizationTechnique": "SFT",
                        "InstanceCount": 1,
                        "ServerlessMeteringType": "Token-based",
                        "Type": "FineTuning",
                        "Versions": ["1.0.0"],
                        "Hardware": "GPU",
                        "SupportedInstanceTypes": [
                            "ml.p5.48xlarge",
                            "ml.p5en.48xlarge",
                        ],
                        "Peft": "LORA",
                        "SequenceLength": "8K",
                        "SmtjRecipeTemplateS3Uri": "s3://jumpstart-cache-prod-us-east-1/recipes/nova_lite_1_0_p5_p5en_12x_gpu_lora_sft_payload_template_sm_jobs_v1.0.20.yaml",
                        "SmtjOverrideParamsS3Uri": "s3://jumpstart-cache-prod-us-east-1/recipes/nova_lite_1_0_p5_p5en_12x_gpu_lora_sft_override_params_sm_jobs_v1.0.20.json",
                        "SmtjImageUri": "708977205387.dkr.ecr.us-east-1.amazonaws.com/nova-fine-tune-repo:SM-TJ-SFT-latest",
                    },
                ]
            }
        }

        result = get_hub_recipe_metadata(
            model=Model.NOVA_LITE,
            method=TrainingMethod.SFT_LORA,
            platform=Platform.SMTJ,
            instance_type="ml.p5.48xlarge",
            region="us-east-1",
        )

        self.assertEqual(result["DisplayName"], "Nova Lite LoRA SFT on GPU P5 P5en")
        self.assertIn("ml.p5.48xlarge", result["SupportedInstanceTypes"])


class TestDownloadRecipeTemplatesFromS3(unittest.TestCase):
    @patch("amzn_nova_customization_sdk.util.recipe.boto3.client")
    def test_download_recipe_templates_smtj_success(self, mock_boto_client):
        recipe = json.dumps(
            {
                "run": {
                    "name": "{{name}}",
                    "model_type": "amazon.nova-2-lite-v1:0:256k",
                },
                "training_config": {
                    "max_steps": "{{max_steps}}",
                    "peft": {"peft_scheme": "lora"},
                },
            }
        )

        overrides_json = json.dumps(
            {
                "name": {"type": "string", "required": True, "default": "my-run"},
                "max_steps": {"type": "integer", "required": True, "default": 10},
            }
        )

        mock_s3 = MagicMock()
        mock_s3.get_object.side_effect = [
            {"Body": BytesIO(recipe.encode("utf-8"))},
            {"Body": BytesIO(overrides_json.encode("utf-8"))},
        ]
        mock_boto_client.return_value = mock_s3

        recipe_metadata = {
            "SmtjRecipeTemplateS3Uri": "s3://jumpstart-cache/recipe.yaml",
            "SmtjOverrideParamsS3Uri": "s3://jumpstart-cache/overrides.json",
            "SmtjImageUri": "image_uri",
        }

        recipe_template, overrides_template, image_uri = download_templates_from_s3(
            recipe_metadata, Platform.SMTJ, TrainingMethod.SFT_LORA
        )

        self.assertIsInstance(recipe_template, dict)
        self.assertIsInstance(overrides_template, dict)
        self.assertIsInstance(image_uri, str)
        self.assertIn("run", recipe_template)
        self.assertIn("name", overrides_template)
        self.assertEqual(mock_s3.get_object.call_count, 2)
        self.assertEqual("image_uri", image_uri)

    @patch("amzn_nova_customization_sdk.util.recipe.boto3.client")
    def test_download_recipe_templates_smhp_success(self, mock_boto_client):
        config = json.dumps(
            {
                "run": {
                    "name": "{{name}}",
                    "model_type": "amazon.nova-2-lite-v1:0:256k",
                },
                "training_config": {
                    "max_steps": "{{max_steps}}",
                    "peft": {"peft_scheme": "lora"},
                },
            }
        )

        recipe = f"""---
# Source: sagemaker-training/templates/training-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: training-config-{{{{name}}}}
containers:
- name: pytorch
  image: image
data:
  config.yaml: |-
{config}
---
# Source: sagemaker-training/templates/training.yaml
apiVersion: kubeflow.org/v1
kind: PyTorchJob"""

        overrides_json = json.dumps(
            {
                "replicas": {"type": "integer", "required": True, "default": 4},
                "namespace": {
                    "type": "string",
                    "required": True,
                    "default": "kubeflow",
                },
            }
        )

        mock_s3 = MagicMock()
        mock_s3.get_object.side_effect = [
            {"Body": BytesIO(recipe.encode("utf-8"))},
            {"Body": BytesIO(overrides_json.encode("utf-8"))},
        ]
        mock_boto_client.return_value = mock_s3

        recipe_metadata = {
            "HpEksPayloadTemplateS3Uri": "s3://jumpstart-cache/hp-recipe.yaml",
            "HpEksOverrideParamsS3Uri": "s3://jumpstart-cache/hp-overrides.json",
        }

        recipe_template, overrides_template, image_uri = download_templates_from_s3(
            recipe_metadata, Platform.SMHP, TrainingMethod.SFT_LORA
        )

        self.assertIsInstance(recipe_template, dict)
        self.assertIsInstance(overrides_template, dict)
        self.assertIsInstance(image_uri, str)
        self.assertIn("run", recipe_template)
        self.assertIn("training_config", recipe_template)
        self.assertEqual("image", image_uri)

    @patch("amzn_nova_customization_sdk.util.recipe.boto3.client")
    def test_download_recipe_templates_missing_smtj_uri(self, mock_boto_client):
        recipe_metadata = {
            "SmtjRecipeTemplateS3Uri": "s3://bucket/recipe.yaml"
            # Missing SmtjOverrideParamsS3Uri
        }

        with self.assertRaises(ValueError) as context:
            download_templates_from_s3(
                recipe_metadata, Platform.SMTJ, TrainingMethod.SFT_LORA
            )

        self.assertIn("Unable to find recipe", str(context.exception))

    @patch("amzn_nova_customization_sdk.util.recipe.boto3.client")
    def test_download_recipe_templates_missing_smhp_uri(self, mock_boto_client):
        recipe_metadata = {
            "HpEksPayloadTemplateS3Uri": "s3://bucket/recipe.yaml"
            # Missing HpEksOverrideParamsS3Uri
        }

        with self.assertRaises(ValueError) as context:
            download_templates_from_s3(
                recipe_metadata, Platform.SMHP, TrainingMethod.SFT_LORA
            )

        self.assertIn("Unable to find recipe", str(context.exception))

    @patch("amzn_nova_customization_sdk.util.recipe.boto3.client")
    def test_download_recipe_templates_s3_client_error(self, mock_boto_client):
        mock_s3 = MagicMock()
        mock_s3.get_object.side_effect = ClientError(
            error_response={"Error": {"Code": "NoSuchKey", "Message": "Not found"}},
            operation_name="GetObject",
        )
        mock_boto_client.return_value = mock_s3

        recipe_metadata = {
            "SmtjRecipeTemplateS3Uri": "s3://bucket/recipe.yaml",
            "SmtjOverrideParamsS3Uri": "s3://bucket/overrides.json",
        }

        with self.assertRaises(FileLoadError):
            download_templates_from_s3(
                recipe_metadata, Platform.SMTJ, TrainingMethod.SFT_LORA
            )

    @patch("amzn_nova_customization_sdk.util.recipe.boto3.client")
    def test_download_recipe_templates_smhp_missing_training_config(
        self, mock_boto_client
    ):
        # SMHP template without training-config.yaml should raise ValueError
        recipe_yaml = """---
apiVersion: v1
kind: ConfigMap
metadata:
  name: some-config
data:
  other.yaml: |-
    some: data"""

        overrides_json = json.dumps({"name": {"type": "string"}})

        mock_s3 = MagicMock()
        mock_s3.get_object.side_effect = [
            {"Body": BytesIO(recipe_yaml.encode("utf-8"))},
            {"Body": BytesIO(overrides_json.encode("utf-8"))},
        ]
        mock_boto_client.return_value = mock_s3

        recipe_metadata = {
            "HpEksPayloadTemplateS3Uri": "s3://bucket/recipe.yaml",
            "HpEksOverrideParamsS3Uri": "s3://bucket/overrides.json",
        }

        with self.assertRaises(ValueError) as context:
            download_templates_from_s3(
                recipe_metadata, Platform.SMHP, TrainingMethod.SFT_LORA
            )

        self.assertIn("Unable to generate HyperPod recipe", str(context.exception))

    @patch("amzn_nova_customization_sdk.util.recipe.boto3.client")
    def test_download_recipe_templates_smtj_missing_image_uri(self, mock_boto_client):
        recipe = json.dumps(
            {
                "run": {
                    "name": "{{name}}",
                    "model_type": "amazon.nova-2-lite-v1:0:256k",
                },
                "training_config": {
                    "max_steps": "{{max_steps}}",
                    "peft": {"peft_scheme": "lora"},
                },
            }
        )

        overrides_json = json.dumps(
            {
                "name": {"type": "string", "required": True, "default": "my-run"},
                "max_steps": {"type": "integer", "required": True, "default": 10},
            }
        )

        mock_s3 = MagicMock()
        mock_s3.get_object.side_effect = [
            {"Body": BytesIO(recipe.encode("utf-8"))},
            {"Body": BytesIO(overrides_json.encode("utf-8"))},
        ]
        mock_boto_client.return_value = mock_s3

        recipe_metadata = {
            "SmtjRecipeTemplateS3Uri": "s3://jumpstart-cache/recipe.yaml",
            "SmtjOverrideParamsS3Uri": "s3://jumpstart-cache/overrides.json",
            # Missing SmtjImageUri
        }

        with self.assertRaises(ValueError) as context:
            download_templates_from_s3(
                recipe_metadata, Platform.SMTJ, TrainingMethod.SFT_LORA
            )

        self.assertIn(
            "SDK does not yet support 'sft_lora' on 'SMTJ'", str(context.exception)
        )

    @patch("amzn_nova_customization_sdk.util.recipe.boto3.client")
    def test_download_recipe_templates_smhp_missing_image_in_template(
        self, mock_boto_client
    ):
        config = json.dumps(
            {
                "run": {
                    "name": "{{name}}",
                    "model_type": "amazon.nova-2-lite-v1:0:256k",
                },
                "training_config": {
                    "max_steps": "{{max_steps}}",
                    "peft": {"peft_scheme": "lora"},
                },
            }
        )

        recipe = f"""---
    # Source: sagemaker-training/templates/training-config.yaml
    apiVersion: v1
    kind: ConfigMap
    metadata:
      name: training-config-{{{{name}}}}
    containers:
    - name: pytorch
    data:
      config.yaml: |-
    {config}
    ---
    # Source: sagemaker-training/templates/training.yaml
    apiVersion: kubeflow.org/v1
    kind: PyTorchJob"""

        overrides_json = json.dumps(
            {
                "replicas": {"type": "integer", "required": True, "default": 4},
                "namespace": {
                    "type": "string",
                    "required": True,
                    "default": "kubeflow",
                },
            }
        )

        mock_s3 = MagicMock()
        mock_s3.get_object.side_effect = [
            {"Body": BytesIO(recipe.encode("utf-8"))},
            {"Body": BytesIO(overrides_json.encode("utf-8"))},
        ]
        mock_boto_client.return_value = mock_s3

        recipe_metadata = {
            "HpEksPayloadTemplateS3Uri": "s3://jumpstart-cache/hp-recipe.yaml",
            "HpEksOverrideParamsS3Uri": "s3://jumpstart-cache/hp-overrides.json",
        }

        with self.assertRaises(ValueError) as context:
            download_templates_from_s3(
                recipe_metadata, Platform.SMHP, TrainingMethod.SFT_LORA
            )

        self.assertIn("Unable to generate image URI", str(context.exception))

    @patch("amzn_nova_customization_sdk.util.recipe.boto3.client")
    def test_download_recipe_templates_smhp_image_extraction(self, mock_boto_client):
        config = json.dumps(
            {
                "run": {"name": "{{name}}"},
                "training_config": {"max_steps": "{{max_steps}}"},
            }
        )

        recipe = f"""---
    # Source: sagemaker-training/templates/training-config.yaml
    apiVersion: v1
    kind: ConfigMap
    metadata:
      name: training-config-{{{{name}}}}
    containers:
    - name: pytorch
      image: 123456789.dkr.ecr.us-east-1.amazonaws.com/my-image:latest
    data:
      config.yaml: |-
    {config}
    ---"""

        overrides_json = json.dumps({"name": {"type": "string"}})

        mock_s3 = MagicMock()
        mock_s3.get_object.side_effect = [
            {"Body": BytesIO(recipe.encode("utf-8"))},
            {"Body": BytesIO(overrides_json.encode("utf-8"))},
        ]
        mock_boto_client.return_value = mock_s3

        recipe_metadata = {
            "HpEksPayloadTemplateS3Uri": "s3://jumpstart-cache/hp-recipe.yaml",
            "HpEksOverrideParamsS3Uri": "s3://jumpstart-cache/hp-overrides.json",
        }

        recipe_template, overrides_template, image_uri = download_templates_from_s3(
            recipe_metadata, Platform.SMHP, TrainingMethod.SFT_LORA
        )

        self.assertEqual(
            "123456789.dkr.ecr.us-east-1.amazonaws.com/my-image:latest", image_uri
        )

    @patch("amzn_nova_customization_sdk.util.recipe.boto3.client")
    def test_download_recipe_templates_smhp_rft_removes_task_type(
        self, mock_boto_client
    ):
        config = """run:
      name: "{{name}}"
      model_type: "amazon.nova-2-lite-v1:0:256k"
    training_config:
      task_type: "rft"
      field: "value"
    """

        recipe = f"""---
    # Source: sagemaker-training/templates/training-config.yaml
    apiVersion: v1
    kind: ConfigMap
    metadata:
      name: training-config-{{{{name}}}}
    containers:
    - name: pytorch
      image: 123456789.dkr.ecr.us-east-1.amazonaws.com/rft-image:latest
    data:
      config.yaml: |-
        {config}
    ---
    # Source: sagemaker-training/templates/training.yaml
    apiVersion: kubeflow.org/v1
    kind: PyTorchJob"""

        overrides_json = json.dumps(
            {
                "name": {"type": "string", "required": True, "default": "rft-run"},
            }
        )

        mock_s3 = MagicMock()
        mock_s3.get_object.side_effect = [
            {"Body": BytesIO(recipe.encode("utf-8"))},
            {"Body": BytesIO(overrides_json.encode("utf-8"))},
        ]
        mock_boto_client.return_value = mock_s3

        recipe_metadata = {
            "HpEksPayloadTemplateS3Uri": "s3://jumpstart-cache/rft-recipe.yaml",
            "HpEksOverrideParamsS3Uri": "s3://jumpstart-cache/rft-overrides.json",
        }

        recipe_template, overrides_template, image_uri = download_templates_from_s3(
            recipe_metadata, Platform.SMHP, TrainingMethod.RFT_LORA
        )

        self.assertIsInstance(recipe_template, dict)
        self.assertIn("training_config", recipe_template)
        self.assertNotIn("task_type", recipe_template["training_config"])
        self.assertEqual(
            "123456789.dkr.ecr.us-east-1.amazonaws.com/rft-image:latest", image_uri
        )

    @patch("amzn_nova_customization_sdk.util.recipe.boto3.client")
    def test_download_recipe_templates_smhp_cpt_preserves_task_type(
        self, mock_boto_client
    ):
        config = json.dumps(
            {
                "run": {
                    "name": "{{name}}",
                    "model_type": "amazon.nova-2-lite-v1:0:256k",
                },
                "training_config": {
                    "task_type": "cpt",
                },
            }
        )

        recipe = f"""---
    # Source: sagemaker-training/templates/training-config.yaml
    apiVersion: v1
    kind: ConfigMap
    metadata:
      name: training-config-{{{{name}}}}
    containers:
    - name: pytorch
      image: 123456789.dkr.ecr.us-east-1.amazonaws.com/cpt-image:latest
    data:
      config.yaml: |-
    {config}
    ---
    # Source: sagemaker-training/templates/training.yaml
    apiVersion: kubeflow.org/v1
    kind: PyTorchJob"""

        overrides_json = json.dumps(
            {
                "name": {"type": "string", "required": True, "default": "cpt-run"},
            }
        )

        mock_s3 = MagicMock()
        mock_s3.get_object.side_effect = [
            {"Body": BytesIO(recipe.encode("utf-8"))},
            {"Body": BytesIO(overrides_json.encode("utf-8"))},
        ]
        mock_boto_client.return_value = mock_s3

        recipe_metadata = {
            "HpEksPayloadTemplateS3Uri": "s3://jumpstart-cache/cpt-recipe.yaml",
            "HpEksOverrideParamsS3Uri": "s3://jumpstart-cache/cpt-overrides.json",
        }

        recipe_template, overrides_template, image_uri = download_templates_from_s3(
            recipe_metadata, Platform.SMHP, TrainingMethod.CPT
        )

        self.assertIsInstance(recipe_template, dict)
        self.assertIn("training_config", recipe_template)
        self.assertIn("task_type", recipe_template["training_config"])
        self.assertEqual("cpt", recipe_template["training_config"]["task_type"])
        self.assertEqual(
            "123456789.dkr.ecr.us-east-1.amazonaws.com/cpt-image:latest", image_uri
        )


class TestDownloadRecipeTemplatesFromLocal(unittest.TestCase):
    def test_download_recipe_templates_from_local_success(self):
        recipe_yaml_content = """
run:
  name: my-training-run
  model_type: amazon.nova-2-lite-v1:0:256k
evaluation:
  task: rft_eval
"""
        overrides_json_content = json.dumps(
            {
                "name": {"type": "string", "required": True, "default": "my-run"},
            }
        )

        recipe_metadata = {
            "RecipeTemplatePath": "/path/to/recipe.yaml",
            "OverrideParamsPath": "/path/to/overrides.json",
            "ImageUri": "test-image-uri",
        }

        with patch(
            "builtins.open", mock_open(read_data=recipe_yaml_content)
        ) as mock_file:
            mock_file.return_value.read.return_value = recipe_yaml_content

            with patch("builtins.open", mock_open(read_data=overrides_json_content)):
                with (
                    patch("yaml.safe_load") as mock_yaml_load,
                    patch("json.load") as mock_json_load,
                ):
                    mock_yaml_load.return_value = {
                        "run": {
                            "name": "my-training-run",
                            "model_type": "amazon.nova-2-lite-v1:0:256k",
                        },
                        "evaluation": {
                            "task": "rft_eval",
                        },
                    }

                    mock_json_load.return_value = {
                        "name": {
                            "type": "string",
                            "required": True,
                            "default": "my-run",
                        }
                    }

                    recipe_template, overrides_template, image_uri = (
                        download_templates_from_local(recipe_metadata)
                    )

                    self.assertIsInstance(recipe_template, dict)
                    self.assertIsInstance(overrides_template, dict)
                    self.assertEqual(image_uri, "test-image-uri")
                    self.assertIn("run", recipe_template)
                    self.assertIn("evaluation", recipe_template)
                    self.assertIn("task", recipe_template["evaluation"])

    def test_download_recipe_templates_from_local_returns_tuple(self):
        """Test that function returns a tuple of (recipe_template, overrides_template, image_uri)"""
        recipe_metadata = {
            "RecipeTemplatePath": "/path/to/recipe.yaml",
            "OverrideParamsPath": "/path/to/overrides.json",
            "ImageUri": "test-image-uri",
        }

        recipe_dict = {"run": {"name": "test"}}
        overrides_dict = {"name": {"type": "string"}}

        with (
            patch("builtins.open", mock_open()),
            patch("yaml.safe_load", return_value=recipe_dict),
            patch("json.load", return_value=overrides_dict),
        ):
            result = download_templates_from_local(recipe_metadata)

            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 3)
            self.assertEqual(result[0], recipe_dict)
            self.assertEqual(result[1], overrides_dict)
            self.assertEqual(result[2], "test-image-uri")

    def test_download_recipe_templates_from_local_raises_value_error_on_exception(self):
        recipe_metadata = {
            "RecipeTemplatePath": "/path/to/recipe.yaml",
            "OverrideParamsPath": "/path/to/overrides.json",
            "ImageUri": "test-image-uri",
            "EvaluationTask": "rft_eval",
            "Platform": "SMTJ",
            "Model": "nova_lite",
        }

        with patch("builtins.open", side_effect=FileNotFoundError("File not found")):
            with self.assertRaises(ValueError) as context:
                download_templates_from_local(recipe_metadata)

            expected_message = (
                f"'{recipe_metadata['EvaluationTask']}' is not supported on "
                f"{recipe_metadata['Platform']} for {recipe_metadata['Model']}"
            )
            self.assertEqual(str(context.exception), expected_message)


if __name__ == "__main__":
    unittest.main()
