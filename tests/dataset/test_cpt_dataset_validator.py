import json
import shutil
import tempfile
import unittest

from amzn_nova_customization_sdk.dataset.dataset_loader import (
    JSONLDatasetLoader,
)
from amzn_nova_customization_sdk.dataset.dataset_validator import CPTDatasetValidator
from amzn_nova_customization_sdk.model.model_enums import Model, TrainingMethod


class TestCPTDatasetValidator(unittest.TestCase):
    def setUp(self):
        """Create a temporary directory before each test for storing test data."""
        self.temp_dir = tempfile.mkdtemp()

    def create_temp_file(self, name: str, data):
        temp_file_name = f"{self.temp_dir}/{name}.jsonl"
        with open(temp_file_name, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        return temp_file_name

    def test_cpt_simple_success(self):
        cpt_simple_success = [
            {"text": "This is a valid text sample for continued pre-training."}
        ]

        test_file = self.create_temp_file("cpt_simple_success", cpt_simple_success)
        JSONLDatasetLoader().load(test_file).validate(
            TrainingMethod.CPT, Model.NOVA_LITE_2
        )

    def test_cpt_multiple_samples_success(self):
        cpt_multiple_samples = [
            {"text": "First sample text for continued pre-training."},
            {"text": "Second sample with different content."},
            {
                "text": "Third sample to ensure validation works across multiple entries."
            },
        ]

        test_file = self.create_temp_file("cpt_multiple_samples", cpt_multiple_samples)
        JSONLDatasetLoader().load(test_file).validate(
            TrainingMethod.CPT, Model.NOVA_PRO
        )

    def test_cpt_long_text_success(self):
        cpt_long_text = [
            {
                "text": "This is a much longer text sample that contains multiple sentences. "
                "It demonstrates that the validator can handle longer content. "
                "Continued pre-training often uses substantial text passages to help "
                "the model learn domain-specific language patterns and knowledge. "
                "This validator ensures the text field is properly formatted."
            }
        ]

        test_file = self.create_temp_file("cpt_long_text", cpt_long_text)
        JSONLDatasetLoader().load(test_file).validate(
            TrainingMethod.CPT, Model.NOVA_LITE
        )

    def test_cpt_empty_text_fail(self):
        cpt_empty_text = [{"text": ""}]

        test_file = self.create_temp_file("cpt_empty_text", cpt_empty_text)
        with self.assertRaises(ValueError) as context:
            JSONLDatasetLoader().load(test_file).validate(
                TrainingMethod.CPT, Model.NOVA_MICRO
            )
        self.assertIn("cannot be empty or whitespace-only", str(context.exception))

    def test_cpt_whitespace_only_text_fail(self):
        cpt_whitespace_text = [{"text": "   \n\t   "}]

        test_file = self.create_temp_file("cpt_whitespace_text", cpt_whitespace_text)
        with self.assertRaises(ValueError) as context:
            JSONLDatasetLoader().load(test_file).validate(
                TrainingMethod.CPT, Model.NOVA_LITE_2
            )
        self.assertIn("cannot be empty or whitespace-only", str(context.exception))

    def test_cpt_missing_text_field_fail(self):
        cpt_missing_text = [{"content": "This should be 'text', not 'content'"}]

        test_file = self.create_temp_file("cpt_missing_text", cpt_missing_text)
        with self.assertRaises(ValueError):
            JSONLDatasetLoader().load(test_file).validate(
                TrainingMethod.CPT, Model.NOVA_LITE_2
            )

    def test_cpt_extra_fields_fail(self):
        cpt_extra_fields = [
            {
                "text": "Valid text content",
                "extra_field": "This field should not be here",
            }
        ]

        test_file = self.create_temp_file("cpt_extra_fields", cpt_extra_fields)
        with self.assertRaises(ValueError) as context:
            JSONLDatasetLoader().load(test_file).validate(
                TrainingMethod.CPT, Model.NOVA_LITE_2
            )
        self.assertIn("Extra inputs are not permitted", str(context.exception))

    def test_cpt_multiple_extra_fields_fail(self):
        cpt_multiple_extra = [
            {
                "text": "Valid text content",
                "id": "sample-001",
                "metadata": {"source": "web"},
                "timestamp": "2024-01-01",
            }
        ]

        test_file = self.create_temp_file("cpt_multiple_extra", cpt_multiple_extra)
        with self.assertRaises(ValueError) as context:
            JSONLDatasetLoader().load(test_file).validate(
                TrainingMethod.CPT, Model.NOVA_LITE_2
            )
        self.assertIn("Extra inputs are not permitted", str(context.exception))

    def test_cpt_mixed_valid_invalid_samples_fail(self):
        cpt_mixed_samples = [
            {"text": "First valid sample"},
            {
                "text": ""  # Invalid: empty text
            },
            {"text": "Third valid sample"},
        ]

        test_file = self.create_temp_file("cpt_mixed_samples", cpt_mixed_samples)
        with self.assertRaises(ValueError) as context:
            JSONLDatasetLoader().load(test_file).validate(
                TrainingMethod.CPT, Model.NOVA_LITE_2
            )
        self.assertIn("Sample 1", str(context.exception))

    def test_cpt_with_special_characters_success(self):
        cpt_special_chars = [
            {
                "text": "Text with special characters: @#$%^&*() and unicode: 你好, Привет, مرحبا"
            }
        ]

        test_file = self.create_temp_file("cpt_special_chars", cpt_special_chars)
        JSONLDatasetLoader().load(test_file).validate(
            TrainingMethod.CPT, Model.NOVA_LITE_2
        )

    def test_cpt_with_newlines_success(self):
        cpt_newlines = [{"text": "Line 1\nLine 2\nLine 3\n\nParagraph 2"}]

        test_file = self.create_temp_file("cpt_newlines", cpt_newlines)
        JSONLDatasetLoader().load(test_file).validate(
            TrainingMethod.CPT, Model.NOVA_LITE_2
        )

    def test_cpt_wrong_field_type_fail(self):
        cpt_wrong_type = [{"text": 12345}]

        test_file = self.create_temp_file("cpt_wrong_type", cpt_wrong_type)
        with self.assertRaises(ValueError) as context:
            JSONLDatasetLoader().load(test_file).validate(
                TrainingMethod.CPT, Model.NOVA_LITE_2
            )
        self.assertIn("Input should be a valid string", str(context.exception))

    def test_cpt_null_text_fail(self):
        cpt_null_text = [{"text": None}]

        test_file = self.create_temp_file("cpt_null_text", cpt_null_text)
        with self.assertRaises(ValueError):
            JSONLDatasetLoader().load(test_file).validate(
                TrainingMethod.CPT, Model.NOVA_LITE_2
            )

    def test_cpt_get_optional_fields(self):
        validator = CPTDatasetValidator()
        optional_fields = validator.get_optional_fields()
        self.assertEqual(optional_fields, [])
        self.assertIsInstance(optional_fields, list)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
