"""Tests for resolve_schema_validator — covers all VALIDATOR_CONFIG branches."""

import unittest
from unittest.mock import MagicMock

from amzn_nova_forge.dataset.dataset_validator import (
    CPTDatasetValidator,
    EvalDatasetValidator,
    RFTDatasetValidator,
    RFTMultiturnDatasetValidator,
    SFTDatasetValidator,
)
from amzn_nova_forge.dataset.dataset_validator.dataset_schema_resolver import (
    resolve_schema_validator,
)
from amzn_nova_forge.model.model_enums import Model, TrainingMethod
from amzn_nova_forge.recipe.recipe_config import EvaluationTask


class TestResolveSchemaValidator(unittest.TestCase):
    """Cover every branch of resolve_schema_validator."""

    # -- SFT returns SFTDatasetValidator (init_arg="none") --

    def test_sft_lora(self):
        v = resolve_schema_validator(TrainingMethod.SFT_LORA, Model.NOVA_LITE_2)
        self.assertIsInstance(v, SFTDatasetValidator)

    def test_sft_full(self):
        v = resolve_schema_validator(TrainingMethod.SFT_FULL, Model.NOVA_LITE_2)
        self.assertIsInstance(v, SFTDatasetValidator)

    # -- DPO returns None (validator is None in config) --

    def test_dpo_lora_returns_none(self):
        self.assertIsNone(resolve_schema_validator(TrainingMethod.DPO_LORA, Model.NOVA_LITE))

    def test_dpo_full_returns_none(self):
        self.assertIsNone(resolve_schema_validator(TrainingMethod.DPO_FULL, Model.NOVA_LITE))

    # -- RFT returns RFTDatasetValidator (init_arg="model") --

    def test_rft_lora(self):
        v = resolve_schema_validator(TrainingMethod.RFT_LORA, Model.NOVA_LITE_2)
        self.assertIsInstance(v, RFTDatasetValidator)

    def test_rft_full(self):
        v = resolve_schema_validator(TrainingMethod.RFT_FULL, Model.NOVA_LITE_2)
        self.assertIsInstance(v, RFTDatasetValidator)

    # -- RFT Multiturn returns RFTMultiturnDatasetValidator (init_arg="model") --

    def test_rft_multiturn_lora(self):
        v = resolve_schema_validator(TrainingMethod.RFT_MULTITURN_LORA, Model.NOVA_LITE_2)
        self.assertIsInstance(v, RFTMultiturnDatasetValidator)

    def test_rft_multiturn_full(self):
        v = resolve_schema_validator(TrainingMethod.RFT_MULTITURN_FULL, Model.NOVA_LITE_2)
        self.assertIsInstance(v, RFTMultiturnDatasetValidator)

    # -- CPT returns CPTDatasetValidator (init_arg="none") --

    def test_cpt(self):
        v = resolve_schema_validator(TrainingMethod.CPT, Model.NOVA_LITE_2)
        self.assertIsInstance(v, CPTDatasetValidator)

    # -- EVALUATION returns EvalDatasetValidator (init_arg="eval_task") --

    def test_evaluation_default(self):
        v = resolve_schema_validator(TrainingMethod.EVALUATION, Model.NOVA_LITE_2)
        self.assertIsInstance(v, EvalDatasetValidator)

    def test_evaluation_with_gen_qa(self):
        v = resolve_schema_validator(
            TrainingMethod.EVALUATION,
            Model.NOVA_LITE_2,
            eval_task=EvaluationTask.GEN_QA,
        )
        self.assertIsInstance(v, EvalDatasetValidator)

    # -- EVALUATION with RFT_MULTITURN_EVAL override --

    def test_evaluation_rft_multiturn_eval_override(self):
        v = resolve_schema_validator(
            TrainingMethod.EVALUATION,
            Model.NOVA_LITE_2,
            eval_task=EvaluationTask.RFT_MULTITURN_EVAL,
        )
        self.assertIsInstance(v, RFTMultiturnDatasetValidator)

    # -- Unknown training method returns None --

    def test_unknown_training_method_returns_none(self):
        fake_method = MagicMock()
        self.assertIsNone(resolve_schema_validator(fake_method, Model.NOVA_LITE_2))


if __name__ == "__main__":
    unittest.main()
