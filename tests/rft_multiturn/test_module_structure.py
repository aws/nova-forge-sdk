"""Unit tests for RFT multiturn module structure and imports."""

import pytest


class TestRFTMultiturnModuleStructure:
    """Test module structure after refactoring."""

    def test_custom_environment_submodule_imports(self):
        """Test that custom_environment submodule imports work correctly."""
        from amzn_nova_forge_sdk.rft_multiturn.custom_environment import (
            MULTI_TURN_TEMPLATE,
            PYPROJECT_TEMPLATE,
            README_TEMPLATE,
            SINGLE_TURN_TEMPLATE,
            CustomEnvironment,
        )

        assert CustomEnvironment is not None
        assert isinstance(SINGLE_TURN_TEMPLATE, str)
        assert isinstance(MULTI_TURN_TEMPLATE, str)
        assert isinstance(PYPROJECT_TEMPLATE, str)
        assert isinstance(README_TEMPLATE, str)

    def test_rft_multiturn_main_exports(self):
        """Test that main RFT multiturn module exports are accessible."""
        from amzn_nova_forge_sdk.rft_multiturn import (
            CustomEnvironment,
            EnvType,
            RFTMultiturnInfrastructure,
            StackOutputs,
            VFEnvId,
        )

        assert RFTMultiturnInfrastructure is not None
        assert CustomEnvironment is not None
        assert EnvType is not None
        assert VFEnvId is not None
        assert StackOutputs is not None

    def test_backward_compatibility_top_level_import(self):
        """Test backward compatibility for top-level SDK imports."""
        from amzn_nova_forge_sdk import (
            EnvType,
            RFTMultiturnInfrastructure,
            VFEnvId,
        )

        assert RFTMultiturnInfrastructure is not None
        assert EnvType is not None
        assert VFEnvId is not None


class TestCustomEnvironmentTemplates:
    """Test custom environment templates."""

    def test_templates_contain_required_content(self):
        """Test that templates contain expected content."""
        from amzn_nova_forge_sdk.rft_multiturn.custom_environment import (
            MULTI_TURN_TEMPLATE,
            PYPROJECT_TEMPLATE,
            README_TEMPLATE,
            SINGLE_TURN_TEMPLATE,
        )

        # Single turn template should have load_environment function
        assert "def load_environment" in SINGLE_TURN_TEMPLATE
        assert "vf.Environment" in SINGLE_TURN_TEMPLATE

        # Multi turn template should have load_environment function
        assert "def load_environment" in MULTI_TURN_TEMPLATE
        assert "vf.Environment" in MULTI_TURN_TEMPLATE

        # Pyproject template should have project metadata
        assert "[project]" in PYPROJECT_TEMPLATE

        # README template should be a string
        assert len(README_TEMPLATE) > 0

    def test_custom_environment_class_exists(self):
        """Test that CustomEnvironment class is accessible."""
        from amzn_nova_forge_sdk.rft_multiturn.custom_environment import (
            CustomEnvironment,
        )

        assert hasattr(CustomEnvironment, "create")
