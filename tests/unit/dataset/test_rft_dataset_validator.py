# Copyright Amazon.com, Inc. or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import tempfile
import unittest

from amzn_nova_forge.core.enums import Model, TrainingMethod
from amzn_nova_forge.dataset.dataset_loader import (
    JSONLDatasetLoader,
)


class TestRFTDatasetValidator(unittest.TestCase):
    def setUp(self):
        """Create a temporary directory before each test for storing test data."""
        self.temp_dir = tempfile.mkdtemp()

    def create_temp_file(self, name: str, data):
        temp_file_name = f"{self.temp_dir}/{name}.jsonl"
        with open(temp_file_name, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        return temp_file_name

    def test_rft_simple_success(self):
        rft_simple_success = [
            {
                "id": "chem-001",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful chemistry assistant",
                    },
                    {
                        "role": "user",
                        "content": "Predict hydrogen bond donors and acceptors for this SMILES: CCN(CC)CCC(=O)c1sc(N)nc1C",
                    },
                ],
                "reference_answer": {"donor_bond_counts": 2, "acceptor_bond_counts": 4},
            }
        ]

        test_file = self.create_temp_file("rft_simple_success", rft_simple_success)
        JSONLDatasetLoader().load(test_file).validate(
            training_method=TrainingMethod.RFT_FULL, model=Model.NOVA_LITE_2
        )

    def test_rft_simple_fail(self):
        rft_simple_fail = [
            {
                "id": "chem-001",
                "NOT messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful chemistry assistant",
                    },
                    {
                        "role": "user",
                        "content": "Predict hydrogen bond donors and acceptors for this SMILES: CCN(CC)CCC(=O)c1sc(N)nc1C",
                    },
                ],
                "reference_answer": {"donor_bond_counts": 2, "acceptor_bond_counts": 4},
            }
        ]

        test_file = self.create_temp_file("rft_simple_fail", rft_simple_fail)

        dataset_loader = JSONLDatasetLoader().load(test_file)
        with self.assertRaises(ValueError):
            dataset_loader.validate(
                training_method=TrainingMethod.RFT_FULL, model=Model.NOVA_LITE_2
            )

    def test_rft_with_tools_success(self):
        rft_with_tools_success = [
            {
                "id": "tool-001",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful game master assistant",
                    },
                    {
                        "role": "user",
                        "content": "Generate a strength stat for a warrior character. Apply a +2 racial bonus modifier.",
                    },
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "StatRollAPI",
                            "description": "Generates character stats by rolling 4d6, dropping the lowest die result, and applying a modifier.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "modifier": {
                                        "description": "An integer representing the modifier to apply to the total of the stat roll.",
                                        "type": "integer",
                                    }
                                },
                                "required": ["modifier"],
                            },
                        },
                    }
                ],
                "reference_answer": {
                    "tool_called": "StatRollAPI",
                    "tool_parameters": {"modifier": 2},
                    "expected_behavior": "Call StatRollAPI with modifier=2 and return the calculated stat value",
                },
            }
        ]

        test_file = self.create_temp_file("rft_with_tools_success", rft_with_tools_success)
        JSONLDatasetLoader().load(test_file).validate(
            training_method=TrainingMethod.RFT_FULL, model=Model.NOVA_LITE_2
        )

    def test_rft_with_tools_fail(self):
        rft_with_tools_fail = [
            {
                "id": "tool-001",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful game master assistant",
                    },
                    {
                        "role": "user",
                        "content": "Generate a strength stat for a warrior character. Apply a +2 racial bonus modifier.",
                    },
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "StatRollAPI",
                            "description": "Generates character stats by rolling 4d6, dropping the lowest die result, and applying a modifier.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "modifier": {
                                        "description": "An integer representing the modifier to apply to the total of the stat roll.",
                                        "type": "integer",
                                    }
                                },
                                "required": ["modifier"],
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "StatRollAPI",
                            "description": "Generates character stats by rolling 4d6, dropping the lowest die result, and applying a modifier.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "modifier": {
                                        "description": "An integer representing the modifier to apply to the total of the stat roll.",
                                        "type": "integer",
                                    }
                                },
                                "required": ["modifier"],
                            },
                        },
                    },
                ],
                "reference_answer": {
                    "tool_called": "StatRollAPI",
                    "tool_parameters": {"modifier": 2},
                    "expected_behavior": "Call StatRollAPI with modifier=2 and return the calculated stat value",
                },
            }
        ]

        test_file = self.create_temp_file("rft_with_tools_fail", rft_with_tools_fail)

        with self.assertRaises(ValueError):
            JSONLDatasetLoader().load(test_file).validate(
                training_method=TrainingMethod.RFT_FULL, model=Model.NOVA_LITE_2
            )

    def test_rft_with_nova_one_fail(self):
        rft_sample = [
            {
                "id": "chem-001",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful chemistry assistant",
                    },
                    {
                        "role": "user",
                        "content": "Predict hydrogen bond donors and acceptors for this SMILES: CCN(CC)CCC(=O)c1sc(N)nc1C",
                    },
                ],
                "reference_answer": {"donor_bond_counts": 2, "acceptor_bond_counts": 4},
            }
        ]

        test_file = self.create_temp_file("rft_sample", rft_sample)
        with self.assertRaises(ValueError):
            JSONLDatasetLoader().load(test_file).validate(
                training_method=TrainingMethod.RFT_FULL, model=Model.NOVA_LITE
            )

    def test_rft_with_missing_optional_param_fail(self):
        rft_mismatched_samples = [
            {
                "id": "chem-001",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful chemistry assistant",
                    },
                    {
                        "role": "user",
                        "content": "Predict hydrogen bond donors and acceptors for this SMILES: CCN(CC)CCC(=O)c1sc(N)nc1C",
                    },
                ],
                "reference_answer": {"donor_bond_counts": 2, "acceptor_bond_counts": 4},
            },
            {
                "not_an_id": "chem-001",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful chemistry assistant",
                    },
                    {
                        "role": "user",
                        "content": "Predict hydrogen bond donors and acceptors for this SMILES: CCN(CC)CCC(=O)c1sc(N)nc1C",
                    },
                ],
                "reference_answer": {"donor_bond_counts": 2, "acceptor_bond_counts": 4},
            },
        ]

        test_file = self.create_temp_file("rft_mismatched_samples", rft_mismatched_samples)
        with self.assertRaises(ValueError):
            JSONLDatasetLoader().load(test_file).validate(
                training_method=TrainingMethod.RFT_FULL, model=Model.NOVA_LITE_2
            )

    # Additional tool configuration tests
    def test_rft_with_empty_tools_list_fail(self):
        """Test that empty tools list fails validation"""
        rft_empty_tools = [
            {
                "id": "empty-tools-001",
                "messages": [{"role": "user", "content": "Hello"}],
                "tools": [],  # Empty tools list should fail
            }
        ]

        test_file = self.create_temp_file("rft_empty_tools", rft_empty_tools)
        with self.assertRaises(ValueError) as context:
            JSONLDatasetLoader().load(test_file).validate(
                training_method=TrainingMethod.RFT_FULL, model=Model.NOVA_LITE_2
            )
        self.assertIn("tools list cannot be empty when provided", str(context.exception))

    def test_rft_without_tools_field_success(self):
        """Test that RFT without tools field (optional) succeeds"""
        rft_no_tools = [
            {
                "id": "no-tools-001",
                "messages": [{"role": "user", "content": "What is 2+2?"}],
                # No tools field at all - this should be valid since tools is optional
                "reference_answer": "2+2 equals 4",
            }
        ]

        test_file = self.create_temp_file("rft_no_tools", rft_no_tools)
        JSONLDatasetLoader().load(test_file).validate(
            training_method=TrainingMethod.RFT_FULL, model=Model.NOVA_LITE_2
        )

    def test_rft_with_invalid_tool_type_fail(self):
        """Test that invalid tool type fails validation"""
        rft_invalid_type = [
            {
                "id": "invalid-type-001",
                "messages": [{"role": "user", "content": "Use a tool"}],
                "tools": [
                    {
                        "type": "invalid_type",  # Should be "function"
                        "function": {
                            "name": "myTool",
                            "description": "A tool",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            }
        ]

        test_file = self.create_temp_file("rft_invalid_type", rft_invalid_type)
        with self.assertRaises(ValueError) as context:
            JSONLDatasetLoader().load(test_file).validate(
                training_method=TrainingMethod.RFT_FULL, model=Model.NOVA_LITE_2
            )
        self.assertIn("Invalid tool type, must be 'function'", str(context.exception))

    def test_rft_with_invalid_parameters_type_fail(self):
        """Test that invalid parameters type fails validation"""
        rft_invalid_params = [
            {
                "id": "invalid-params-001",
                "messages": [{"role": "user", "content": "Use a tool"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "myTool",
                            "description": "A tool",
                            "parameters": {
                                "type": "string",  # Should be "object"
                                "properties": {},
                            },
                        },
                    }
                ],
            }
        ]

        test_file = self.create_temp_file("rft_invalid_params", rft_invalid_params)
        with self.assertRaises(ValueError) as context:
            JSONLDatasetLoader().load(test_file).validate(
                training_method=TrainingMethod.RFT_FULL, model=Model.NOVA_LITE_2
            )
        self.assertIn("Invalid parameters type, must be 'object'", str(context.exception))

    def test_rft_with_empty_function_name_fail(self):
        """Test that empty function name fails validation"""
        rft_empty_name = [
            {
                "id": "empty-name-001",
                "messages": [{"role": "user", "content": "Use a tool"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "",  # Empty name should fail
                            "description": "A tool",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            }
        ]

        test_file = self.create_temp_file("rft_empty_name", rft_empty_name)
        with self.assertRaises(ValueError) as context:
            JSONLDatasetLoader().load(test_file).validate(
                training_method=TrainingMethod.RFT_FULL, model=Model.NOVA_LITE_2
            )
        self.assertIn("Invalid function name, cannot be empty", str(context.exception))

    def test_rft_with_empty_function_description_fail(self):
        """Test that empty function description fails validation"""
        rft_empty_desc = [
            {
                "id": "empty-desc-001",
                "messages": [{"role": "user", "content": "Use a tool"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "myTool",
                            "description": "",  # Empty description should fail
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            }
        ]

        test_file = self.create_temp_file("rft_empty_desc", rft_empty_desc)
        with self.assertRaises(ValueError) as context:
            JSONLDatasetLoader().load(test_file).validate(
                training_method=TrainingMethod.RFT_FULL, model=Model.NOVA_LITE_2
            )
        self.assertIn("Invalid function description, cannot be empty", str(context.exception))

    def test_rft_with_multiple_valid_tools_success(self):
        """Test that multiple different tools (no duplicates) succeeds"""
        rft_multiple_tools = [
            {
                "id": "multi-tools-001",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant with access to multiple tools.",
                    },
                    {"role": "user", "content": "Get weather and calculate something"},
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "getWeather",
                            "description": "Get weather information",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "location": {
                                        "type": "string",
                                        "description": "The location to get weather for",
                                    }
                                },
                                "required": ["location"],
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "calculate",
                            "description": "Perform calculations",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "expression": {
                                        "type": "string",
                                        "description": "The expression to calculate",
                                    }
                                },
                                "required": ["expression"],
                            },
                        },
                    },
                ],
                "reference_answer": "I'll get the weather and perform the calculation for you.",
            }
        ]

        test_file = self.create_temp_file("rft_multiple_tools", rft_multiple_tools)
        JSONLDatasetLoader().load(test_file).validate(
            training_method=TrainingMethod.RFT_FULL, model=Model.NOVA_LITE_2
        )

    def test_rft_rejects_multimodal_in_message(self):
        """Test that multimodal fields in RFT messages are rejected."""
        rft_with_image = [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Describe this",
                        "image": {
                            "format": "png",
                            "source": {"s3Location": {"uri": "s3://b/i.png", "bucketOwner": "123"}},
                        },
                    },
                ],
            }
        ]

        test_file = self.create_temp_file("rft_with_image", rft_with_image)
        with self.assertRaises(ValueError) as context:
            JSONLDatasetLoader().load(test_file).validate(
                training_method=TrainingMethod.RFT_FULL, model=Model.NOVA_LITE_2
            )
        self.assertIn("extra", str(context.exception).lower())

    def test_rft_accepts_custom_fields_at_sample_level(self):
        """Test that arbitrary custom fields at the sample level are accepted."""
        rft_with_custom_fields = [
            {
                "messages": [
                    {"role": "user", "content": "Solve: 2x + 5 = 13"},
                ],
                "reference_answer": {"solution": "x = 4"},
                "task_id": "algebra_001",
                "difficulty_level": "easy",
                "domain": "algebra",
                "expected_reasoning_steps": 3,
            }
        ]

        test_file = self.create_temp_file("rft_custom_fields", rft_with_custom_fields)
        JSONLDatasetLoader().load(test_file).validate(
            training_method=TrainingMethod.RFT_FULL, model=Model.NOVA_LITE_2
        )

    def test_rft_developer_role_normalized_to_system(self):
        """Test that developer role is accepted and normalized to system."""
        from amzn_nova_forge.dataset.dataset_validator.rft_dataset_validator import RFTMessage

        msg = RFTMessage(role="developer", content="You are a helper")
        self.assertEqual(msg.role, "system")

    def test_rft_content_as_list_accepted(self):
        """Test that content as a list of content objects is accepted."""
        rft_list_content = [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this image"},
                            {
                                "type": "image_url",
                                "image_url": {"url": "https://example.com/img.png"},
                            },
                        ],
                    },
                ],
            }
        ]

        test_file = self.create_temp_file("rft_list_content", rft_list_content)
        JSONLDatasetLoader().load(test_file).validate(
            training_method=TrainingMethod.RFT_FULL, model=Model.NOVA_LITE_2
        )

    def test_rft_tool_calls_and_tool_call_id_accepted(self):
        """Test that tool_calls and tool_call_id are accepted as optional fields."""
        rft_with_tool_calls = [
            {
                "messages": [
                    {"role": "user", "content": "What's the weather in Paris?"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city": "Paris"}',
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "content": "Sunny, 22°C",
                        "tool_call_id": "call_1",
                    },
                    {"role": "assistant", "content": "The weather in Paris is sunny at 22°C."},
                ],
            }
        ]

        test_file = self.create_temp_file("rft_tool_calls", rft_with_tool_calls)
        JSONLDatasetLoader().load(test_file).validate(
            training_method=TrainingMethod.RFT_FULL, model=Model.NOVA_LITE_2
        )

    def tearDown(self):
        """Clean up temporary files created during each unit test."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)
