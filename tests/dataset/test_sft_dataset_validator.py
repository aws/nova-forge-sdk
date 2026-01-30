import json
import tempfile
import unittest

from amzn_nova_customization_sdk.dataset.dataset_loader import (
    JSONLDatasetLoader,
)
from amzn_nova_customization_sdk.model.model_enums import Model, TrainingMethod

# TODO: Clean up these unit tests to use helper functions to set up the datasets for readability.


class TestSFTDatasetValidator(unittest.TestCase):
    def setUp(self):
        """Create a temporary directory before each test for storing test data."""
        self.temp_dir = tempfile.mkdtemp()

    def create_temp_file(self, name: str, data):
        temp_file_name = f"{self.temp_dir}/{name}.jsonl"
        with open(temp_file_name, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        return temp_file_name

    # SFT 1.0 Tests
    def test_nova_micro_with_multimodal_data_fail(self):
        """Test that Nova Micro rejects multimodal data (image/video/document)"""
        nova_micro_image_fail = [
            {
                "schemaVersion": "bedrock-conversation-2024",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"text": "What's in this image?"},
                            {
                                "image": {
                                    "format": "png",
                                    "source": {
                                        "s3Location": {
                                            "uri": "s3://bucket/image.png",
                                            "bucketOwner": "123456789012",
                                        }
                                    },
                                }
                            },
                        ],
                    },
                    {"role": "assistant", "content": [{"text": "Response"}]},
                ],
            }
        ]

        test_file = self.create_temp_file(
            "nova_micro_image_fail", nova_micro_image_fail
        )
        with self.assertRaises(ValueError) as context:
            JSONLDatasetLoader().load(test_file).validate(
                TrainingMethod.SFT_LORA, Model.NOVA_MICRO
            )

    def test_sft_one_with_text_data_success(self):
        dataset_loader = JSONLDatasetLoader()
        dataset_loader.load(
            path="src/NovaCustomizationSDK/tests/test_data/sft_train_samples_converse.jsonl"
        )

        dataset_loader.validate(TrainingMethod.SFT_LORA, Model.NOVA_LITE)

    def test_sft_one_with_text_data_fail(self):
        text_data_fail = [
            {
                "schemaVersion": "bedrock-conversation-2024",
                "messages": [
                    {"typo": "user", "content": [{"text": "What is 2+2?"}]},
                    {"role": "assistant", "content": [{"text": "4"}]},
                ],
            }
        ]

        test_file = self.create_temp_file("text_data_fail", text_data_fail)

        dataset_loader = JSONLDatasetLoader().load(test_file)
        with self.assertRaises(ValueError):
            dataset_loader.validate(TrainingMethod.SFT_LORA, Model.NOVA_LITE)

    def test_sft_one_with_video_format_success(self):
        sft_one_video_success = [
            {
                "schemaVersion": "bedrock-conversation-2024",
                "system": [
                    {
                        "text": "You are a helpful assistant designed to answer questions crisply and to the point"
                    }
                ],
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"text": "How many white items are visible in this video?"},
                            {
                                "video": {
                                    "format": "mp4",
                                    "source": {
                                        "s3Location": {
                                            "uri": "s3://your-bucket/your-path/your-video.mp4",
                                            "bucketOwner": "your-aws-account-id",
                                        }
                                    },
                                }
                            },
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "text": "There are at least eight visible items that are white"
                            }
                        ],
                    },
                ],
            }
        ]

        test_file = self.create_temp_file(
            "sft_one_video_success", sft_one_video_success
        )
        JSONLDatasetLoader().load(test_file).validate(
            TrainingMethod.SFT_LORA, Model.NOVA_LITE
        )

    def test_sft_one_with_video_format_fail(self):
        sft_one_video_fail = [
            {
                "schemaVersion": "bedrock-conversation-2024",
                "system": [
                    {
                        "text": "You are a helpful assistant designed to answer questions crisply and to the point"
                    }
                ],
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"text": "How many white items are visible in this video?"},
                            {
                                "video": {
                                    "format": "fake format",
                                    "source": {
                                        "s3Location": {
                                            "uri": "s3://your-bucket/your-path/your-video.mp4",
                                            "bucketOwner": "your-aws-account-id",
                                        }
                                    },
                                }
                            },
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "text": "There are at least eight visible items that are white"
                            }
                        ],
                    },
                ],
            }
        ]

        test_file = self.create_temp_file("sft_one_video_fail", sft_one_video_fail)

        dataset_loader = JSONLDatasetLoader().load(test_file)
        with self.assertRaises(ValueError):
            dataset_loader.validate(TrainingMethod.SFT_LORA, Model.NOVA_LITE)

    def test_sft_one_with_image_success(self):
        sft_one_image_success = [
            {
                "schemaVersion": "bedrock-conversation-2024",
                "system": [
                    {
                        "text": "You are a smart assistant that answers questions respectfully"
                    }
                ],
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"text": "What does the text in this image say?"},
                            {
                                "image": {
                                    "format": "png",
                                    "source": {
                                        "s3Location": {
                                            "uri": "s3://your-bucket/your-path/your-image.png",
                                            "bucketOwner": "your-aws-account-id",
                                        }
                                    },
                                }
                            },
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"text": "The text in the attached image says 'LOL'."}
                        ],
                    },
                ],
            }
        ]

        test_file = self.create_temp_file(
            "sft_one_image_success", sft_one_image_success
        )
        JSONLDatasetLoader().load(test_file).validate(
            TrainingMethod.SFT_LORA, Model.NOVA_LITE
        )

    def test_sft_one_with_image_fail(self):
        sft_one_image_fail = [
            {
                "schemaVersion": "bedrock-conversation-2024",
                "system": [
                    {
                        "text": "You are a smart assistant that answers questions respectfully"
                    }
                ],
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"text": "What does the text in this image say?"},
                            {
                                "image": {
                                    "format": "fake_format",
                                    "source": {
                                        "s3Location": {
                                            "uri": "s3://your-bucket/your-path/your-image.png",
                                            "bucketOwner": "your-aws-account-id",
                                        }
                                    },
                                }
                            },
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"text": "The text in the attached image says 'LOL'."}
                        ],
                    },
                ],
            }
        ]

        test_file = self.create_temp_file("sft_one_image_fail", sft_one_image_fail)

        dataset_loader = JSONLDatasetLoader().load(test_file)
        with self.assertRaises(ValueError):
            dataset_loader.validate(TrainingMethod.SFT_LORA, Model.NOVA_LITE)

    def test_sft_one_mixing_image_and_video_fail(self):
        sft_one_img_vid_fail = [
            {
                "schemaVersion": "bedrock-conversation-2024",
                "system": [
                    {
                        "text": "You are a smart assistant that answers questions respectfully"
                    }
                ],
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"text": "What does the text in this image say?"},
                            {
                                "image": {
                                    "format": "png",
                                    "source": {
                                        "s3Location": {
                                            "uri": "s3://your-bucket/your-path/your-image.png",
                                            "bucketOwner": "your-aws-account-id",
                                        }
                                    },
                                },
                                "video": {
                                    "format": "mp4",
                                    "source": {
                                        "s3Location": {
                                            "uri": "s3://your-bucket/your-path/your-video.mp4",
                                            "bucketOwner": "your-aws-account-id",
                                        }
                                    },
                                },
                            },
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"text": "The text in the attached image says 'LOL'."}
                        ],
                    },
                ],
            }
        ]

        test_file = self.create_temp_file("sft_one_img_vid_fail", sft_one_img_vid_fail)
        dataset_loader = JSONLDatasetLoader().load(test_file)
        with self.assertRaises(ValueError):
            dataset_loader.validate(TrainingMethod.SFT_LORA, Model.NOVA_LITE)

    def test_sft_one_reasoning_content_fail(self):
        sft_one_reasoning_content_fail = [
            {
                "schemaVersion": "bedrock-conversation-2024",
                "system": [
                    {"text": "You are a digital assistant with a friendly personality"}
                ],
                "messages": [
                    {
                        "role": "user",
                        "content": [{"text": "What is the capital of France?"}],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "reasoningContent": {
                                    "reasoningText": {
                                        "text": "I need to recall basic world geography knowledge"
                                    }
                                }
                            },
                            {"text": "The capital of France is Paris."},
                        ],
                    },
                ],
            }
        ]

        test_file = self.create_temp_file(
            "sft_one_reasoning_content_fail", sft_one_reasoning_content_fail
        )
        with self.assertRaises(ValueError):
            JSONLDatasetLoader().load(test_file).validate(
                TrainingMethod.SFT_LORA, Model.NOVA_LITE
            )

    # SFT 2.0 Tests (Reasoning Content)
    def test_sft_two_success(self):
        sft_two_success = [
            {
                "schemaVersion": "bedrock-conversation-2024",
                "system": [
                    {"text": "You are a digital assistant with a friendly personality"}
                ],
                "messages": [
                    {
                        "role": "user",
                        "content": [{"text": "What is the capital of France?"}],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "reasoningContent": {
                                    "reasoningText": {
                                        "text": "I need to recall basic world geography knowledge"
                                    }
                                }
                            },
                            {"text": "The capital of France is Paris."},
                        ],
                    },
                ],
            }
        ]

        test_file = self.create_temp_file("sft_two_success", sft_two_success)
        JSONLDatasetLoader().load(test_file).validate(
            TrainingMethod.SFT_LORA, Model.NOVA_LITE_2
        )

    def test_sft_two_fail(self):
        test_sft_two_fail = [
            {
                "schemaVersion": "bedrock-conversation-2024",
                "system": [
                    {"text": "You are a digital assistant with a friendly personality"}
                ],
                "messages": [
                    {
                        "role": "user",
                        "content": [{"text": "What is the capital of France?"}],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "reasoningContent": {
                                    "improper_form": {
                                        "text": "I need to recall basic world geography knowledge"
                                    }
                                }
                            },
                            {"text": "The capital of France is Paris."},
                        ],
                    },
                ],
            }
        ]

        test_file = self.create_temp_file("test_sft_two_fail", test_sft_two_fail)
        with self.assertRaises(ValueError):
            JSONLDatasetLoader().load(test_file).validate(
                TrainingMethod.SFT_LORA, Model.NOVA_LITE_2
            )

    def test_sft_two_doc_success(self):
        test_sft_two_doc_success = [
            {
                "schemaVersion": "bedrock-conversation-2024",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "text": "What are the ways in which a customer can experience issues during checkout on Amazon?"
                            },
                            {
                                "document": {
                                    "format": "pdf",
                                    "source": {
                                        "s3Location": {
                                            "uri": "s3://my-bucket-name/path/to/documents/test-document.pdf",
                                            "bucketOwner": "123456789012",
                                        }
                                    },
                                }
                            },
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"text": "Assistant Response"}],
                        "reasoning_content": [
                            {
                                "text": "I need to find the relevant section in the document to answer the question.",
                                "type": "text",
                            }
                        ],
                    },
                ],
            }
        ]
        test_file = self.create_temp_file(
            "test_sft_two_doc_success", test_sft_two_doc_success
        )
        JSONLDatasetLoader().load(test_file).validate(
            TrainingMethod.SFT_LORA, Model.NOVA_LITE_2
        )

    def test_sft_two_doc_fail(self):
        test_sft_two_doc_fail = [
            {
                "schemaVersion": "bedrock-conversation-2024",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "text": "What are the ways in which a customer can experience issues during checkout on Amazon?"
                            },
                            {
                                "document": {
                                    "format": "fake format",
                                    "source": {
                                        "s3Location": {
                                            "uri": "s3://my-bucket-name/path/to/documents/test-document.pdf",
                                            "bucketOwner": "123456789012",
                                        }
                                    },
                                }
                            },
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"text": "Assistant Response"}],
                        "reasoning_content": [
                            {
                                "text": "I need to find the relevant section in the document to answer the question.",
                                "type": "text",
                            }
                        ],
                    },
                ],
            }
        ]
        test_file = self.create_temp_file(
            "test_sft_two_doc_fail", test_sft_two_doc_fail
        )
        with self.assertRaises(ValueError):
            JSONLDatasetLoader().load(test_file).validate(
                TrainingMethod.SFT_LORA, Model.NOVA_LITE_2
            )

    def test_sft_with_invalid_video_format_fails(self):
        sft_two_video_invalid_format = [
            {
                "schemaVersion": "bedrock-conversation-2024",
                "system": [
                    {
                        "text": "You are a helpful assistant designed to answer questions crisply and to the point"
                    }
                ],
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"text": "How many white items are visible in this video?"},
                            {
                                "video": {
                                    "format": "webm",  # This format is allowed for SFT 1.0 but not SFT 2.0
                                    "source": {
                                        "s3Location": {
                                            "uri": "s3://your-bucket/your-path/your-video.mp4",
                                            "bucketOwner": "your-aws-account-id",
                                        }
                                    },
                                }
                            },
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "text": "There are at least eight visible items that are white"
                            }
                        ],
                    },
                ],
            }
        ]

        test_file = self.create_temp_file(
            "sft_two_video_invalid_format", sft_two_video_invalid_format
        )

        # Works for SFT 1.0
        JSONLDatasetLoader().load(test_file).validate(
            TrainingMethod.SFT_LORA, Model.NOVA_LITE
        )

        # Doesn't work for SFT 2.0
        with self.assertRaises(ValueError):
            JSONLDatasetLoader().load(test_file).validate(
                TrainingMethod.SFT_LORA, Model.NOVA_LITE_2
            )

    def test_sft_with_invalid_img_format_fails(self):
        sft_two_img_invalid_format = [
            {
                "schemaVersion": "bedrock-conversation-2024",
                "system": [
                    {
                        "text": "You are a helpful assistant designed to answer questions crisply and to the point"
                    }
                ],
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"text": "How many white items are visible in this image?"},
                            {
                                "image": {
                                    "format": "webp",  # This format is allowed for SFT 1.0 but not SFT 2.0
                                    "source": {
                                        "s3Location": {
                                            "uri": "s3://your-bucket/your-path/your-img.webp",
                                            "bucketOwner": "your-aws-account-id",
                                        }
                                    },
                                }
                            },
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "text": "There are at least eight visible items that are white"
                            }
                        ],
                    },
                ],
            }
        ]

        test_file = self.create_temp_file(
            "sft_two_img_invalid_format", sft_two_img_invalid_format
        )

        # Works for SFT 1.0
        JSONLDatasetLoader().load(test_file).validate(
            TrainingMethod.SFT_LORA, Model.NOVA_LITE
        )

        # Doesn't work for SFT 2.0
        with self.assertRaises(ValueError):
            JSONLDatasetLoader().load(test_file).validate(
                TrainingMethod.SFT_LORA, Model.NOVA_LITE_2
            )

    # Tool Configuration Tests
    def test_tool_config_success_with_nova_lite_2(self):
        """Test valid tool configuration with Nova Lite 2.0"""
        tool_config_success = [
            {
                "schemaVersion": "bedrock-conversation-2024",
                "system": [{"text": "You are an expert in composing function calls."}],
                "toolConfig": {
                    "tools": [
                        {
                            "toolSpec": {
                                "name": "getWeather",
                                "description": "Get weather information for a location",
                                "inputSchema": {
                                    "json": {
                                        "type": "object",
                                        "properties": {"location": {"type": "string"}},
                                        "required": ["location"],
                                    }
                                },
                            }
                        }
                    ]
                },
                "messages": [
                    {
                        "role": "user",
                        "content": [{"text": "What's the weather in Paris?"}],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "toolUse": {
                                    "toolUseId": "weather_1",
                                    "name": "getWeather",
                                    "input": {"location": "Paris"},
                                }
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "toolResult": {
                                    "toolUseId": "weather_1",
                                    "content": [{"text": "Sunny, 22°C"}],
                                }
                            }
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "text": "The weather in Paris is sunny with a temperature of 22°C."
                            }
                        ],
                    },
                ],
            }
        ]

        test_file = self.create_temp_file("tool_config_success", tool_config_success)
        JSONLDatasetLoader().load(test_file).validate(
            TrainingMethod.SFT_LORA, Model.NOVA_LITE_2
        )

    def test_tool_config_fail_with_nova_lite_1(self):
        """Test that tool configuration fails with Nova Lite 1.0"""
        tool_config_fail = [
            {
                "schemaVersion": "bedrock-conversation-2024",
                "toolConfig": {
                    "tools": [
                        {
                            "toolSpec": {
                                "name": "getWeather",
                                "description": "Get weather information",
                                "inputSchema": {"json": {"type": "object"}},
                            }
                        }
                    ]
                },
                "messages": [
                    {"role": "user", "content": [{"text": "Hello"}]},
                    {"role": "assistant", "content": [{"text": "Hi"}]},
                ],
            }
        ]

        test_file = self.create_temp_file("tool_config_fail_nova_1", tool_config_fail)
        with self.assertRaises(ValueError) as context:
            JSONLDatasetLoader().load(test_file).validate(
                TrainingMethod.SFT_LORA, Model.NOVA_LITE
            )
        self.assertIn(
            "Tool configuration is only supported for Nova Lite 2.0",
            str(context.exception),
        )

    def test_tool_config_empty_tools_list_fail(self):
        """Test that empty tools list fails validation"""
        empty_tools = [
            {
                "schemaVersion": "bedrock-conversation-2024",
                "toolConfig": {"tools": []},
                "messages": [
                    {"role": "user", "content": [{"text": "Hello"}]},
                    {"role": "assistant", "content": [{"text": "Hi"}]},
                ],
            }
        ]

        test_file = self.create_temp_file("empty_tools", empty_tools)
        with self.assertRaises(ValueError) as context:
            JSONLDatasetLoader().load(test_file).validate(
                TrainingMethod.SFT_LORA, Model.NOVA_LITE_2
            )
        self.assertIn("tools list cannot be empty", str(context.exception))

    def test_tool_config_duplicate_names_fail(self):
        """Test that duplicate tool names fail validation"""
        duplicate_tools = [
            {
                "schemaVersion": "bedrock-conversation-2024",
                "toolConfig": {
                    "tools": [
                        {
                            "toolSpec": {
                                "name": "getTool",
                                "description": "First tool",
                                "inputSchema": {"json": {"type": "object"}},
                            }
                        },
                        {
                            "toolSpec": {
                                "name": "getTool",
                                "description": "Duplicate name",
                                "inputSchema": {"json": {"type": "object"}},
                            }
                        },
                    ]
                },
                "messages": [
                    {"role": "user", "content": [{"text": "Hello"}]},
                    {"role": "assistant", "content": [{"text": "Hi"}]},
                ],
            }
        ]

        test_file = self.create_temp_file("duplicate_tools", duplicate_tools)
        with self.assertRaises(ValueError) as context:
            JSONLDatasetLoader().load(test_file).validate(
                TrainingMethod.SFT_LORA, Model.NOVA_LITE_2
            )
        self.assertIn("duplicate tool names found", str(context.exception))

    def test_tool_use_in_user_message_fail(self):
        """Test that toolUse in user message fails validation"""
        tool_use_wrong_role = [
            {
                "schemaVersion": "bedrock-conversation-2024",
                "toolConfig": {
                    "tools": [
                        {
                            "toolSpec": {
                                "name": "getTool",
                                "description": "A tool",
                                "inputSchema": {"json": {"type": "object"}},
                            }
                        }
                    ]
                },
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "toolUse": {
                                    "toolUseId": "tool_1",
                                    "name": "getTool",
                                    "input": {},
                                }
                            }
                        ],
                    },
                    {"role": "assistant", "content": [{"text": "Hi"}]},
                ],
            }
        ]

        test_file = self.create_temp_file("tool_use_wrong_role", tool_use_wrong_role)
        with self.assertRaises(ValueError) as context:
            JSONLDatasetLoader().load(test_file).validate(
                TrainingMethod.SFT_LORA, Model.NOVA_LITE_2
            )
        self.assertIn(
            "toolUse can only be included in assistant messages", str(context.exception)
        )

    def test_tool_result_in_assistant_message_fail(self):
        """Test that toolResult in assistant message fails validation"""
        tool_result_wrong_role = [
            {
                "schemaVersion": "bedrock-conversation-2024",
                "toolConfig": {
                    "tools": [
                        {
                            "toolSpec": {
                                "name": "getTool",
                                "description": "A tool",
                                "inputSchema": {"json": {"type": "object"}},
                            }
                        }
                    ]
                },
                "messages": [
                    {"role": "user", "content": [{"text": "Hello"}]},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "toolResult": {
                                    "toolUseId": "tool_1",
                                    "content": [{"text": "Result"}],
                                }
                            }
                        ],
                    },
                ],
            }
        ]

        test_file = self.create_temp_file(
            "tool_result_wrong_role", tool_result_wrong_role
        )
        with self.assertRaises(ValueError) as context:
            JSONLDatasetLoader().load(test_file).validate(
                TrainingMethod.SFT_LORA, Model.NOVA_LITE_2
            )
        self.assertIn(
            "toolResult can only be included in user messages", str(context.exception)
        )

    def test_tool_result_invalid_id_fail(self):
        """Test that tool result with non-existent toolUseId fails"""
        invalid_tool_id = [
            {
                "schemaVersion": "bedrock-conversation-2024",
                "toolConfig": {
                    "tools": [
                        {
                            "toolSpec": {
                                "name": "getTool",
                                "description": "A tool",
                                "inputSchema": {"json": {"type": "object"}},
                            }
                        }
                    ]
                },
                "messages": [
                    {"role": "user", "content": [{"text": "Hello"}]},
                    {"role": "assistant", "content": [{"text": "Let me help"}]},
                    {
                        "role": "user",
                        "content": [
                            {
                                "toolResult": {
                                    "toolUseId": "non_existent_id",
                                    "content": [{"text": "Result"}],
                                }
                            }
                        ],
                    },
                    {"role": "assistant", "content": [{"text": "Done"}]},
                ],
            }
        ]

        test_file = self.create_temp_file("invalid_tool_id", invalid_tool_id)
        with self.assertRaises(ValueError) as context:
            JSONLDatasetLoader().load(test_file).validate(
                TrainingMethod.SFT_LORA, Model.NOVA_LITE_2
            )
        self.assertIn("not found in any preceding toolUse", str(context.exception))

    def test_tool_name_not_in_config_fail(self):
        """Test that using a tool not in configuration fails"""
        tool_not_in_config = [
            {
                "schemaVersion": "bedrock-conversation-2024",
                "toolConfig": {
                    "tools": [
                        {
                            "toolSpec": {
                                "name": "allowedTool",
                                "description": "Allowed tool",
                                "inputSchema": {"json": {"type": "object"}},
                            }
                        }
                    ]
                },
                "messages": [
                    {"role": "user", "content": [{"text": "Hello"}]},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "toolUse": {
                                    "toolUseId": "tool_1",
                                    "name": "notAllowedTool",
                                    "input": {},
                                }
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "toolResult": {
                                    "toolUseId": "tool_1",
                                    "content": [{"text": "Result"}],
                                }
                            }
                        ],
                    },
                    {"role": "assistant", "content": [{"text": "Done"}]},
                ],
            }
        ]

        test_file = self.create_temp_file("tool_not_in_config", tool_not_in_config)
        with self.assertRaises(ValueError) as context:
            JSONLDatasetLoader().load(test_file).validate(
                TrainingMethod.SFT_LORA, Model.NOVA_LITE_2
            )
        self.assertIn("not found in toolConfig", str(context.exception))

    def test_tool_result_with_json_content_success(self):
        """Test that tool result with JSON content is valid"""
        tool_result_json = [
            {
                "schemaVersion": "bedrock-conversation-2024",
                "toolConfig": {
                    "tools": [
                        {
                            "toolSpec": {
                                "name": "getInfo",
                                "description": "Get information",
                                "inputSchema": {"json": {"type": "object"}},
                            }
                        }
                    ]
                },
                "messages": [
                    {"role": "user", "content": [{"text": "Get info"}]},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "toolUse": {
                                    "toolUseId": "info_1",
                                    "name": "getInfo",
                                    "input": {},
                                }
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "toolResult": {
                                    "toolUseId": "info_1",
                                    "content": [
                                        {"json": {"status": "success", "data": "info"}}
                                    ],
                                }
                            }
                        ],
                    },
                    {"role": "assistant", "content": [{"text": "Got the info"}]},
                ],
            }
        ]

        test_file = self.create_temp_file("tool_result_json", tool_result_json)
        JSONLDatasetLoader().load(test_file).validate(
            TrainingMethod.SFT_LORA, Model.NOVA_LITE_2
        )

    def test_tool_with_reasoning_content_success(self):
        """Test that tool use with reasoning content in separate ContentItems is valid"""
        tool_with_reasoning = [
            {
                "schemaVersion": "bedrock-conversation-2024",
                "toolConfig": {
                    "tools": [
                        {
                            "toolSpec": {
                                "name": "calculate",
                                "description": "Perform calculation",
                                "inputSchema": {"json": {"type": "object"}},
                            }
                        }
                    ]
                },
                "messages": [
                    {"role": "user", "content": [{"text": "Calculate 2+2"}]},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "reasoningContent": {
                                    "reasoningText": {"text": "I need to add 2+2"}
                                }
                            },
                            {
                                "toolUse": {
                                    "toolUseId": "calc_1",
                                    "name": "calculate",
                                    "input": {"operation": "add", "a": 2, "b": 2},
                                }
                            },
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "toolResult": {
                                    "toolUseId": "calc_1",
                                    "content": [{"text": "4"}],
                                }
                            }
                        ],
                    },
                    {"role": "assistant", "content": [{"text": "2+2 equals 4"}]},
                ],
            }
        ]

        test_file = self.create_temp_file("tool_with_reasoning", tool_with_reasoning)
        JSONLDatasetLoader().load(test_file).validate(
            TrainingMethod.SFT_LORA, Model.NOVA_LITE_2
        )

    def test_tool_use_and_result_in_same_content_item_fail(self):
        """Test that toolUse and toolResult in the same ContentItem fails"""
        tool_same_item = [
            {
                "schemaVersion": "bedrock-conversation-2024",
                "toolConfig": {
                    "tools": [
                        {
                            "toolSpec": {
                                "name": "getTool",
                                "description": "A tool",
                                "inputSchema": {"json": {"type": "object"}},
                            }
                        }
                    ]
                },
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "toolUse": {
                                    "toolUseId": "tool_1",
                                    "name": "getTool",
                                    "input": {},
                                },
                                "toolResult": {
                                    "toolUseId": "tool_1",
                                    "content": [{"text": "Result"}],
                                },
                            }
                        ],
                    },
                    {"role": "assistant", "content": [{"text": "Done"}]},
                ],
            }
        ]

        test_file = self.create_temp_file("tool_same_item", tool_same_item)
        with self.assertRaises(ValueError) as context:
            JSONLDatasetLoader().load(test_file).validate(
                TrainingMethod.SFT_LORA, Model.NOVA_LITE_2
            )
        self.assertIn("cannot coexist in the same ContentItem", str(context.exception))

    def tearDown(self):
        """Clean up temporary files created during each unit test."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)
