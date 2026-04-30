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
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Mock pymediainfo at module level so all tests use the mock
sys.modules.pop("pymediainfo", None)
_mock_mediainfo_cls = MagicMock()
_mock_video_track = MagicMock()
_mock_video_track.duration = 5000  # 5 seconds in ms
_mock_video_track.track_type = "Video"
_mock_image_track = MagicMock()
_mock_image_track.track_type = "Image"
_mock_image_track.width = 100
_mock_image_track.height = 100
_mock_image_track.format = "PNG"
_mock_mediainfo_cls.parse.return_value = MagicMock(tracks=[_mock_video_track, _mock_image_track])
_mock_pymediainfo = MagicMock()
_mock_pymediainfo.MediaInfo = _mock_mediainfo_cls
sys.modules["pymediainfo"] = _mock_pymediainfo

from amzn_nova_forge.core.enums import Model, TrainingMethod
from amzn_nova_forge.dataset.dataset_loader import (
    JSONLDatasetLoader,
)

# TODO: Clean up these unit tests to use helper functions to set up the datasets for readability.


class TestSFTDatasetValidator(unittest.TestCase):
    def setUp(self):
        """Create a temporary directory before each test for storing test data."""
        self.temp_dir = tempfile.mkdtemp()
        # Mock boto3 S3 client so multimodal file size validation doesn't hit real S3
        self._boto3_patcher = patch(
            "amzn_nova_forge.dataset.dataset_validator.dataset_validator.boto3"
        )
        mock_boto3 = self._boto3_patcher.start()
        mock_boto3.client.return_value = _mock_s3()

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

        test_file = self.create_temp_file("nova_micro_image_fail", nova_micro_image_fail)
        with self.assertRaises(ValueError) as context:
            JSONLDatasetLoader().load(test_file).validate(
                training_method=TrainingMethod.SFT_LORA, model=Model.NOVA_MICRO
            )

    def test_sft_one_with_text_data_success(self):
        dataset_loader = JSONLDatasetLoader()
        dataset_loader.load(path="tests/test_data/sft_train_samples_converse.jsonl")

        dataset_loader.validate(training_method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE)

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
            dataset_loader.validate(training_method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE)

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
                            {"text": "There are at least eight visible items that are white"}
                        ],
                    },
                ],
            }
        ]

        test_file = self.create_temp_file("sft_one_video_success", sft_one_video_success)
        JSONLDatasetLoader().load(test_file).validate(
            training_method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE
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
                            {"text": "There are at least eight visible items that are white"}
                        ],
                    },
                ],
            }
        ]

        test_file = self.create_temp_file("sft_one_video_fail", sft_one_video_fail)

        dataset_loader = JSONLDatasetLoader().load(test_file)
        with self.assertRaises(ValueError):
            dataset_loader.validate(training_method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE)

    def test_sft_one_with_image_success(self):
        sft_one_image_success = [
            {
                "schemaVersion": "bedrock-conversation-2024",
                "system": [
                    {"text": "You are a smart assistant that answers questions respectfully"}
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
                        "content": [{"text": "The text in the attached image says 'LOL'."}],
                    },
                ],
            }
        ]

        test_file = self.create_temp_file("sft_one_image_success", sft_one_image_success)
        JSONLDatasetLoader().load(test_file).validate(
            training_method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE
        )

    def test_sft_one_with_image_fail(self):
        sft_one_image_fail = [
            {
                "schemaVersion": "bedrock-conversation-2024",
                "system": [
                    {"text": "You are a smart assistant that answers questions respectfully"}
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
                        "content": [{"text": "The text in the attached image says 'LOL'."}],
                    },
                ],
            }
        ]

        test_file = self.create_temp_file("sft_one_image_fail", sft_one_image_fail)

        dataset_loader = JSONLDatasetLoader().load(test_file)
        with self.assertRaises(ValueError):
            dataset_loader.validate(training_method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE)

    def test_sft_one_with_image_no_source_fail(self):
        """Test that image with no source fails for SFT 1.0"""
        sft_image_no_source = [
            {
                "schemaVersion": "bedrock-conversation-2024",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"text": "What does the text in this image say?"},
                            {
                                "image": {
                                    "format": "png",
                                    "source": {},
                                }
                            },
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"text": "Response"}],
                    },
                ],
            }
        ]

        test_file = self.create_temp_file("sft_image_no_source", sft_image_no_source)
        with self.assertRaises(ValueError):
            JSONLDatasetLoader().load(test_file).validate(
                training_method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE
            )

    def test_sft_one_mixing_image_and_video_fail(self):
        sft_one_img_vid_fail = [
            {
                "schemaVersion": "bedrock-conversation-2024",
                "system": [
                    {"text": "You are a smart assistant that answers questions respectfully"}
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
                        "content": [{"text": "The text in the attached image says 'LOL'."}],
                    },
                ],
            }
        ]

        test_file = self.create_temp_file("sft_one_img_vid_fail", sft_one_img_vid_fail)
        dataset_loader = JSONLDatasetLoader().load(test_file)
        with self.assertRaises(ValueError):
            dataset_loader.validate(training_method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE)

    def test_sft_one_reasoning_content_fail(self):
        sft_one_reasoning_content_fail = [
            {
                "schemaVersion": "bedrock-conversation-2024",
                "system": [{"text": "You are a digital assistant with a friendly personality"}],
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
                training_method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE
            )

    # SFT 2.0 Tests (Reasoning Content)
    def test_sft_two_success(self):
        sft_two_success = [
            {
                "schemaVersion": "bedrock-conversation-2024",
                "system": [{"text": "You are a digital assistant with a friendly personality"}],
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
            training_method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE_2
        )

    def test_sft_two_fail(self):
        test_sft_two_fail = [
            {
                "schemaVersion": "bedrock-conversation-2024",
                "system": [{"text": "You are a digital assistant with a friendly personality"}],
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
                training_method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE_2
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
                    },
                ],
            }
        ]
        test_file = self.create_temp_file("test_sft_two_doc_success", test_sft_two_doc_success)
        JSONLDatasetLoader().load(test_file).validate(
            training_method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE_2
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
                    },
                ],
            }
        ]
        test_file = self.create_temp_file("test_sft_two_doc_fail", test_sft_two_doc_fail)
        with self.assertRaises(ValueError):
            JSONLDatasetLoader().load(test_file).validate(
                training_method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE_2
            )

    def test_sft_one_doc_fail(self):
        """Test that document content is rejected for Nova 1.0 models"""
        sft_one_doc_fail = [
            {
                "schemaVersion": "bedrock-conversation-2024",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"text": "Summarize this document"},
                            {
                                "document": {
                                    "format": "pdf",
                                    "source": {
                                        "s3Location": {
                                            "uri": "s3://bucket/doc.pdf",
                                            "bucketOwner": "123456789012",
                                        }
                                    },
                                }
                            },
                        ],
                    },
                    {"role": "assistant", "content": [{"text": "Summary"}]},
                ],
            }
        ]

        test_file = self.create_temp_file("sft_one_doc_fail", sft_one_doc_fail)
        with self.assertRaises(ValueError) as context:
            JSONLDatasetLoader().load(test_file).validate(
                training_method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE
            )
        self.assertIn("Invalid format", str(context.exception))

    def test_sft_with_webm_video_format_succeeds(self):
        sft_webm_video = [
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
                                    "format": "webm",
                                    "source": {
                                        "s3Location": {
                                            "uri": "s3://your-bucket/your-path/your-video.webm",
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
                            {"text": "There are at least eight visible items that are white"}
                        ],
                    },
                ],
            }
        ]

        test_file = self.create_temp_file("sft_webm_video", sft_webm_video)

        # webm is valid for both SFT 1.0 and SFT 2.0
        JSONLDatasetLoader().load(test_file).validate(
            training_method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE
        )
        JSONLDatasetLoader().load(test_file).validate(
            training_method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE_2
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
                                    "format": "tiff",  # tiff is not supported for either SFT 1.0 or 2.0
                                    "source": {
                                        "s3Location": {
                                            "uri": "s3://your-bucket/your-path/your-img.tiff",
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
                            {"text": "There are at least eight visible items that are white"}
                        ],
                    },
                ],
            }
        ]

        test_file = self.create_temp_file("sft_two_img_invalid_format", sft_two_img_invalid_format)

        # Doesn't work for SFT 1.0
        with self.assertRaises(ValueError):
            JSONLDatasetLoader().load(test_file).validate(
                training_method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE
            )

        # Doesn't work for SFT 2.0
        with self.assertRaises(ValueError):
            JSONLDatasetLoader().load(test_file).validate(
                training_method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE_2
            )

    def test_sft_with_webp_img_format_succeeds(self):
        """Test that webp format is now accepted for both SFT 1.0 and 2.0"""
        sft_webp_img = [
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
                                    "format": "webp",
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
                            {"text": "There are at least eight visible items that are white"}
                        ],
                    },
                ],
            }
        ]

        test_file = self.create_temp_file("sft_webp_img", sft_webp_img)

        # Works for SFT 1.0
        JSONLDatasetLoader().load(test_file).validate(
            training_method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE
        )

        # Now also works for SFT 2.0
        JSONLDatasetLoader().load(test_file).validate(
            training_method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE_2
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
                            {"text": "The weather in Paris is sunny with a temperature of 22°C."}
                        ],
                    },
                ],
            }
        ]

        test_file = self.create_temp_file("tool_config_success", tool_config_success)
        JSONLDatasetLoader().load(test_file).validate(
            training_method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE_2
        )

    def test_tool_config_success_with_nova_lite_1(self):
        """Test that tool configuration is accepted with Nova Lite 1.0"""
        tool_config_success = [
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

        test_file = self.create_temp_file("tool_config_success_nova_1", tool_config_success)
        JSONLDatasetLoader().load(test_file).validate(
            training_method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE
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
                training_method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE_2
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
                training_method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE_2
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
                training_method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE_2
            )
        self.assertIn("toolUse can only be included in assistant messages", str(context.exception))

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

        test_file = self.create_temp_file("tool_result_wrong_role", tool_result_wrong_role)
        with self.assertRaises(ValueError) as context:
            JSONLDatasetLoader().load(test_file).validate(
                training_method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE_2
            )
        self.assertIn("toolResult can only be included in user messages", str(context.exception))

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
                training_method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE_2
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
                training_method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE_2
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
                                    "content": [{"json": {"status": "success", "data": "info"}}],
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
            training_method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE_2
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
                            {"reasoningContent": {"reasoningText": {"text": "I need to add 2+2"}}},
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
            training_method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE_2
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
                training_method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE_2
            )
        self.assertIn("cannot coexist in the same ContentItem", str(context.exception))

    def test_sft_rejects_extra_fields_in_sample(self):
        """Test that unrecognized fields at sample level are rejected."""
        sft_extra_field = [
            {
                "schemaVersion": "bedrock-conversation-2024",
                "messages": [
                    {"role": "user", "content": [{"text": "Hello"}]},
                    {"role": "assistant", "content": [{"text": "Hi"}]},
                ],
                "unknown_field": "should be rejected",
            }
        ]

        test_file = self.create_temp_file("sft_extra_field", sft_extra_field)
        with self.assertRaises(ValueError) as context:
            JSONLDatasetLoader().load(test_file).validate(
                training_method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE_2
            )
        self.assertIn("extra", str(context.exception).lower())

    def test_sft_rejects_extra_fields_in_message(self):
        """Test that unrecognized fields at message level are rejected."""
        sft_extra_msg_field = [
            {
                "schemaVersion": "bedrock-conversation-2024",
                "messages": [
                    {
                        "role": "user",
                        "content": [{"text": "Hello"}],
                        "metadata": "should be rejected",
                    },
                    {"role": "assistant", "content": [{"text": "Hi"}]},
                ],
            }
        ]

        test_file = self.create_temp_file("sft_extra_msg_field", sft_extra_msg_field)
        with self.assertRaises(ValueError) as context:
            JSONLDatasetLoader().load(test_file).validate(
                training_method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE_2
            )
        self.assertIn("extra", str(context.exception).lower())

    def tearDown(self):
        """Clean up temporary files created during each unit test."""
        import shutil

        self._boto3_patcher.stop()
        shutil.rmtree(self.temp_dir, ignore_errors=True)


from unittest.mock import MagicMock

from pydantic import ValidationError as PydanticValidationError

from amzn_nova_forge.dataset.configs.dataset_checks_config import (
    MAX_IMAGE_FILE_SIZE_BYTES,
    MAX_IMAGES_PER_MESSAGE,
    MAX_VIDEO_DURATION_SECONDS,
    MAX_VIDEO_FILE_SIZE_BYTES,
    MAX_VIDEOS_PER_MESSAGE,
)
from amzn_nova_forge.dataset.dataset_validator.dataset_validator import (
    InfrastructureError,
)
from amzn_nova_forge.dataset.dataset_validator.sft_dataset_validator import (
    ImageContent,
    SFTConverseDatasetSample,
    VideoContent,
)


def _image_content_dict(uri="s3://bucket/img.png"):
    return {
        "format": "png",
        "source": {"s3Location": {"uri": uri, "bucketOwner": "123456789012"}},
    }


def _video_content_dict(uri="s3://bucket/vid.mp4"):
    return {
        "format": "mp4",
        "source": {"s3Location": {"uri": uri, "bucketOwner": "123456789012"}},
    }


_MAGIC_BYTES = {
    "png": b"\x89PNG\r\n\x1a\n" + b"\x00" * 254,
    "jpg": b"\xff\xd8\xff\xe0" + b"\x00" * 258,
    "jpeg": b"\xff\xd8\xff\xe0" + b"\x00" * 258,
    "gif": b"GIF89a" + b"\x00" * 256,
    "webp": b"RIFF\x24\x00\x00\x00WEBPVP8 " + b"\x00" * 248,
    "mp4": b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isomiso2mp41" + b"\x00" * 230,
    "mov": b"\x00\x00\x00\x14ftypqt  \x00\x00\x00\x00qt  " + b"\x00" * 248,
    "mkv": b"\x1a\x45\xdf\xa3\x93\x42\x82\x88matroska" + b"\x00" * 248,
    "webm": b"\x1a\x45\xdf\xa3\x93\x42\x82\x84webm" + b"\x00" * 250,
    "pdf": b"%PDF-1.4" + b"\x00" * 254,
}


def _mock_s3(content_length=1024):
    """Create a mock S3 client with head_object and get_object (magic bytes)."""
    s3 = MagicMock()
    s3.head_object.return_value = {"ContentLength": content_length}

    def _get_object(**kwargs):
        key = kwargs.get("Key", "")
        ext = key.rsplit(".", 1)[-1].lower() if "." in key else ""
        body = MagicMock()
        body.read.return_value = _MAGIC_BYTES.get(ext, b"\x00" * 16)
        return {"Body": body}

    s3.get_object.side_effect = _get_object
    return s3


def _make_sample_with_images(count):
    images = [{"image": _image_content_dict(f"s3://bucket/img{i}.png")} for i in range(count)]
    return {
        "schemaVersion": "bedrock-conversation-2024",
        "messages": [
            {"role": "user", "content": [{"text": "Describe"}] + images},
            {"role": "assistant", "content": [{"text": "Done."}]},
        ],
    }


def _make_sample_with_videos(count):
    videos = [{"video": _video_content_dict(f"s3://bucket/vid{i}.mp4")} for i in range(count)]
    return {
        "schemaVersion": "bedrock-conversation-2024",
        "messages": [
            {"role": "user", "content": [{"text": "Describe"}] + videos},
            {"role": "assistant", "content": [{"text": "Done."}]},
        ],
    }


class TestImageFileSizeValidation(unittest.TestCase):
    def setUp(self):
        sys.modules["pymediainfo"].MediaInfo.parse.return_value = MagicMock(
            tracks=[_mock_video_track, _mock_image_track]
        )

    def test_image_under_limit_passes(self):
        s3 = _mock_s3(5 * 1024 * 1024)
        context = {"model": Model.NOVA_LITE_2, "s3_client": s3}
        ImageContent.model_validate(_image_content_dict(), context=context)

    def test_image_at_limit_passes(self):
        s3 = _mock_s3(MAX_IMAGE_FILE_SIZE_BYTES)
        context = {"model": Model.NOVA_LITE_2, "s3_client": s3}
        ImageContent.model_validate(_image_content_dict(), context=context)

    def test_image_over_limit_fails(self):
        s3 = _mock_s3(MAX_IMAGE_FILE_SIZE_BYTES + 1)
        context = {"model": Model.NOVA_LITE_2, "s3_client": s3}
        with self.assertRaises(PydanticValidationError):
            ImageContent.model_validate(_image_content_dict(), context=context)

    def test_no_s3_client_in_context_skips(self):
        context = {"model": Model.NOVA_LITE_2}
        with self.assertRaises(InfrastructureError):
            ImageContent.model_validate(_image_content_dict(), context=context)

    def test_s3_error_raises(self):
        s3 = _mock_s3()
        s3.head_object.side_effect = Exception("Access Denied")
        context = {"model": Model.NOVA_LITE_2, "s3_client": s3}
        with self.assertRaises(InfrastructureError):
            ImageContent.model_validate(_image_content_dict(), context=context)


class TestVideoFileSizeValidation(unittest.TestCase):
    def test_video_under_limit_passes(self):
        s3 = _mock_s3(20 * 1024 * 1024)
        context = {"model": Model.NOVA_LITE_2, "s3_client": s3}
        VideoContent.model_validate(_video_content_dict(), context=context)

    def test_video_at_limit_passes(self):
        s3 = _mock_s3(MAX_VIDEO_FILE_SIZE_BYTES)
        context = {"model": Model.NOVA_LITE_2, "s3_client": s3}
        VideoContent.model_validate(_video_content_dict(), context=context)

    def test_video_over_limit_fails(self):
        s3 = _mock_s3(MAX_VIDEO_FILE_SIZE_BYTES + 1)
        context = {"model": Model.NOVA_LITE_2, "s3_client": s3}
        with self.assertRaises(PydanticValidationError):
            VideoContent.model_validate(_video_content_dict(), context=context)

    def test_no_s3_client_in_context_skips(self):
        context = {"model": Model.NOVA_LITE_2}
        with self.assertRaises(InfrastructureError):
            VideoContent.model_validate(_video_content_dict(), context=context)

    def test_s3_error_raises(self):
        s3 = _mock_s3()
        s3.head_object.side_effect = Exception("Access Denied")
        context = {"model": Model.NOVA_LITE_2, "s3_client": s3}
        with self.assertRaises(InfrastructureError):
            VideoContent.model_validate(_video_content_dict(), context=context)


class TestVideoDurationValidation(unittest.TestCase):
    def tearDown(self):
        sys.modules["pymediainfo"].MediaInfo.parse.return_value = MagicMock(
            tracks=[_mock_video_track, _mock_image_track]
        )

    def _make_context(self):
        s3 = _mock_s3()
        return {"model": Model.NOVA_LITE_2, "s3_client": s3}

    def test_under_limit_passes(self):
        mock_track = MagicMock()
        mock_track.duration = (MAX_VIDEO_DURATION_SECONDS - 1) * 1000
        mock_track.track_type = "Video"
        sys.modules["pymediainfo"].MediaInfo.parse.return_value = MagicMock(tracks=[mock_track])
        VideoContent.model_validate(_video_content_dict(), context=self._make_context())

    def test_at_limit_passes(self):
        mock_track = MagicMock()
        mock_track.duration = MAX_VIDEO_DURATION_SECONDS * 1000
        mock_track.track_type = "Video"
        sys.modules["pymediainfo"].MediaInfo.parse.return_value = MagicMock(tracks=[mock_track])
        VideoContent.model_validate(_video_content_dict(), context=self._make_context())

    def test_over_limit_fails(self):
        mock_track = MagicMock()
        mock_track.duration = (MAX_VIDEO_DURATION_SECONDS + 1) * 1000
        mock_track.track_type = "Video"
        sys.modules["pymediainfo"].MediaInfo.parse.return_value = MagicMock(tracks=[mock_track])
        with self.assertRaises(PydanticValidationError):
            VideoContent.model_validate(_video_content_dict(), context=self._make_context())

    def test_none_duration_raises(self):
        mock_track = MagicMock()
        mock_track.duration = None
        mock_track.track_type = "Video"
        sys.modules["pymediainfo"].MediaInfo.parse.return_value = MagicMock(tracks=[mock_track])
        with self.assertRaises(InfrastructureError):
            VideoContent.model_validate(_video_content_dict(), context=self._make_context())

    def test_no_s3_client_skips(self):
        context = {"model": Model.NOVA_LITE_2}
        with self.assertRaises(InfrastructureError):
            VideoContent.model_validate(_video_content_dict(), context=context)


class TestImageCountValidation(unittest.TestCase):
    def setUp(self):
        # Restore default pymediainfo mock (TestVideoDurationValidation may override it)
        sys.modules["pymediainfo"].MediaInfo.parse.return_value = MagicMock(
            tracks=[_mock_video_track, _mock_image_track]
        )

    def _context(self):
        return {"model": Model.NOVA_LITE_2, "s3_client": _mock_s3()}

    def test_under_limit_passes(self):
        sample = _make_sample_with_images(5)
        SFTConverseDatasetSample.model_validate(sample, context=self._context())

    def test_at_limit_passes(self):
        sample = _make_sample_with_images(MAX_IMAGES_PER_MESSAGE)
        SFTConverseDatasetSample.model_validate(sample, context=self._context())

    def test_over_limit_fails(self):
        sample = _make_sample_with_images(MAX_IMAGES_PER_MESSAGE + 1)
        with self.assertRaises(PydanticValidationError):
            SFTConverseDatasetSample.model_validate(sample, context=self._context())


class TestVideoCountValidation(unittest.TestCase):
    def _context(self):
        return {"model": Model.NOVA_LITE_2, "s3_client": _mock_s3()}

    def test_one_video_passes(self):
        sample = _make_sample_with_videos(1)
        SFTConverseDatasetSample.model_validate(sample, context=self._context())

    def test_over_limit_fails(self):
        sample = _make_sample_with_videos(MAX_VIDEOS_PER_MESSAGE + 1)
        with self.assertRaises(PydanticValidationError):
            SFTConverseDatasetSample.model_validate(sample, context=self._context())


class TestImageFormatConsistency(unittest.TestCase):
    """Magic number check in format_allowlist executor."""

    def test_matching_format_passes(self):
        s3 = _mock_s3()
        context = {"model": Model.NOVA_LITE_2, "s3_client": s3}
        ImageContent.model_validate(_image_content_dict(), context=context)

    def test_mismatched_format_fails(self):
        s3 = _mock_s3()
        # Override get_object to return JPEG magic bytes for a .png URI
        body = MagicMock()
        body.read.return_value = b"\xff\xd8\xff\xe0" + b"\x00" * 258
        s3.get_object.side_effect = None
        s3.get_object.return_value = {"Body": body}
        context = {"model": Model.NOVA_LITE_2, "s3_client": s3}
        with self.assertRaises(PydanticValidationError) as ctx:
            ImageContent.model_validate(_image_content_dict(), context=context)
        self.assertIn("does not match", str(ctx.exception))

    def test_no_s3_client_skips(self):
        context = {"model": Model.NOVA_LITE_2}
        with self.assertRaises(InfrastructureError):
            ImageContent.model_validate(_image_content_dict(), context=context)

    def test_s3_get_object_error_raises_infrastructure(self):
        s3 = _mock_s3()
        s3.get_object.side_effect = Exception("Access Denied")
        context = {"model": Model.NOVA_LITE_2, "s3_client": s3}
        with self.assertRaises(InfrastructureError):
            ImageContent.model_validate(_image_content_dict(), context=context)


class TestImageDimensions(unittest.TestCase):
    """Image dimensions check via pymediainfo."""

    def setUp(self):
        sys.modules["pymediainfo"].MediaInfo.parse.return_value = MagicMock(
            tracks=[_mock_video_track, _mock_image_track]
        )

    def test_valid_dimensions_passes(self):
        _mock_image_track.width = 100
        _mock_image_track.height = 100
        s3 = _mock_s3()
        context = {"model": Model.NOVA_LITE_2, "s3_client": s3}
        ImageContent.model_validate(_image_content_dict(), context=context)

    def test_below_min_width_fails(self):
        _mock_image_track.width = 10
        _mock_image_track.height = 100
        s3 = _mock_s3()
        context = {"model": Model.NOVA_LITE_2, "s3_client": s3}
        with self.assertRaises(PydanticValidationError) as ctx:
            ImageContent.model_validate(_image_content_dict(), context=context)
        self.assertIn("below minimum", str(ctx.exception))

    def test_below_min_height_fails(self):
        _mock_image_track.width = 100
        _mock_image_track.height = 10
        s3 = _mock_s3()
        context = {"model": Model.NOVA_LITE_2, "s3_client": s3}
        with self.assertRaises(PydanticValidationError) as ctx:
            ImageContent.model_validate(_image_content_dict(), context=context)
        self.assertIn("below minimum", str(ctx.exception))

    def test_at_min_passes(self):
        _mock_image_track.width = 25
        _mock_image_track.height = 25
        s3 = _mock_s3()
        context = {"model": Model.NOVA_LITE_2, "s3_client": s3}
        ImageContent.model_validate(_image_content_dict(), context=context)

    def test_no_s3_client_skips(self):
        context = {"model": Model.NOVA_LITE_2}
        with self.assertRaises(InfrastructureError):
            ImageContent.model_validate(_image_content_dict(), context=context)

    def test_no_tracks_raises_infrastructure(self):
        sys.modules["pymediainfo"].MediaInfo.parse.return_value = MagicMock(tracks=[])
        s3 = _mock_s3()
        context = {"model": Model.NOVA_LITE_2, "s3_client": s3}
        with self.assertRaises(InfrastructureError):
            ImageContent.model_validate(_image_content_dict(), context=context)

    def tearDown(self):
        _mock_image_track.width = 100
        _mock_image_track.height = 100
        sys.modules["pymediainfo"].MediaInfo.parse.return_value = MagicMock(
            tracks=[_mock_video_track, _mock_image_track]
        )
