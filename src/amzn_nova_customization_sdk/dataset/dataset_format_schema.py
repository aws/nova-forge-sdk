# Copyright 2025 Amazon Inc

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
SFT_NOVA_ONE_CONVERSE_2024 = {
    "type": "object",
    "properties": {
        "schemaVersion": {"type": "string", "const": "bedrock-conversation-2024"},
        "system": {
            "type": "array",
            "maxItems": 1,
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
        },
        "messages": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "role": {"type": "string", "enum": ["user", "assistant"]},
                    "content": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "oneOf": [
                                # Text content
                                {
                                    "properties": {"text": {"type": "string"}},
                                    "required": ["text"],
                                    "additionalProperties": False,
                                },
                                # Image content
                                {
                                    "properties": {
                                        "image": {
                                            "type": "object",
                                            "properties": {
                                                "format": {
                                                    "type": "string",
                                                    "enum": ["png", "jpg", "jpeg"],
                                                },
                                                "source": {
                                                    "type": "object",
                                                    "properties": {
                                                        "s3Location": {
                                                            "type": "object",
                                                            "properties": {
                                                                "uri": {
                                                                    "type": "string"
                                                                },
                                                                "bucketOwner": {
                                                                    "type": "string"
                                                                },
                                                            },
                                                            "required": [
                                                                "uri",
                                                                "bucketOwner",
                                                            ],
                                                            "additionalProperties": False,
                                                        }
                                                    },
                                                    "required": ["s3Location"],
                                                    "additionalProperties": False,
                                                },
                                            },
                                            "required": ["format", "source"],
                                            "additionalProperties": False,
                                        }
                                    },
                                    "required": ["image"],
                                    "additionalProperties": False,
                                },
                                # Video content
                                {
                                    "properties": {
                                        "video": {
                                            "type": "object",
                                            "properties": {
                                                "format": {
                                                    "type": "string",
                                                    "enum": ["mp4", "mov", "avi"],
                                                },
                                                "source": {
                                                    "type": "object",
                                                    "properties": {
                                                        "s3Location": {
                                                            "type": "object",
                                                            "properties": {
                                                                "uri": {
                                                                    "type": "string"
                                                                },
                                                                "bucketOwner": {
                                                                    "type": "string"
                                                                },
                                                            },
                                                            "required": [
                                                                "uri",
                                                                "bucketOwner",
                                                            ],
                                                            "additionalProperties": False,
                                                        }
                                                    },
                                                    "required": ["s3Location"],
                                                    "additionalProperties": False,
                                                },
                                            },
                                            "required": ["format", "source"],
                                            "additionalProperties": False,
                                        }
                                    },
                                    "required": ["video"],
                                    "additionalProperties": False,
                                },
                            ],
                        },
                    },
                },
                "required": ["role", "content"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["schemaVersion", "messages"],
    "additionalProperties": False,
}

SFT_NOVA_TWO_CONVERSE_2024 = {
    "type": "object",
    "properties": {
        "schemaVersion": {"type": "string", "const": "bedrock-conversation-2024"},
        "system": {
            "type": "array",
            "maxItems": 1,
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
        },
        "messages": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "role": {"type": "string", "enum": ["user", "assistant"]},
                    "content": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "oneOf": [
                                # Text content
                                {
                                    "properties": {"text": {"type": "string"}},
                                    "required": ["text"],
                                    "additionalProperties": False,
                                },
                                # Reasoning content
                                {
                                    "properties": {
                                        "reasoningContent": {
                                            "type": "object",
                                            "properties": {
                                                "reasoningText": {
                                                    "type": "object",
                                                    "properties": {
                                                        "text": {"type": "string"}
                                                    },
                                                    "required": ["text"],
                                                    "additionalProperties": False,
                                                }
                                            },
                                            "required": ["reasoningText"],
                                            "additionalProperties": False,
                                        }
                                    },
                                    "required": ["reasoningContent"],
                                    "additionalProperties": False,
                                },
                                {
                                    "properties": {
                                        "toolUse": {
                                            "type": "object",
                                            "properties": {
                                                "toolUseId": {"type": "string"},
                                                "name": {"type": "string"},
                                                "input": {"type": "object"},
                                            },
                                            "required": ["toolUseId", "name", "input"],
                                            "additionalProperties": False,
                                        }
                                    },
                                    "required": ["toolUse"],
                                    "additionalProperties": False,
                                },
                                {
                                    "properties": {
                                        "toolResult": {
                                            "type": "object",
                                            "properties": {
                                                "toolUseId": {"type": "string"},
                                                "content": {
                                                    "type": "array",
                                                    "items": {
                                                        "type": "object",
                                                        "properties": {
                                                            "text": {"type": "string"}
                                                        },
                                                        "required": ["text"],
                                                        "additionalProperties": False,
                                                    },
                                                },
                                            },
                                            "required": ["toolUseId", "content"],
                                            "additionalProperties": False,
                                        }
                                    },
                                    "required": ["toolResult"],
                                    "additionalProperties": False,
                                },
                                # Image content
                                {
                                    "properties": {
                                        "image": {
                                            "type": "object",
                                            "properties": {
                                                "format": {
                                                    "type": "string",
                                                    "enum": ["jpeg", "jpg", "png"],
                                                },
                                                "source": {
                                                    "type": "object",
                                                    "properties": {
                                                        "s3Location": {
                                                            "type": "object",
                                                            "properties": {
                                                                "uri": {
                                                                    "type": "string"
                                                                },
                                                                "bucketOwner": {
                                                                    "type": "string"
                                                                },
                                                            },
                                                            "required": [
                                                                "uri",
                                                                "bucketOwner",
                                                            ],
                                                            "additionalProperties": False,
                                                        }
                                                    },
                                                    "required": ["s3Location"],
                                                    "additionalProperties": False,
                                                },
                                            },
                                            "required": ["format", "source"],
                                            "additionalProperties": False,
                                        }
                                    },
                                    "required": ["image"],
                                    "additionalProperties": False,
                                },
                                # Video content
                                {
                                    "properties": {
                                        "video": {
                                            "type": "object",
                                            "properties": {
                                                "format": {
                                                    "type": "string",
                                                    "enum": ["mp4", "mov", "avi"],
                                                },
                                                "source": {
                                                    "type": "object",
                                                    "properties": {
                                                        "s3Location": {
                                                            "type": "object",
                                                            "properties": {
                                                                "uri": {
                                                                    "type": "string"
                                                                },
                                                                "bucketOwner": {
                                                                    "type": "string"
                                                                },
                                                            },
                                                            "required": [
                                                                "uri",
                                                                "bucketOwner",
                                                            ],
                                                            "additionalProperties": False,
                                                        }
                                                    },
                                                    "required": ["s3Location"],
                                                    "additionalProperties": False,
                                                },
                                            },
                                            "required": ["format", "source"],
                                            "additionalProperties": False,
                                        }
                                    },
                                    "required": ["video"],
                                    "additionalProperties": False,
                                },
                            ],
                        },
                    },
                },
                "required": ["role", "content"],
                "additionalProperties": False,
            },
        },
        "toolConfig": {
            "type": "object",
            "properties": {
                "tools": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "toolSpec": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "description": {"type": "string"},
                                    "inputSchema": {
                                        "type": "object",
                                        "properties": {"json": {"type": "object"}},
                                        "required": ["json"],
                                        "additionalProperties": False,
                                    },
                                },
                                "required": ["name", "inputSchema"],
                                "additionalProperties": False,
                            }
                        },
                        "required": ["toolSpec"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["tools"],
            "additionalProperties": False,
        },
    },
    "required": ["schemaVersion", "messages"],
    "additionalProperties": False,
}

OPENAI_FORMAT = {
    "type": "object",
    "properties": {
        "messages": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "role": {
                        "type": "string",
                        "enum": ["system", "user", "assistant", "tool"],
                    },
                    "content": {
                        "type": ["string", "null"]
                    },  # Allow null for assistant messages with only tool_calls
                    "tool_calls": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "type": {"type": "string"},
                                "function": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "arguments": {"type": "string"},
                                    },
                                    "required": ["name", "arguments"],
                                },
                            },
                            "required": ["id", "type", "function"],
                        },
                    },
                    "tool_call_id": {"type": "string"},
                    "reasoning": {"type": "string"},
                },
                "required": ["role"],
                "additionalProperties": False,
            },
        },
        "tools": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "function": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "parameters": {"type": "object"},
                        },
                        "required": ["name"],
                    },
                },
                "required": ["type", "function"],
            },
        },
    },
    "required": ["messages"],
    "additionalProperties": False,
}

# TODO: Verify the core requirements for RFT formatting. Specifically "role" for names and what's required.
RFT_OPENAI_FORMAT = {
    "type": "object",
    "properties": {
        "messages": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "role": {
                        "type": "string",
                        "enum": ["system", "user", "assistant", "developer", "tool"],
                    },
                    "content": {
                        "type": ["string", "null"]
                    },  # Allow null for assistant messages with only tool_calls
                    "tool_calls": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "type": {"type": "string"},
                                "function": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "arguments": {"type": "string"},
                                    },
                                    "required": ["name", "arguments"],
                                },
                            },
                            "required": ["id", "type", "function"],
                        },
                    },
                    "tool_call_id": {"type": "string"},
                    "reasoning": {"type": "string"},
                },
                "required": ["role"],
            },
            "minItems": 1,
        },
        "reference_answer": {
            "type": "object",
            "additionalProperties": True,  # Allows any structure inside reference_answer
        },
        "tools": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "function": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "parameters": {"type": "object"},
                        },
                        "required": ["name"],
                    },
                },
                "required": ["type", "function"],
            },
        },
    },
    "required": ["messages"],
    "additionalProperties": True,  # This allows any additional top-level fields
}

EVALUATION_FORMAT = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "response": {"type": "string"},
        "system": {"type": "string"},
        "images": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"data": {"type": "string"}},
                "required": ["data"],
                "additionalProperties": False,
            },
        },
        "metadata": {"type": "string"},
    },
    "required": ["query", "response"],
    "additionalProperties": False,
}

CPT_FORMAT = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
    },
    "required": ["text"],
    "additionalProperties": False,
}
