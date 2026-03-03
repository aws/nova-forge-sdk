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
"""
This class contains the different transform functions needed to switch between OpenAI, Converse, and Generic formats.
The functions here are called from the dataset_loader class and should not be edited unless formatting issues occur.

Running list of potential default values:
    SFT: question, answer
        Optional: system, [image/video required options]: image_format, video_format, s3_uri, bucket_owner
        2.0: reasoning_text, tools/toolsConfig
    RFT: question, reference_answer (for transforming from plain JSONL -> OpenAI)
        Optional: system, id, tools
    Evaluation: query, response
        Optional: images, metadata
    CPT: text
"""

import json


class DatasetTransformer:
    default_system_msg = "You are a helpful assistant who answers the question based on the task assigned"

    @staticmethod
    def convert_to_converse_sft_nova_one(rec, column_mappings):
        # These are the required columns for SFT Converse format for Nova 1.0.
        question_col = column_mappings.get("question")
        answer_col = column_mappings.get("answer")

        # These are the optional columns for SFT
        system_col = column_mappings.get("system")
        image_format_col = column_mappings.get("image_format")
        video_format_col = column_mappings.get("video_format")
        s3_uri_col = column_mappings.get("s3_uri")
        bucket_owner_col = column_mappings.get("bucket_owner")

        # Checks if the minimum required columns exist - errors if not found.
        if (question_col not in rec) or (answer_col not in rec):
            raise ValueError(
                f"Question column or answer column not found in record {rec}, which is needed for SFT.\n"
                f"Make sure to add the correct column mappings when initializing DatasetLoader."
            )

        # TODO: Clean up the checks here into a helper function.

        # If it has the required columns, put together the rest of the message.
        system_message = (
            [{"text": rec[system_col]}]
            if system_col and system_col in rec
            else [{"text": DatasetTransformer.default_system_msg}]
        )

        # User message
        user_content = []

        # Add video or image (only one is allowed) if present.
        if video_format_col and video_format_col in rec:
            video_uri = rec.get(s3_uri_col)
            bucket_owner = rec.get(bucket_owner_col)
            if video_uri and bucket_owner:
                user_content.append(
                    {
                        "video": {
                            "format": rec[video_format_col],
                            "source": {
                                "s3Location": {
                                    "uri": rec[s3_uri_col],
                                    "bucketOwner": rec[bucket_owner_col],
                                }
                            },
                        }
                    },
                )
                # Add the text part of the message.
                user_content.append({"text": rec[question_col]})
            else:
                raise ValueError(
                    f"Tried to add a video record, but required column(s) are missing: s3_uri and/or bucket_owner"
                )
        elif image_format_col and image_format_col in rec:
            if s3_uri_col and bucket_owner_col:
                image_uri = rec.get(s3_uri_col)
                bucket_owner = rec.get(bucket_owner_col)
                if image_uri and bucket_owner:
                    user_content.append(
                        {
                            "image": {
                                "format": rec[image_format_col],
                                "source": {
                                    "s3Location": {
                                        "uri": image_uri,
                                        "bucketOwner": bucket_owner,
                                    }
                                },
                            }
                        }
                    )
                    # Add the text part of the message.
                    user_content.append({"text": rec[question_col]})
                else:
                    raise ValueError(
                        f"Tried to add an image record, but required column(s) are missing: s3_uri and/or bucket_owner"
                    )

        # Final JSON Line
        conversation = {
            "schemaVersion": "bedrock-conversation-2024",
            "system": system_message,
            "messages": [
                {
                    "role": "user",
                    "content": user_content
                    if user_content
                    else [{"text": rec[question_col]}],
                },
                {"role": "assistant", "content": [{"text": rec[answer_col]}]},
            ],
        }
        return conversation

    @staticmethod
    def convert_to_converse_sft_nova_two(rec, column_mappings):
        # These are the required columns for SFT Converse format for Nova 2.0.
        question_col = column_mappings.get("question")
        answer_col = column_mappings.get("answer")

        # These are the optional columns for SFT
        system_col = column_mappings.get("system")
        image_format_col = column_mappings.get("image_format")
        video_format_col = column_mappings.get("video_format")
        s3_uri_col = column_mappings.get("s3_uri")
        bucket_owner_col = column_mappings.get("bucket_owner")
        reasoning_text_col = column_mappings.get("reasoning_text")

        # Checks if the minimum required columns exist - errors if not found.
        if (question_col not in rec) or (answer_col not in rec):
            raise ValueError(
                f"Question column or answer column not found in record {rec}, which is needed for SFT.\n"
                f"Make sure to add the correct column mappings when initializing DatasetLoader."
            )

        # If it has the required columns, put together the rest of the message.
        system_message = (
            [{"text": rec[system_col]}]
            if system_col and system_col in rec
            else [{"text": DatasetTransformer.default_system_msg}]
        )

        # User message
        user_content = []

        # Add video or image (only one is allowed) if present.
        if video_format_col and video_format_col in rec:
            video_uri = rec.get(s3_uri_col)
            bucket_owner = rec.get(bucket_owner_col)
            if video_uri and bucket_owner:
                user_content.append(
                    {
                        "video": {
                            "format": rec[video_format_col],
                            "source": {
                                "s3Location": {
                                    "uri": rec[s3_uri_col],
                                    "bucketOwner": rec[bucket_owner_col],
                                }
                            },
                        }
                    },
                )
                # Add the text part of the message.
                user_content.append({"text": rec[question_col]})
            else:
                raise ValueError(
                    f"Tried to add a video record, but required column(s) are missing: s3_uri and/or bucket_owner"
                )
        elif image_format_col and image_format_col in rec:
            if s3_uri_col and bucket_owner_col:
                image_uri = rec.get(s3_uri_col)
                bucket_owner = rec.get(bucket_owner_col)
                if image_uri and bucket_owner:
                    user_content.append(
                        {
                            "image": {
                                "format": rec[image_format_col],
                                "source": {
                                    "s3Location": {
                                        "uri": image_uri,
                                        "bucketOwner": bucket_owner,
                                    }
                                },
                            }
                        }
                    )
                    # Add the text part of the message.
                    user_content.append({"text": rec[question_col]})
                else:
                    raise ValueError(
                        f"Tried to add an image record, but required column(s) are missing: s3_uri and/or bucket_owner"
                    )

        # Create assistant line:
        assistant_content = []

        # Reasoning text is optional.
        if reasoning_text_col in rec and rec[reasoning_text_col]:
            reasoning_text = {"reasoningText": {"text": rec[reasoning_text_col]}}
            assistant_content.append({"reasoningContent": reasoning_text})

        # Always include the assistant’s answer
        if answer_col in rec:
            assistant_content.append({"text": rec[answer_col]})

        # Final JSON Line
        conversation = {
            "schemaVersion": "bedrock-conversation-2024",
            "system": system_message,
            "messages": [
                {
                    "role": "user",
                    "content": user_content
                    if user_content
                    else [{"text": rec[question_col]}],
                },
                {"role": "assistant", "content": assistant_content},
            ],
        }
        return conversation

    @staticmethod
    def convert_to_openai_rft(rec, column_mappings):
        # These are the required columns for RFT OpenAI format.
        question_col = column_mappings.get("question")
        ref_answer_col = column_mappings.get("reference_answer")

        # These are the optional columns for RFT OpenAI format.
        system_col = column_mappings.get("system")
        id_col = column_mappings.get("id")

        # Check if these columns exist before returning a formatted response.
        if (question_col not in rec) or (ref_answer_col not in rec):
            raise ValueError(
                f"Question column or Reference Answer column not found in record {rec}, which is needed for RFT."
                f"Make sure to add the correct column mappings when initializing DatasetLoader."
            )
        else:
            # Check if an ID is included, if so, add it first.
            rft_format = {}
            if id_col and id_col in rec:
                rft_format["id"] = rec[id_col]

            # Add the remaining fields for RFT.
            rft_format.update(
                {
                    "messages": [
                        {
                            "role": "system",
                            "content": rec.get(
                                system_col, "You are a helpful AI assistant."
                            ),
                        },
                        {"role": "user", "content": rec[question_col]},
                    ],
                    "reference_answer": rec[
                        ref_answer_col
                    ],  # TODO: Maybe move this to metadata
                }
            )
            # Add metadata -- can be any length/number of mappings.
            for key, value in column_mappings.items():
                if key not in [question_col, ref_answer_col, system_col, id_col]:
                    if value in rec:
                        rft_format[key] = rec[value]

            return rft_format

    @staticmethod
    def convert_to_evaluation(rec, column_mappings):
        # These are the required columns for Eval format.
        query_col = column_mappings.get("query")
        response_col = column_mappings.get("response")

        # These are the optional columns for Eval format.
        system_col = column_mappings.get("system")
        image_col = column_mappings.get("images")
        metadata_col = column_mappings.get("metadata")

        # Check if these columns exist before returning a formatted response.
        if (query_col not in rec) or (response_col not in rec):
            raise ValueError(
                f"Query column or response column not found in record {rec}, which is needed for Evaluation.\n"
                f"Make sure to add the correct column mappings when initializing DatasetLoader."
            )
        else:
            result = {"query": rec[query_col], "response": rec[response_col]}
            if system_col and system_col in rec:
                result["system"] = rec[system_col]

            if image_col and image_col in rec:
                images = rec[image_col]  # Gets all images if there are multiple
                if isinstance(images, list):
                    result["images"] = [{"data": img} for img in images]
                elif isinstance(images, str):
                    result["images"] = [{"data": images}]

            if metadata_col and metadata_col in rec:
                result["metadata"] = rec[metadata_col]

            return result

    @staticmethod
    def _parse_tool_arguments(arguments_str):
        """
        Parse tool arguments from string format to dict.
        Handles both JSON strings and already parsed dicts.
        """
        if isinstance(arguments_str, dict):
            return arguments_str

        try:
            return json.loads(arguments_str)
        except (json.JSONDecodeError, TypeError):
            # If parsing fails, return as a simple dict with the string
            return {"arguments": arguments_str}

    @staticmethod
    def _convert_openai_tools_to_converse_toolconfig(tools):
        """
        Convert OpenAI tools definition to Bedrock Converse toolConfig format.
        """
        if not tools:
            return None

        converse_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                function = tool.get("function", {})
                tool_spec = {
                    "toolSpec": {
                        "name": function.get("name", ""),
                        "description": function.get("description", ""),
                        "inputSchema": {"json": function.get("parameters", {})},
                    }
                }
                converse_tools.append(tool_spec)

        if converse_tools:
            return {"toolConfig": {"tools": converse_tools}}

        return None

    @staticmethod
    def _convert_openai_messages_to_converse(messages, nova_version="1.0"):
        """
        Convert OpenAI messages to Converse format for Nova 1.0 and 2.0 SFT.
        Merges consecutive tool messages into a single user message to ensure alternating roles.
        """
        system_content = DatasetTransformer.default_system_msg
        converse_messages = []
        pending_tool_results = []  # Collect tool results to merge

        for i, msg in enumerate(messages):
            role = msg.get("role")
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls", [])
            tool_call_id = msg.get("tool_call_id")

            # Reasoning only present in Nova 2.0
            reasoning = msg.get("reasoning") if nova_version == "2.0" else None

            if role == "system":
                system_content = content
            elif role == "user":
                # If we have pending tool results, add them first as a user message
                if nova_version == "2.0" and pending_tool_results:
                    converse_messages.append(
                        {"role": "user", "content": pending_tool_results}
                    )
                    pending_tool_results = []

                # Then add the regular user message if it has content
                if content:
                    converse_messages.append(
                        {"role": "user", "content": [{"text": content}]}
                    )
            elif role == "assistant":
                # If we have pending tool results, add them first as a user message
                if nova_version == "2.0" and pending_tool_results:
                    converse_messages.append(
                        {"role": "user", "content": pending_tool_results}
                    )
                    pending_tool_results = []

                assistant_content = []

                # Add reasoning content if present (Nova 2.0 feature)
                if nova_version == "2.0" and reasoning:
                    assistant_content.append(
                        {"reasoningContent": {"reasoningText": {"text": reasoning}}}
                    )

                # Add text content if present
                if content:
                    assistant_content.append({"text": content})

                # toolUse goes under assistant (only for Nova 2.0)
                if nova_version == "2.0":
                    for tool_call in tool_calls:
                        if tool_call.get("type") == "function":
                            function_data = tool_call.get("function", {})
                            tool_use_id = tool_call.get("id", f"tool_call_{i}")
                            tool_use_block = {
                                "toolUse": {
                                    "toolUseId": tool_use_id,
                                    "name": function_data.get("name", ""),
                                    "input": DatasetTransformer._parse_tool_arguments(
                                        function_data.get("arguments", "{}")
                                    ),
                                }
                            }
                            assistant_content.append(tool_use_block)

                if assistant_content:
                    converse_messages.append(
                        {"role": "assistant", "content": assistant_content}
                    )
            elif role == "tool":
                # Collect tool results to merge (only for Nova 2.0)
                if nova_version == "2.0":
                    pending_tool_results.append(
                        {
                            "toolResult": {
                                "toolUseId": tool_call_id,
                                "content": [{"text": content}],
                            }
                        }
                    )

        # Add any remaining tool results at the end
        if nova_version == "2.0" and pending_tool_results:
            converse_messages.append({"role": "user", "content": pending_tool_results})

        return system_content, converse_messages

    @staticmethod
    def _build_converse_conversation(system_content, converse_messages, tools=None):
        """
        Build a Converse format conversation from components.
        Validates that required roles are present and adds tool configuration if provided.
        """
        # Validate we have at least one user and one assistant message
        roles_present = [m["role"] for m in converse_messages]
        if "user" not in roles_present or "assistant" not in roles_present:
            raise ValueError(
                f"OpenAI format must contain at least one 'user' and one 'assistant' message. "
                f"Found roles: {roles_present}"
            )

        conversation = {
            "schemaVersion": "bedrock-conversation-2024",
            "system": [{"text": system_content}],
            "messages": converse_messages,
        }

        # Add toolConfig if tools are present
        tool_config = DatasetTransformer._convert_openai_tools_to_converse_toolconfig(
            tools
        )
        if tool_config:
            conversation.update(tool_config)

        return conversation

    @staticmethod
    def convert_openai_to_converse_sft_nova_one(rec, column_mappings=None):
        """
        Convert OpenAI format to Converse format for Nova 1.0.
        """
        if "messages" not in rec:
            raise ValueError(
                f"'messages' key not found in record {rec}. Expected OpenAI format."
            )

        messages = rec["messages"]

        # Check if tools are present and raise error as it is only supported for Nova 2.0
        if rec.get("tools") or any(
            msg.get("tool_calls") or msg.get("tool_call_id") for msg in messages
        ):
            raise ValueError(
                "Tool/function calling is not supported in Nova 1.0. "
                "Please use Nova 2.0 for tool calling capabilities."
            )

        system_content, converse_messages = (
            DatasetTransformer._convert_openai_messages_to_converse(
                messages, nova_version="1.0"
            )
        )

        return DatasetTransformer._build_converse_conversation(
            system_content, converse_messages, tools=None
        )

    @staticmethod
    def convert_openai_to_converse_sft_nova_two(rec, column_mappings=None):
        """
        Convert OpenAI format to Converse format for Nova 2.0.
        Supports optional 'reasoning' field in assistant messages, tool/function calls, and tool configurations.
        """
        if "messages" not in rec:
            raise ValueError(
                f"'messages' key not found in record {rec}. Expected OpenAI format."
            )

        messages = rec["messages"]
        tools = rec.get("tools")

        system_content, converse_messages = (
            DatasetTransformer._convert_openai_messages_to_converse(
                messages, nova_version="2.0"
            )
        )

        return DatasetTransformer._build_converse_conversation(
            system_content, converse_messages, tools
        )

    @staticmethod
    def convert_to_cpt(rec, column_mappings):
        # These are the required columns for CPT format.
        text_col = column_mappings.get("text")

        # Check if these columns exist before returning a formatted response.
        if text_col not in rec:
            raise ValueError(
                f"'text' column not found in record {rec}, which is required for CPT.\n"
                f"Make sure to add a column mapping for 'text' when initializing DatasetLoader."
            )
        else:
            result = {"text": rec[text_col]}
            return result

    @staticmethod
    def convert_to_rft_multiturn(rec, column_mappings):
        """
        Convert flat format to nested RFT multiturn format.

        Args:
            rec: A single dataset record (dict) to transform. Can be in flat or nested format.
                Flat format example: {"id": "s1", "prompt": "Q", "answer": "A", "task": "t", "info": {}}
                Nested format example: {"id": "s1", "metadata": {"prompt": "Q", "answer": "A", ...}}
            column_mappings: Dictionary mapping field names to column names in the dataset.
                Example: {"id": "sample_id", "prompt": "question", "answer": "response", "task": "category", "info": "metadata"}
                The function uses this to locate fields in the flat format record.
                NOTE: This dictionary is mutated during processing - an internal counter (_id_counter)
                is stored in it to track sequential ID generation across multiple records.

        Returns:
            dict: Record in nested RFT multiturn format with structure:
                {"id": str, "metadata": {"prompt": str|list, "answer": str, "task": str, "info": dict|str}}

        Notes:
            - Supports two input formats (flat and nested)
            - Auto-generates sequential IDs (sample_001, sample_002, etc.) if missing
            - Info field is kept as-is (can be dict, string, or other type)
            - Prompt can be a string or OpenAI message format (list of dicts)
        """

        # Check if already in nested format
        if "metadata" in rec:
            # Check if nested format has id, generate if missing
            if "id" not in rec or not rec["id"]:
                # Use a counter stored in column_mappings (mutable dict)
                if "_id_counter" not in column_mappings:
                    column_mappings["_id_counter"] = 0
                column_mappings["_id_counter"] += 1
                rec["id"] = f"sample_{column_mappings['_id_counter']:03d}"

            return rec

        # Convert flat format to nested format
        # Only use fields that were explicitly provided in column_mappings
        id_col = column_mappings.get("id")
        prompt_col = column_mappings.get("prompt")
        answer_col = column_mappings.get("answer")
        task_col = column_mappings.get("task")
        info_col = column_mappings.get("info")

        # Check/generate ID
        if id_col and (id_col not in rec or not rec[id_col]):
            # Auto-generate sequential ID
            if "_id_counter" not in column_mappings:
                column_mappings["_id_counter"] = 0
            column_mappings["_id_counter"] += 1
            generated_id = f"sample_{column_mappings['_id_counter']:03d}"
        elif id_col:
            generated_id = rec[id_col]
        else:
            # No id mapping provided, auto-generate
            if "_id_counter" not in column_mappings:
                column_mappings["_id_counter"] = 0
            column_mappings["_id_counter"] += 1
            generated_id = f"sample_{column_mappings['_id_counter']:03d}"

        # Check required prompt field
        if not prompt_col:
            raise ValueError(
                f"'prompt' column mapping is required for RFT Multiturn.\n"
                f"Make sure to add prompt='your_column_name' when initializing DatasetLoader."
            )
        if prompt_col not in rec:
            raise ValueError(
                f"'prompt' column '{prompt_col}' not found in record {rec}, which is required for RFT Multiturn.\n"
                f"Make sure the column exists in your data."
            )

        # Build metadata dict
        metadata = {"prompt": rec[prompt_col]}

        # Add optional answer field with type conversion (only if mapping provided)
        if answer_col and answer_col in rec and rec[answer_col]:
            answer_value = rec[answer_col]
            if not isinstance(answer_value, str):
                answer_value = str(answer_value)
            metadata["answer"] = answer_value

        # Add optional task field with type conversion (only if mapping provided)
        if task_col and task_col in rec and rec[task_col]:
            task_value = rec[task_col]
            if not isinstance(task_value, str):
                task_value = str(task_value)
            metadata["task"] = task_value

        # Add optional info field (only if mapping provided)
        if info_col and info_col in rec and rec[info_col]:
            # Keep info as-is, whether it's a dict, string, or other type
            metadata["info"] = rec[info_col]

        return {"id": generated_id, "metadata": metadata}
