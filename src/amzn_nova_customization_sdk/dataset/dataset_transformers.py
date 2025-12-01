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
        2.0: reasoning_text
    RFT: question, reference_answer (for transforming from plain JSONL -> OpenAI)
        Optional: system, id
    Evaluation: query, response
        Optional: images, metadata
"""

# TODO: Is there a way to simplify the SFT Nova 1.0 vs 2.0 workflows (only difference is reasoning text).


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
