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
"""Reward function verification utility for RFT training."""

import json
import re
from typing import Any, Dict, List, Optional

import boto3
from botocore.exceptions import ClientError
from pydantic import ValidationError

from amzn_nova_forge_sdk.dataset.dataset_validator.rft_dataset_validator import (
    RFTDatasetValidator,
)
from amzn_nova_forge_sdk.model.model_enums import Model, Platform
from amzn_nova_forge_sdk.util.logging import logger


def _validate_output_format(result: Dict[str, Any], idx: int) -> List[str]:
    """
    Validate the output format of a single reward function result.

    Args:
        result: The output dict from the reward function
        idx: Index of the result for error messages

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    if not isinstance(result, dict):
        errors.append(f"Output {idx}: Expected dict, got {type(result).__name__}")
        return errors

    # Check required fields
    if "id" not in result:
        errors.append(f"Output {idx}: Missing 'id' field (required for RFT)")

    if "aggregate_reward_score" not in result:
        errors.append(
            f"Output {idx}: Missing 'aggregate_reward_score' field (required for RFT)"
        )
    elif not isinstance(result.get("aggregate_reward_score"), (int, float)):
        errors.append(f"Output {idx}: 'aggregate_reward_score' should be a number")

    # Validate metrics_list if present (optional for both training and evaluation)
    if "metrics_list" in result:
        metrics_list = result["metrics_list"]
        if not isinstance(metrics_list, list):
            errors.append(f"Output {idx}: 'metrics_list' should be a list")
        else:
            for metric_idx, metric in enumerate(metrics_list):
                if not isinstance(metric, dict):
                    errors.append(
                        f"Output {idx}, metric {metric_idx}: Expected dict, got {type(metric).__name__}"
                    )
                else:
                    # Validate required metric fields
                    if "name" not in metric:
                        errors.append(
                            f"Output {idx}, metric {metric_idx}: Missing 'name' field"
                        )
                    elif not isinstance(metric["name"], str):
                        errors.append(
                            f"Output {idx}, metric {metric_idx}: 'name' should be a string"
                        )

                    if "value" not in metric:
                        errors.append(
                            f"Output {idx}, metric {metric_idx}: Missing 'value' field"
                        )
                    elif not isinstance(metric["value"], (int, float)):
                        errors.append(
                            f"Output {idx}, metric {metric_idx}: 'value' should be a number"
                        )

                    if "type" not in metric:
                        errors.append(
                            f"Output {idx}, metric {metric_idx}: Missing 'type' field"
                        )
                    elif metric["type"] not in ["Metric", "Reward"]:
                        errors.append(
                            f"Output {idx}, metric {metric_idx}: 'type' should be 'Metric' or 'Reward', got '{metric['type']}'"
                        )

    return errors


def verify_reward_function(
    reward_function: str,
    sample_data: List[Dict[str, Any]],
    region: str = "us-east-1",
    validate_format: bool = True,
    platform: Optional[Platform] = None,
) -> Dict[str, Any]:
    """
    Verify a reward function with sample data before using it in RFT training or evaluation.

    This function allows you to test your reward function implementation with sample
    conversation data to ensure it works correctly before submitting a training or evaluation job.

    Args:
        reward_function: Either a Lambda ARN (string starting with 'arn:aws:lambda:')
                        or a path to a local Python file containing the reward function.
        sample_data: List of conversation samples to test. Each sample should be a dict
                    with 'id', 'messages', and optionally 'reference_answer' keys.
        region: AWS region for Lambda invocation (default: us-east-1).
        validate_format: If True, validates that sample_data matches RFT format and
                        output matches expected format (default: True).
        platform: Platform enum (Platform.SMHP or Platform.SMTJ). Required when using Lambda ARN.
                 When set to Platform.SMHP, validates that Lambda ARN contains 'SageMaker' in
                 the function name as required by SageMaker HyperPod. Optional for local files.

    Returns:
        Dict containing:
            - success: bool (always True if no exception raised)
            - results: list of individual test results
            - total_samples: total number of samples tested
            - successful_samples: number of successful tests
            - warnings: list of warning messages (e.g., missing reference_answer)

    Raises:
        ValueError: If any validation errors are encountered, with a detailed error message
                   listing all issues found.

    Example:
        >>> # Test training reward function with Lambda ARN
        >>> result = verify_reward_function(
        ...     reward_function="arn:aws:lambda:us-east-1:123456789012:function:my-reward",
        ...     sample_data=[
        ...         {
        ...             "id": "sample_1",
        ...             "reference_answer": "correct answer",
        ...             "messages": [
        ...                 {"role": "user", "content": "question"},
        ...                 {"role": "assistant", "content": "response"}
        ...             ]
        ...         }
        ...     ]
        ... )

        >>> # Test reward function with local Python file
        >>> result = verify_reward_function(
        ...     reward_function="./my_reward.py",
        ...     sample_data=[
        ...         {
        ...             "id": "sample_1",
        ...             "reference_answer": "correct answer",
        ...             "messages": [
        ...                 {"role": "user", "content": "question"},
        ...                 {"role": "assistant", "content": "response"}
        ...             ]
        ...         }
        ...     ]
        ... )

        >>> # Test with SMHP platform validation
        >>> result = verify_reward_function(
        ...     reward_function="arn:aws:lambda:us-east-1:123456789012:function:MySageMakerReward",
        ...     sample_data=[...],
        ...     platform=Platform.SMHP
        ... )
    """
    # Determine if it's a Lambda ARN or local file
    is_lambda = reward_function.startswith("arn:aws:lambda:")

    # Platform is required for Lambda ARNs
    if is_lambda and platform is None:
        raise ValueError(
            "The 'platform' parameter is required when using a Lambda ARN. "
            "Please specify platform=Platform.SMHP or platform=Platform.SMTJ."
        )

    # Validate Lambda ARN format for SMHP platform
    if platform == Platform.SMHP and is_lambda:
        # Extract function name from ARN: arn:aws:lambda:region:account:function:function-name
        function_name_match = re.search(
            r"arn:aws:lambda:[^:]+:[^:]+:function:([^:]+)", reward_function
        )
        if function_name_match:
            function_name = function_name_match.group(1)
            # Check if function name contains 'SageMaker' (case-insensitive)
            if not re.search(r"sagemaker", function_name, re.IGNORECASE):
                raise ValueError(
                    f"Lambda ARN for SMHP must contain 'SageMaker' in the function name. "
                    f"Current function name: '{function_name}'. "
                    f"Expected format: 'arn:aws:lambda:*:*:function:*SageMaker*'"
                )
        else:
            raise ValueError(
                f"Invalid Lambda ARN format: {reward_function}. "
                f"Expected format: 'arn:aws:lambda:region:account:function:function-name'"
            )

    results = []
    warnings = []

    # Validate input format if requested
    if validate_format:
        # Use RFT dataset validator for consistent validation
        validator = RFTDatasetValidator(model=Model.NOVA_LITE_2)
        input_errors = []

        for idx, sample in enumerate(sample_data):
            try:
                # Validate using the RFT dataset validator
                validator.get_sample_model()(**sample)

                # Additional check: reference_answer is recommended for meaningful rewards
                # (it's optional in RFT dataset format)
                if (
                    "reference_answer" not in sample
                    or sample["reference_answer"] is None
                ):
                    warnings.append(
                        f"Sample {idx} (id: {sample.get('id', 'unknown')}): No 'reference_answer' provided. "
                        f"Without reference_answer, your reward function cannot compare model outputs against expected answers. "
                        f"Consider adding reference answers for more effective reward signals."
                    )

            except ValidationError as e:
                # Convert Pydantic validation errors to our format
                for error in e.errors():
                    field_path = " -> ".join(str(loc) for loc in error["loc"])
                    input_errors.append(
                        f"Sample {idx}, field '{field_path}': {error['msg']}"
                    )
            except Exception as e:
                input_errors.append(f"Sample {idx}: {str(e)}")

        # If there are input validation errors, raise immediately
        if input_errors:
            error_message = (
                "Input validation failed with the following errors:\n"
                + "\n".join(f"  - {err}" for err in input_errors)
            )
            raise ValueError(error_message)

    if is_lambda:
        # Test with Lambda
        lambda_client = boto3.client("lambda", region_name=region)

        try:
            # Log Lambda invocation details for debugging
            logger.info(f"Invoking Lambda: {reward_function}")
            logger.info(f"Number of samples: {len(sample_data)}")

            # Invoke Lambda with all samples at once (Lambda expects array)
            response = lambda_client.invoke(
                FunctionName=reward_function,
                InvocationType="RequestResponse",
                Payload=json.dumps(sample_data),
            )

            # Parse response
            payload = json.loads(response["Payload"].read())

            # Process results
            if isinstance(payload, list):
                logger.info(f"Lambda returned list with {len(payload)} result(s)")
                for idx, result in enumerate(payload):
                    # Log input for this sample
                    if idx < len(sample_data):
                        input_str = json.dumps(sample_data[idx], indent=2)
                        logger.info(f"Sample {idx} INPUT:\n{input_str}")

                    # Validate output format
                    sample_errors = []
                    if validate_format:
                        sample_errors = _validate_output_format(result, idx)

                    # Log output for this sample
                    result_str = json.dumps(result, indent=2)
                    status = "PASS" if not sample_errors else "FAIL"
                    logger.info(f"Sample {idx} OUTPUT [{status}]:\n{result_str}")
                    if sample_errors:
                        logger.warning(
                            f"Sample {idx} validation errors: {', '.join(sample_errors)}"
                        )

                    results.append(
                        {
                            "sample_index": idx,
                            "input": sample_data[idx] if idx < len(sample_data) else {},
                            "output": result,
                            "status": "error" if sample_errors else "success",
                            "errors": sample_errors,
                        }
                    )
            else:
                # Single result format (not a list) - could be Lambda error or single dict response
                if isinstance(payload, dict) and (
                    "errorMessage" in payload or "errorType" in payload
                ):
                    # Lambda execution error - treat all samples as failed with a single shared error
                    error_msg = payload.get("errorMessage", "Unknown error")
                    error_type = payload.get("errorType", "Unknown")
                    error_text = f"Lambda execution error - {error_type}: {error_msg}"

                    # Mark all samples as failed, but only include error text in first result
                    # to avoid repeating the same error 200 times
                    for idx in range(len(sample_data)):
                        results.append(
                            {
                                "sample_index": idx,
                                "input": sample_data[idx],
                                "output": payload,
                                "status": "error",
                                "errors": [error_text] if idx == 0 else [],
                            }
                        )
                else:
                    # Single successful result
                    logger.info("Lambda returned single result (non-list format)")
                    result_str = (
                        json.dumps(payload, indent=2)
                        if isinstance(payload, dict)
                        else str(payload)
                    )
                    logger.info(f"Result:\n{result_str}")

                    results.append(
                        {
                            "sample_index": 0,
                            "input": sample_data,
                            "output": payload,
                            "status": "success",
                            "errors": [],
                        }
                    )

        except (ClientError, Exception) as e:
            error_msg = f"Lambda invocation failed: {str(e)}"
            # Mark all samples as failed
            for idx in range(len(sample_data)):
                results.append(
                    {
                        "sample_index": idx,
                        "input": sample_data[idx],
                        "status": "error",
                        "errors": [error_msg],
                    }
                )

    else:
        # Test with local Python file
        logger.info(f"Testing local Python file: {reward_function}")
        logger.info(f"Number of samples: {len(sample_data)}")

        try:
            # Read and execute the Python file
            with open(reward_function, "r") as f:
                code = f.read()

            # Create a namespace for execution
            namespace: Dict[str, Any] = {}
            exec(code, namespace)

            # Find the lambda_handler function
            if "lambda_handler" not in namespace:
                raise ValueError(
                    "Local Python file must contain a 'lambda_handler' function"
                )

            handler = namespace["lambda_handler"]

            # Call handler with all samples at once (matches Lambda behavior)
            try:
                result = handler(sample_data, {})

                # Log handler response
                result_str = (
                    json.dumps(result, indent=2)
                    if isinstance(result, (dict, list))
                    else str(result)
                )
                logger.info(f"Handler response:\n{result_str}")

                # Process results
                if isinstance(result, list):
                    logger.info(f"Lambda returned list with {len(result)} result(s)")
                    for idx, item in enumerate(result):
                        # Log input for this sample
                        if idx < len(sample_data):
                            input_str = json.dumps(sample_data[idx], indent=2)
                            logger.info(f"Sample {idx} INPUT:\n{input_str}")

                        # Validate output format
                        sample_errors = []
                        if validate_format:
                            sample_errors = _validate_output_format(item, idx)

                        # Log output for this sample
                        item_str = json.dumps(item, indent=2)
                        status = "PASS" if not sample_errors else "FAIL"
                        logger.info(f"Sample {idx} OUTPUT [{status}]:\n{item_str}")
                        if sample_errors:
                            logger.warning(
                                f"Sample {idx} validation errors: {', '.join(sample_errors)}"
                            )

                        results.append(
                            {
                                "sample_index": idx,
                                "input": sample_data[idx]
                                if idx < len(sample_data)
                                else {},
                                "output": item,
                                "status": "error" if sample_errors else "success",
                                "errors": sample_errors,
                            }
                        )
                else:
                    # Single result format
                    logger.info("Lambda returned single result (non-list format)")
                    results.append(
                        {
                            "sample_index": 0,
                            "input": sample_data,
                            "output": result,
                            "status": "success",
                            "errors": [],
                        }
                    )

            except Exception as e:
                error_msg = f"Handler execution failed: {str(e)}"
                # Mark all samples as failed
                for idx in range(len(sample_data)):
                    results.append(
                        {
                            "sample_index": idx,
                            "input": sample_data[idx] if idx < len(sample_data) else {},
                            "status": "error",
                            "errors": [error_msg],
                        }
                    )

        except FileNotFoundError:
            error_msg = f"Python file not found: {reward_function}"
            # Mark all samples as failed (no results to process)
            for idx in range(len(sample_data)):
                results.append(
                    {
                        "sample_index": idx,
                        "input": sample_data[idx],
                        "status": "error",
                        "errors": [error_msg],
                    }
                )

        except Exception as e:
            error_msg = f"Failed to load Python file: {str(e)}"
            # Mark all samples as failed
            for idx in range(len(sample_data)):
                results.append(
                    {
                        "sample_index": idx,
                        "input": sample_data[idx],
                        "status": "error",
                        "errors": [error_msg],
                    }
                )

    # Log warnings if any
    if warnings:
        logger.warning(
            f"Reward function verification completed with {len(warnings)} warning(s):"
        )
        for warning in warnings:
            logger.warning(f"  - {warning}")

    # Count successful samples
    successful_samples = len([r for r in results if r.get("status") == "success"])
    total_samples = len(sample_data)
    failed_samples = total_samples - successful_samples

    # Log verification summary
    if failed_samples == 0:
        logger.info(f"All {total_samples} sample(s) passed validation")
    else:
        logger.warning(f"{failed_samples}/{total_samples} sample(s) failed validation")

    # Raise error if any samples failed
    if failed_samples > 0:
        # Collect all errors from results
        all_errors = []
        for r in results:
            if r.get("status") == "error" and "errors" in r:
                all_errors.extend(r["errors"])

        # Build simplified error message
        error_parts = [
            f"Reward function verification failed: {failed_samples}/{total_samples} sample(s) failed validation.",
            f"Only {successful_samples}/{total_samples} sample(s) passed.",
            "",
        ]

        # Include all validation errors
        if all_errors:
            error_parts.append("Validation errors:")
            for err in all_errors:
                error_parts.append(f"  - {err}")
            error_parts.append("")

        # Add helpful guidance
        error_parts.extend(
            [
                "Please check your reward function output format. Each output must include:",
                "  - 'id': string identifier",
                "  - 'aggregate_reward_score': numeric reward value",
                "",
            ]
        )

        error_message = "\n".join(error_parts)
        raise ValueError(error_message)

    return {
        "success": True,
        "results": results,
        "total_samples": total_samples,
        "successful_samples": successful_samples,
        "warnings": warnings,
    }
