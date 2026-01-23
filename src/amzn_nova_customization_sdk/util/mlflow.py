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
import re
from typing import Any, Dict, List, Optional, Tuple

from amzn_nova_customization_sdk.util.logging import logger


def get_default_mlflow_tracking_uri(region_name: Optional[str] = None) -> Optional[str]:
    """
    Auto-discover the DefaultMLFlowApp for the current AWS account.

    This function attempts to find the DefaultMLFlowApp if it exists.

    Args:
        region_name: AWS region name (e.g., 'us-east-1'). If None, uses default region.

    Returns:
        The ARN of DefaultMLFlowApp if found, or None if:
        - DefaultMLFlowApp doesn't exist
        - boto3 is not available
        - AWS credentials are not configured
        - An error occurs during discovery
    """
    try:
        import boto3
        from botocore.exceptions import (
            ClientError,
            NoCredentialsError,
            NoRegionError,
        )
    except ImportError:
        logger.debug("boto3 not available, skipping MLflow auto-discovery")
        return None

    try:
        # Create SageMaker client
        if region_name:
            sagemaker_client = boto3.client("sagemaker", region_name=region_name)
        else:
            sagemaker_client = boto3.client("sagemaker")

        # Find if DefaultMLFlowApp exists
        try:
            response = sagemaker_client.list_mlflow_apps()

            # Look for DefaultMLFlowApp
            for app in response.get("Summaries", []):
                if app.get("Name") == "DefaultMLFlowApp":
                    # Return the app ARN regardless of status
                    app_arn = app.get("Arn")
                    status = app.get("Status", "")
                    logger.info(f"Using DefaultMLFlowApp: {app_arn} (status: {status})")
                    return app_arn

            logger.error(
                "DefaultMLFlowApp not found, you must specify a Mlflow app/server ARN to use Mlflow monitor"
            )
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            if error_code == "AccessDeniedException":
                logger.error(
                    "Access denied to list MLflow apps while trying to find DefaultMLFlowApp, check a valid Mlflow app/server exists"
                )
            else:
                logger.error(f"Error listing MLflow apps: {e}")

        # No DefaultMLFlowApp found
        return None

    except NoCredentialsError:
        logger.error(
            "AWS credentials not configured during MLflow DefaultMLFlowApp discovery"
        )
        return None
    except NoRegionError:
        logger.error(
            "AWS region not configured during skipping DefaultMLFlowApp discovery"
        )
        return None
    except Exception as e:
        logger.error(f"Unexpected error during MLflow auto-discovery: {e}")
        return None


def validate_mlflow_tracking_uri_format(tracking_uri: str) -> bool:
    """
    Validate the format of an MLflow tracking URI.

    For SageMaker, the URI should be in ARN format:
    - MLflow tracking server: arn:aws:sagemaker:REGION:ACCOUNT:mlflow-tracking-server/NAME
    - MLflow app: arn:aws:sagemaker:REGION:ACCOUNT:mlflow-app/APP-ID

    Args:
        tracking_uri: The MLflow tracking URI to validate

    Returns:
        True if the URI format is valid, False otherwise
    """
    if not tracking_uri:
        # Empty string is allowed
        return True

    # Check for MLflow tracking server ARN format
    tracking_server_pattern = r"^arn:aws:sagemaker:[a-z0-9-]+:\d{12}:mlflow-tracking-server/[a-zA-Z0-9][\w-]*$"
    if re.match(tracking_server_pattern, tracking_uri):
        return True

    # Check for MLflow app ARN format
    app_pattern = r"^arn:aws:sagemaker:[a-z0-9-]+:\d{12}:mlflow-app/app-[A-Z0-9]+$"
    if re.match(app_pattern, tracking_uri):
        return True

    return False


def validate_mlflow_arn_exists(
    tracking_uri: str, region_name: Optional[str] = None
) -> Tuple[bool, str]:
    """
    Validate if an MLflow tracking URI (ARN) exists and is accessible.

    This function validates that ARN format URIs point to existing MLflow resources.

    Args:
        tracking_uri: The MLflow tracking URI to validate (ARN format)
        region_name: AWS region name. If None, will be extracted from ARN or use default

    Returns:
        A tuple of (is_valid, message) where:
        - is_valid: True if the ARN exists and is accessible, False otherwise
        - message: Success or error message describing the validation result
    """
    if not tracking_uri:
        return False, "MLflow tracking URI is empty"

    # Check for MLflow tracking server ARN format
    tracking_server_pattern = r"^arn:aws:sagemaker:([a-z0-9-]+):(\d{12}):mlflow-tracking-server/([a-zA-Z0-9][\w-]*)$"
    app_pattern = (
        r"^arn:aws:sagemaker:([a-z0-9-]+):(\d{12}):mlflow-app/(app-[A-Z0-9]+)$"
    )

    tracking_server_match = re.match(tracking_server_pattern, tracking_uri)
    app_match = re.match(app_pattern, tracking_uri)

    if not tracking_server_match and not app_match:
        # Not an ARN format - skip existence validation
        logger.error(f"MLflow tracking URI is not an ARN format: {tracking_uri}")
        return True, "Non-ARN format URI - existence check skipped"

    # Extract components from ARN
    if tracking_server_match:
        arn_region = tracking_server_match.group(1)
        account_id = tracking_server_match.group(2)
        resource_name = tracking_server_match.group(3)
        resource_type = "tracking-server"
    elif app_match:
        arn_region = app_match.group(1)
        account_id = app_match.group(2)
        resource_name = app_match.group(3)
        resource_type = "app"
    else:
        # This should not happen as we've already checked, but needed for type safety
        return True, "Non-ARN format URI - existence check skipped"

    # Use region from ARN if not explicitly provided
    if not region_name:
        region_name = arn_region

    try:
        import boto3
        from botocore.exceptions import (
            ClientError,
            NoCredentialsError,
            NoRegionError,
        )
    except ImportError:
        logger.warning("boto3 not available, cannot validate MLflow ARN existence")
        return True, "boto3 not available - validation skipped"

    try:
        # Create SageMaker client
        if region_name:
            sagemaker_client = boto3.client("sagemaker", region_name=region_name)
        else:
            sagemaker_client = boto3.client("sagemaker")

        # Check if the MLflow resource exists
        try:
            if resource_type == "app":
                # For mlflow-app, use list_mlflow_apps and check if it exists
                response = sagemaker_client.list_mlflow_apps()

                for app in response.get("Summaries", []):
                    if app.get("Arn") == tracking_uri:
                        status = app.get("Status", "")
                        # Return success regardless of status
                        return (
                            True,
                            f"MLflow app exists (status: {status})",
                        )

                return False, f"MLflow app not found: {resource_name}"

            else:  # tracking-server
                response = sagemaker_client.list_mlflow_tracking_servers()
                # Check if any app matches the tracking server name
                for tracking_server in response.get("TrackingServerSummaries", []):
                    if (
                        tracking_server.get("TrackingServerName") == resource_name
                        or tracking_server.get("TrackingServerArn") == tracking_uri
                    ):
                        status = tracking_server.get("IsActive", "")
                        # Return success regardless of status
                        return (
                            True,
                            f"MLflow tracking server exists (status: {status})",
                        )

                return False, f"MLflow tracking server not found: {resource_name}"

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            if error_code == "AccessDeniedException":
                logger.warning(
                    "Access denied to list MLflow apps - cannot validate existence"
                )
                return True, "Access denied - assuming ARN is valid"
            else:
                return False, f"Error checking MLflow resource: {e}"

    except NoCredentialsError:
        logger.error("AWS credentials not configured, cannot validate MLflow ARN")
        return True, "AWS credentials not configured - cannot validate MLflow ARN"
    except NoRegionError:
        logger.error("AWS region not configured, cannot validate MLflow ARN")
        return True, "AWS region not configured - cannot validate MLflow ARN"
    except Exception as e:
        logger.error(f"Unexpected error validating MLflow ARN: {e}")
        return True, f"Unexpected error - assuming ARN is valid: {e}"


def validate_mlflow_overrides(
    overrides: Dict[str, Any],
    check_exists: bool = True,
    region_name: Optional[str] = None,
) -> List[str]:
    """
    Validate MLflow fields from overrides.

    This is the main validation function that should be used by validators
    to check MLflow configuration.

    Args:
        overrides: Dictionary containing override parameters
        check_exists: Whether to check if the ARN actually exists (default: True)
        region_name: AWS region name for existence check

    Returns:
        List of validation error messages (empty if valid)
    """
    if not overrides:
        return []

    errors = []

    # Get MLflow fields
    tracking_uri = overrides.get("mlflow_tracking_uri")
    experiment_name = overrides.get("mlflow_experiment_name")
    run_name = overrides.get("mlflow_run_name")

    if tracking_uri is not None and tracking_uri != "":
        # Validate format
        if not validate_mlflow_tracking_uri_format(tracking_uri):
            errors.append(
                f"Invalid MLflow tracking URI format: '{tracking_uri}'. "
                "Expected ARN format (arn:aws:sagemaker:REGION:ACCOUNT:mlflow-tracking-server/NAME "
                "or arn:aws:sagemaker:REGION:ACCOUNT:mlflow-app/APP-ID)."
            )
        elif check_exists:
            # Validate existence if format is valid and check is requested
            exists, message = validate_mlflow_arn_exists(tracking_uri, region_name)
            if not exists:
                errors.append(f"MLflow tracking URI validation failed: {message}")

    # If experiment_name or run_name is provided, tracking_uri should also be provided
    if (experiment_name or run_name) and not tracking_uri:
        logger.warning(
            "MLflow experiment_name or run_name provided without tracking_uri. "
            "MLflow tracking may not work as expected."
        )

    # Validate that experiment_name and run_name are non-empty strings if provided
    if experiment_name is not None and experiment_name == "":
        errors.append("MLflow experiment_name cannot be an empty string if provided.")

    if run_name is not None and run_name == "":
        errors.append("MLflow run_name cannot be an empty string if provided.")

    return errors
