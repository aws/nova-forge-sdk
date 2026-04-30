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
"""Telemetry module to collect usage data and metrics."""

from __future__ import absolute_import

import functools
import inspect
import os
from typing import Optional

import boto3
import requests

from amzn_nova_forge.core.constants import REGION_TO_ESCROW_ACCOUNT_MAPPING
from amzn_nova_forge.telemetry.constants import (
    DEFAULT_AWS_REGION,
    OS_NAME_VERSION,
    PLATFORM_TO_CODE,
    PYTHON_VERSION,
    SDK_VERSION,
    TELEMETRY_OPT_OUT_MESSAGING,
    TRAINING_METHOD_TO_CODE,
    Feature,
    Status,
)
from amzn_nova_forge.util.logging import logger

_telemetry_notice_shown = False


def _is_telemetry_opted_out() -> bool:
    """Check if telemetry is opted out via environment variable."""
    return os.getenv("TELEMETRY_OPT_OUT", "False").lower() == "true"


def _telemetry_emitter(feature: Feature, func_name: str, extra_info_fn=None):
    """Decorator that emits telemetry for the wrapped function call.

    Args:
        feature: The telemetry feature category.
        func_name: Human-readable function name for telemetry.
        extra_info_fn: Optional callable that receives (*args, **kwargs) of the
            wrapped function and returns a dict of additional key-value pairs
            to include in the telemetry extra string.  Positional arguments are
            normalised to keyword arguments via ``inspect.signature.bind`` so
            that ``kwargs.get(...)`` inside the callback works regardless of
            how the caller invoked the function.
    """

    def decorator(func):
        _sig = inspect.signature(func) if extra_info_fn is not None else None

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            global _telemetry_notice_shown
            if not _telemetry_notice_shown:
                logger.info(TELEMETRY_OPT_OUT_MESSAGING)
                _telemetry_notice_shown = True

            response = None
            caught_ex = None

            # Check if telemetry is opted out
            telemetry_opt_out_flag = _is_telemetry_opted_out()
            logger.debug("TelemetryOptOut flag is set to: %s", telemetry_opt_out_flag)

            feature_code = feature.value

            extra = (
                f"&x-func={func_name}"
                f"&x-sdkVersion={SDK_VERSION}"
                f"&x-env={PYTHON_VERSION}"
                f"&x-sys={OS_NAME_VERSION}"
            )

            # Append additional extra info if provided
            if extra_info_fn is not None:
                try:
                    bound = _sig.bind_partial(*args, **kwargs)
                    bound.apply_defaults()
                    bound_kwargs = {}
                    for param_name, value in bound.arguments.items():
                        param = _sig.parameters[param_name]
                        if param.kind == param.VAR_KEYWORD:
                            # Flatten **kwargs into the top-level dict
                            bound_kwargs.update(value)
                        elif param.kind == param.VAR_POSITIONAL:
                            # Skip *args — not useful for named lookups
                            pass
                        else:
                            bound_kwargs[param_name] = value
                    additional = extra_info_fn(**bound_kwargs)
                    if additional:
                        for key, value in additional.items():
                            if key == "platform" and value in PLATFORM_TO_CODE:
                                value = PLATFORM_TO_CODE[value]
                            elif key == "method" and value in TRAINING_METHOD_TO_CODE:
                                value = TRAINING_METHOD_TO_CODE[value]
                            elif key == "dryRun" and isinstance(value, bool):
                                value = int(value)
                            extra += f"&x-{key}={value}"
                except Exception:
                    logger.debug("Failed to extract additional telemetry info")

            try:
                response = func(*args, **kwargs)
                if not telemetry_opt_out_flag:
                    try:
                        _send_telemetry_request(
                            Status.SUCCESS.value,
                            feature_code,
                            None,
                            extra,
                        )
                    except Exception:
                        logger.debug("Failed to emit success telemetry")
            except Exception as e:
                caught_ex = e
                if not telemetry_opt_out_flag:
                    try:
                        _send_telemetry_request(
                            Status.FAILURE.value,
                            feature_code,
                            e.__class__.__name__,
                            extra,
                        )
                    except Exception:
                        logger.debug("Failed to emit failure telemetry")
            finally:
                if caught_ex:
                    raise caught_ex
                return response

        return wrapper

    return decorator


def _send_telemetry_request(
    status: int,
    feature: int,
    failure_type: Optional[str] = None,
    extra_info: Optional[str] = None,
) -> None:
    """Make GET request to an empty object in S3 bucket"""
    try:
        accountId = _get_accountId()
        region = boto3.session.Session().region_name or DEFAULT_AWS_REGION

        if region not in REGION_TO_ESCROW_ACCOUNT_MAPPING:
            logger.debug(
                "Region not found in supported regions. Telemetry request will not be emitted."
            )
            return

        url = _construct_url(
            accountId,
            region,
            str(status),
            str(feature),
            failure_type,
            extra_info,
        )
        # Send the telemetry request
        logger.debug("Sending telemetry request to [%s]", url)
        _requests_helper(url, 2)
        logger.debug("Telemetry successfully emitted.")
    except Exception:
        logger.debug("Telemetry not emitted!")


def _construct_url(
    accountId: str,
    region: str,
    status: str,
    feature: str,
    failure_type: Optional[str] = None,
    extra_info: Optional[str] = None,
) -> str:
    """Construct the URL for the telemetry request"""

    base_url = (
        f"https://nfsdk-t-{region}.s3.{region}.amazonaws.com/telemetry?"
        f"x-accountId={accountId}"
        f"&x-status={status}"
        f"&x-feature={feature}"
    )
    if failure_type:
        base_url += f"&x-failureType={failure_type}"
    if extra_info:
        base_url += f"{extra_info}"

    return base_url


def _requests_helper(url, timeout):
    """Make a GET request to the given URL"""

    response = None
    try:
        response = requests.get(url, timeout=timeout)
    except requests.exceptions.RequestException as e:
        logger.debug("Request exception: %s", str(e))
    return response


def _get_accountId():
    """Return the account ID from the boto session"""

    try:
        sts = boto3.client("sts")
        return sts.get_caller_identity()["Account"]
    except Exception:
        return None
