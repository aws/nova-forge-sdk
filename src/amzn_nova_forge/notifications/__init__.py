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
from amzn_nova_forge.core.enums import Platform
from amzn_nova_forge.core.result.job_result import BaseJobResult
from amzn_nova_forge.notifications.notification_manager import (
    NotificationManager,
    NotificationManagerInfraError,
)
from amzn_nova_forge.notifications.smhp_notification_manager import (
    SMHPNotificationManager,
)
from amzn_nova_forge.notifications.smtj_notification_manager import (
    SMTJNotificationManager,
)

__all__ = [
    "NotificationManager",
    "NotificationManagerInfraError",
    "SMTJNotificationManager",
    "SMHPNotificationManager",
]


def _create_notification_manager(platform, region, **kwargs):
    """Factory for notification managers, registered on BaseJobResult."""
    if platform == Platform.SMTJ:
        return SMTJNotificationManager(region=region)
    elif platform == Platform.SMHP:
        return SMHPNotificationManager(cluster_name=kwargs["cluster_name"], region=region)
    raise ValueError(f"Unsupported platform: {platform}")


BaseJobResult._register_notification_factory(_create_notification_manager)
