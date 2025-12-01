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
from typing import Any, Dict, Optional

import amzn_nova_customization_sdk.recipe_config.v_one.sft_config as v1_sft
import amzn_nova_customization_sdk.recipe_config.v_two.sft_config as v2_sft
from amzn_nova_customization_sdk.model.model_enums import (
    Model,
    Platform,
    TrainingMethod,
    Version,
)
from amzn_nova_customization_sdk.recipe_builder.base_recipe_builder import (
    BaseRecipeBuilder,
)
from amzn_nova_customization_sdk.recipe_config.base_recipe_config import (
    BaseRecipeConfig,
    BaseRunConfig,
)
from amzn_nova_customization_sdk.validation.sft_validator import (
    SFTValidator,
)


class SFTRecipeBuilder(BaseRecipeBuilder):
    def __init__(
        self,
        job_name: str,
        platform: Platform,
        model: Model,
        method: TrainingMethod,
        instance_type: str,
        instance_count: int,
        data_s3_path: str,
        output_s3_path: str,
        overrides: Dict[str, Any],
        infra: Optional[Any] = None,
        model_path: Optional[str] = None,
    ):
        self.platform = platform
        self.model = model
        self.method = method
        self.version = model.version

        if model not in Model:
            raise ValueError(f"Unsupported model: {model}")

        super().__init__(
            job_name=job_name,
            platform=platform,
            method=method,
            model_type=model.model_type,
            model_path=model_path or model.model_path,
            instance_type=instance_type,
            instance_count=instance_count,
            data_s3_path=data_s3_path,
            output_s3_path=output_s3_path,
            overrides=overrides,
            infra=infra,
        )

    def _get_default_max_length(self) -> int:
        if self.instance_count == 1 or self.version == Version.TWO:
            return 8192
        return 32768

    def _get_default_global_batch_size(self) -> int:
        if self.method == TrainingMethod.SFT_FULLRANK and self.version == Version.ONE:
            return 16 if self.instance_type.startswith("ml.p5") else 64
        elif self.method == TrainingMethod.SFT_LORA and self.version == Version.ONE:
            return 32 if self.model == Model.NOVA_PRO else 64
        return 64

    def _get_default_lr(self) -> float:
        if self.method == TrainingMethod.SFT_FULLRANK:
            if self.instance_count > 1:
                return 5e-6
        return 1e-5

    def _get_default_min_lr(self) -> float:
        if self.method == TrainingMethod.SFT_FULLRANK and self.version == Version.ONE:
            if self.instance_count > 1:
                return 5e-7
        return 1e-6

    def _get_default_max_epochs(self) -> int:
        if self.method == TrainingMethod.SFT_FULLRANK and self.model != Model.NOVA_PRO:
            return 1
        return 2

    def _validate_user_input(
        self, validation_config: Optional[Dict[str, bool]] = None
    ) -> None:
        SFTValidator.validate(
            platform=self.platform,
            model=self.model,
            method=self.method,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            overrides=self.overrides,
            infra=self.infra,
            validation_config=validation_config,
        )

    def _create_sft_v1_config(self, run: BaseRunConfig) -> v1_sft.SFTRecipeConfig:
        # SFT Lora
        if self.method == TrainingMethod.SFT_LORA:
            lora_tuning = v1_sft.LoraTuningConfig(
                **{
                    k: v
                    for k, v in self.overrides.items()
                    if hasattr(v1_sft.LoraTuningConfig, k)
                }
            )
            peft = v1_sft.PeftConfig(
                peft_scheme=v1_sft.PeftScheme.LORA, lora_tuning=lora_tuning
            )
        # SFT Full
        else:
            peft = v1_sft.PeftConfig(peft_scheme=v1_sft.PeftScheme.NULL)

        sched = v1_sft.SchedConfig(
            **{
                k: v
                for k, v in self.overrides.items()
                if hasattr(v1_sft.SchedConfig, k) and k not in ("min_lr")
            },
            min_lr=self._get_value("min_lr", self._get_default_min_lr),
        )

        optim = v1_sft.OptimConfig(
            **{
                k: v
                for k, v in self.overrides.items()
                if hasattr(v1_sft.OptimConfig, k) and k not in ("lr", "sched")
            },
            lr=self._get_value("lr", self._get_default_lr),
            sched=sched,
        )

        model = v1_sft.ModelConfig(
            **{
                k: v
                for k, v in self.overrides.items()
                if hasattr(v1_sft.ModelConfig, k)
            },
            peft=peft,
            optim=optim,
        )

        trainer = v1_sft.TrainerConfig(
            max_epochs=self._get_value("max_epochs", self._get_default_max_epochs)
        )

        training_config = v1_sft.SFTTrainingConfig(
            global_batch_size=self._get_value(
                "global_batch_size", self._get_default_global_batch_size
            ),
            max_length=self._get_value("max_length", self._get_default_max_length),
            trainer=trainer,
            model=model,
        )

        return v1_sft.SFTRecipeConfig(run=run, training_config=training_config)

    def _create_sft_v2_config(self, run: BaseRunConfig) -> v2_sft.SFTRecipeConfig:
        # SFT Lora
        if self.method == TrainingMethod.SFT_LORA:
            lora_tuning = v2_sft.LoraTuningConfig(
                **{
                    k: v
                    for k, v in self.overrides.items()
                    if hasattr(v2_sft.LoraTuningConfig, k)
                }
            )
            peft = v2_sft.Peft(
                peft_scheme=v2_sft.PeftScheme.LORA, lora_tuning=lora_tuning
            )
        # SFT Full
        else:
            peft = v2_sft.Peft(peft_scheme=v2_sft.PeftScheme.NULL)

        lr_scheduler = v2_sft.LrScheduler(
            **{
                k: v
                for k, v in self.overrides.items()
                if hasattr(v2_sft.LrScheduler, k)
            }
        )

        optim_config = v2_sft.OptimConfig(
            **{
                k: v
                for k, v in self.overrides.items()
                if hasattr(v2_sft.OptimConfig, k) and k not in ("lr")
            },
            lr=self._get_value("lr", self._get_default_lr),
        )

        training_config = v2_sft.SFTTrainingConfig(
            **{
                k: v
                for k, v in self.overrides.items()
                if hasattr(v2_sft.SFTTrainingConfig, k)
                and k
                not in (
                    "lr_scheduler",
                    "optim_config",
                    "peft",
                )
            },
            lr_scheduler=lr_scheduler,
            optim_config=optim_config,
            peft=peft,
        )

        return v2_sft.SFTRecipeConfig(run=run, training_config=training_config)

    def _build_recipe_config(self) -> BaseRecipeConfig:
        run = self._create_base_run_config()
        for k, v in self.overrides.items():
            if hasattr(run, k):
                setattr(run, k, v)

        if self.version is Version.ONE:
            return self._create_sft_v1_config(run=run)
        if self.version is Version.TWO:
            return self._create_sft_v2_config(run=run)

        raise ValueError(
            f"Invalid Nova version provided for SFT training: '{self.version.value}'"
        )
