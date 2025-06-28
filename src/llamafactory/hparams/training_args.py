# Copyright 2025 the LlamaFactory team.
#
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
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional, Union

from transformers import Seq2SeqTrainingArguments
from transformers.training_args import _convert_str_dict

from ..extras.misc import use_ray

from layerskip import EARLY_EXIT_LOSS_SCALE_FUNCTIONS

@dataclass
class RayArguments:
    r"""Arguments pertaining to the Ray training."""

    ray_run_name: Optional[str] = field(
        default=None,
        metadata={"help": "The training results will be saved at `<ray_storage_path>/ray_run_name`."},
    )
    ray_storage_path: str = field(
        default="./saves",
        metadata={"help": "The storage path to save training results to"},
    )
    ray_storage_filesystem: Optional[Literal["s3", "gs", "gcs"]] = field(
        default=None,
        metadata={"help": "The storage filesystem to use. If None specified, local filesystem will be used."},
    )
    ray_num_workers: int = field(
        default=1,
        metadata={"help": "The number of workers for Ray training. Default is 1 worker."},
    )
    resources_per_worker: Union[dict, str] = field(
        default_factory=lambda: {"GPU": 1},
        metadata={"help": "The resources per worker for Ray training. Default is to use 1 GPU per worker."},
    )
    placement_strategy: Literal["SPREAD", "PACK", "STRICT_SPREAD", "STRICT_PACK"] = field(
        default="PACK",
        metadata={"help": "The placement strategy for Ray training. Default is PACK."},
    )
    ray_init_kwargs: Optional[dict] = field(
        default=None,
        metadata={"help": "The arguments to pass to ray.init for Ray training. Default is None."},
    )

    def __post_init__(self):
        self.use_ray = use_ray()
        if isinstance(self.resources_per_worker, str) and self.resources_per_worker.startswith("{"):
            self.resources_per_worker = _convert_str_dict(json.loads(self.resources_per_worker))
        if self.ray_storage_filesystem is not None:
            if self.ray_storage_filesystem not in ["s3", "gs", "gcs"]:
                raise ValueError(
                    f"ray_storage_filesystem must be one of ['s3', 'gs', 'gcs'], got {self.ray_storage_filesystem}"
                )

            import pyarrow.fs as fs

            if self.ray_storage_filesystem == "s3":
                self.ray_storage_filesystem = fs.S3FileSystem()
            elif self.ray_storage_filesystem == "gs" or self.ray_storage_filesystem == "gcs":
                self.ray_storage_filesystem = fs.GcsFileSystem()


@dataclass
class LayerSkipArguments:
    r"""Arguments pertaining to the layer skip training."""

    layerskip_training: bool = field(
        default=False,
        metadata={"help": "Whether to use layer skip training."},
    )
    # 1. Early exit loss parameters.
    always_train_last_layer: bool = field(
        default=True,
        metadata={"help": "Whether to always train the last layer."},
    )
    early_exit_loss_curriculum: str = field(
        default="rotational",
        metadata={
            "help": "The loss curriculum for early exit. Only supports 'rotational' and 'gradual'."
        },
    )
    early_exit_loss_scale: float = field(
        default=1.0,
        metadata={"help": "The loss scale for early exit."},
    )
    early_exit_loss_scale_fct_name: str = field(
        default="uniform",
        metadata={
            "help": "The loss scale function for early exit. Only supports 'uniform', 'linear', 'sum', 'sqrt', 'inv' and 'inv_sqrt'."
        },
    )
    early_exit_loss_scale_fct: Callable = field(init=False)
    do_output_hidden_states: str = field(default="::16", metadata={"help": "Specify which layers to early exit."})
    # 2. Layer dropout parameters.
    layer_dropout_prob_max: float = field(default=0.0, metadata={"help":"Maximum layer dropout probability."})
    layer_dropout_scale_fct: str = field(default="uniform", metadata={"help": "Layer dropout scale function. Only supports 'uniform', 'exp', 'linear', 'log', 'sin', 'sigmoid' and 'step'."})
    layer_dropout_layers: Optional[str] = field(default=None, metadata={"help": "The layers to apply layer dropout to. If None, all layers will be used."})

    def __post_init__(self):
        if not self.layerskip_training:
            return  # skip validation if not using layer skip training
        if self.early_exit_loss_curriculum not in ["rotational", "gradual"]:
            raise ValueError(
                f"early_exit_loss_curriculum must be one of ['rotational', 'gradual'], got {self.early_exit_loss_curriculum}"
            )
        if self.early_exit_loss_scale_fct_name not in [
            "uniform",
            "linear",
            "sum",
            "sqrt",
            "inv",
            "inv_sqrt",
        ]:
            raise ValueError(
                f"loss_scale_fct_name must be one of ['uniform', 'linear', 'sum', 'sqrt', 'inv', 'inv_sqrt'], got {self.early_exit_loss_scale_fct_name}"
            )
        if not 0 <= self.early_exit_loss_scale <= 1:
            raise ValueError(
                f"early_exit_loss_scale must be in [0, 1], got {self.early_exit_loss_scale}"
            )
        # Initialize the early exit loss scale function here.
        self.early_exit_loss_scale_fct = EARLY_EXIT_LOSS_SCALE_FUNCTIONS[self.early_exit_loss_scale_fct_name]

@dataclass
class TrainingArguments(LayerSkipArguments, RayArguments, Seq2SeqTrainingArguments):
    r"""Arguments pertaining to the trainer."""

    def __post_init__(self):
        Seq2SeqTrainingArguments.__post_init__(self)
        RayArguments.__post_init__(self)
        LayerSkipArguments.__post_init__(self)
