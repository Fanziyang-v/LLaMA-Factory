# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
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
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import numpy as np
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler

from layerskip.utils import slice_str_to_array
from layerskip.layer_dropout import prepare_layer_dropout
from layerskip.early_exit_loss import (
    EarlyExitCurriculum,
    RotationalEarlyExitCurriculum,
    GradualEarlyExitCurriculum,
    early_exit_loss,
)

from transformers import LlamaForCausalLM, Qwen2ForCausalLM

if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments, TrainingArguments


logger = logging.get_logger(__name__)


class LayerSkipTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        args: "TrainingArguments",
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        gen_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: PreTrainedTokenizer = kwargs.get("tokenizer")

        super().__init__(args=args, **kwargs)
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        # early exit loss
        self.layerskip_training = args.layerskip_training
        self.always_train_last_layer = args.always_train_last_layer
        self.early_exit_loss_curriculum_name = args.early_exit_loss_curriculum
        self.early_exit_loss_scale = args.early_exit_loss_scale
        self.early_exit_loss_scale_fct = args.early_exit_loss_scale_fct
        self.do_output_hidden_states = args.do_output_hidden_states
        self.early_exit_curriculum: Optional[EarlyExitCurriculum] = None
        # layer dropout
        self.layer_dropout_prob_max = args.layer_dropout_prob_max
        self.layer_dropout_scale_fct = args.layer_dropout_scale_fct
        self.layer_dropout_layers = args.layer_dropout_layers

        self.num_hidden_layers = len(self._get_llama_or_qwen_model(self.model).model.layers)
        # Apply layer dropout to the model.
        if self.layer_dropout_prob_max > 0:
            prepare_layer_dropout(
                layers=self._get_llama_or_qwen_model(self.model).model.layers,
                prob_max=self.layer_dropout_prob_max,
                prob_layer_scale=self.layer_dropout_scale_fct,
                layers_str=self.layer_dropout_layers,
            )

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    @override
    def compute_loss(self, model, inputs, *args, **kwargs):
        # Initialize early exit curriculum if not exists
        if self.early_exit_curriculum is None:
            if self.early_exit_loss_curriculum_name == "rotational":
                self.early_exit_curriculum = RotationalEarlyExitCurriculum(
                    do_output_hidden_states=slice_str_to_array(self.do_output_hidden_states, self.num_hidden_layers),
                    max_steps=self.state.max_steps,
                    train_last_layer=self.always_train_last_layer,
                    last_step=self.state.global_step,
                )
            elif self.early_exit_loss_curriculum_name == "gradual":
                self.early_exit_curriculum = GradualEarlyExitCurriculum(
                    do_output_hidden_states=slice_str_to_array(self.do_output_hidden_states, self.num_hidden_layers),
                    max_steps=self.state.max_steps,
                    train_last_layer=self.always_train_last_layer,
                    last_step=self.state.global_step,
                )
        labels = inputs.pop("labels")
        _model = self._get_llama_or_qwen_model(model)
        outputs = model(**inputs, output_hidden_states=True)
        do_output_hidden_states = self.early_exit_curriculum.get()
        hidden_states_dict: Dict[int, torch.Tensor] = {}
        for layer_idx, hidden_states in enumerate(outputs.hidden_states[1:]):
            hidden_states = hidden_states if layer_idx != self.num_hidden_layers - 1 else _model.model.norm(hidden_states)
            if do_output_hidden_states[layer_idx]:
                hidden_states_dict[layer_idx] = hidden_states

        loss = early_exit_loss(
            _model,
            hidden_states_dict=hidden_states_dict,
            labels=labels,
            loss_fn=torch.nn.CrossEntropyLoss(),
            e_scale=self.early_exit_loss_scale,
            loss_scale_fn=self.early_exit_loss_scale_fct,
        )
        if self.state.global_step != self.early_exit_curriculum._step:
            self.early_exit_curriculum.step() # only update in new global step

        return loss

    def _get_llama_or_qwen_model(self, model: torch.nn.Module) -> Union[LlamaForCausalLM, Qwen2ForCausalLM]:
        r"""Get the LlamaForCausalLM model from the trainer."""
        for module in model.modules():
            if isinstance(module, (LlamaForCausalLM, Qwen2ForCausalLM)):
                return module
        raise AssertionError(f"LlamaForCausalLM or Qwen2ForCausalLM model not found in {model.__class__.__name__}.")


    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")
