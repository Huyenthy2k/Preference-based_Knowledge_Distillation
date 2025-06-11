import contextlib
import functools
import json
import os
import random
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import tqdm
import wandb
from loss.aligment_loss.distiller import Distiller
from loss.aligment_loss.dual_space_kd_with_cross_model_attention import DualSpaceKDWithCMA
from loss.loss import (
    _get_batch_logps,
    _get_batch_logps_KDtisdpo,
    concatenated_inputs,
    get_ptoken_logps,
    preference_loss,
    prompt_remove,
    token_level_dpo_loss,
)
from omegaconf import DictConfig
from torch import nn
from torch.distributed.fsdp import (
    BackwardPrefetch,
    CPUOffload,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp.api import FullStateDictConfig
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from utils.preference_datasets import CustomCollate, PrefData
from utils.utils import (
    all_gather_if_needed,
    formatted_dict,
    get_block_class_from_model,
    pad_to_length,
    rank0_print,
    slice_and_move_batch_for_device,
)

"""
from Loss_CoT-KD import DualSpaceKDWithCMA_OT_Rationale
"""


class BasicTrainer(object):
    def __init__(
        self,
        policy: nn.Module,
        seed: int,
        run_dir: str,
        config: DictConfig,
        reference_model: Optional[nn.Module] = None,
        transform_config=None,
        # transform_config không để là None được vì có transform_config.get() ở preference_datasets.py
        rank: int = 0,
        world_size: int = 1,
    ):
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.config = config
        self.run_dir = run_dir
        self.base_data_dir = config.base_data_dir

        teacher_tokenizer_name_or_path = (
            config.model.teacher_tokenizer_name_or_path or config.model.teacher_name_or_path
        )
        rank0_print(f"Loading teacher tokenizer {teacher_tokenizer_name_or_path}")
        self.teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_tokenizer_name_or_path)
        student_tokenizer_name_or_path = (
            config.model.student_tokenizer_name_or_path or config.model.student_name_or_path
        )
        rank0_print(f"Loading student tokenizer {student_tokenizer_name_or_path}")
        self.student_tokenizer = AutoTokenizer.from_pretrained(student_tokenizer_name_or_path)
        if self.teacher_tokenizer.pad_token_id is None:
            self.teacher_tokenizer.pad_token_id = self.teacher_tokenizer.eos_token_id
            # self.teacher_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        if self.student_tokenizer.pad_token_id is None:
            self.student_tokenizer.pad_token_id = self.student_tokenizer.eos_token_id
            # self.student_tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        self.tokenizer = {
            "teacher": self.teacher_tokenizer,
            "student": self.student_tokenizer,
        }

        # data_iterator_kwargs = dict(
        #     # names=config.datasets,
        #     tokenizer=self.tokenizer,
        #     shuffle=True,
        #     max_length=config.max_length,
        #     max_prompt_length=config.max_prompt_length,
        #     sft_mode=config.loss.name == "sft",
        #     seed=seed,
        #     reverse_dataset=config.reverse_dataset,
        #     base_data_dir=config.base_data_dir,
        # )

        self.policy = policy
        self.reference_model = reference_model

        # Use the passed transform_config if available
        self.transform_config = transform_config

        print(self.transform_config)

        if self.config.loss.name == "tisdpo_KDAlign":
            self.distiller = Distiller(
                student_model=self.policy,
                teacher_model=self.reference_model,
                student_tokenizer=self.tokenizer["student"],
                teacher_tokenizer=self.tokenizer["teacher"],
                config=self.config,
            )
            self.criterion = DualSpaceKDWithCMA(args=self.config.loss)

        self.train_dataset = PrefData(
            data_path=config.datasets,
            train_test_split="train",
            sft_mode=(config.loss.name == "sft"),
            reverse_dataset=config.reverse_dataset,
            transform_config=self.transform_config,
        )
        self.eval_dataset = PrefData(
            data_path=config.datasets,
            train_test_split="test",
            sft_mode=(config.loss.name == "sft"),
            reverse_dataset=config.reverse_dataset,
            transform_config=self.transform_config,
        )

        self.train_iterator = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            # collate_fn=get_collate_fn(self.tokenizer),
            collate_fn=CustomCollate(self.tokenizer),
            pin_memory=True,
            num_workers=2,
            drop_last=True,
        )
        rank0_print("Loaded train data iterator")
        # self.train_iterator = get_batch_iterator(
        #     **data_iterator_kwargs,
        #     split="train",
        #     n_epochs=config.n_epochs,
        #     n_examples=config.n_examples,
        #     batch_size=config.batch_size,
        #     silent=rank != 0,
        #     transform_config=transform_config,
        # )
        self.eval_iterator = DataLoader(
            self.eval_dataset,
            batch_size=config.eval_batch_size,
            shuffle=True,
            # collate_fn=get_collate_fn(self.tokenizer),
            collate_fn=CustomCollate(self.tokenizer),
            pin_memory=True,
            num_workers=2,
            drop_last=True,
        )
        # self.eval_iterator = get_batch_iterator(
        #     **data_iterator_kwargs,
        #     split="test",
        #     n_examples=config.n_eval_examples,
        #     batch_size=config.eval_batch_size,
        #     silent=rank != 0,
        #     transform_config=transform_config,
        # )
        self.eval_batches = list(self.eval_iterator)
        rank0_print(
            f"Loaded {len(self.eval_batches)} eval batches of size {config.eval_batch_size}"
        )

    def get_batch_samples(self, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        """Generate samples from the policy (and reference model, if doing DPO training) for the given batch of inputs."""

        # FSDP generation according to https://github.com/pytorch/pytorch/issues/100069
        ctx = lambda: (
            FSDP.summon_full_params(self.policy, writeback=False, recurse=False)
            if "FSDP" in self.config.trainer
            else contextlib.nullcontext()
        )
        with ctx():
            policy_output = self.policy.generate(
                batch[f"prompt_{self.config.policy_mode}_input_ids"],
                attention_mask=batch[f"prompt_{self.config.policy_mode}_attention_mask"],
                max_length=self.config.max_length,
                do_sample=True,
                pad_token_id=self.tokenizer[self.config.policy_mode].pad_token_id,
            )

        if self.config.loss.name in {"dpo", "ipo", "tdpo", "tisdpo", "tisdpo_KDAlign", "KD_tisdpo"}:
            ctx = lambda: (
                FSDP.summon_full_params(self.reference_model, writeback=False, recurse=False)
                if "FSDP" in self.config.trainer
                else contextlib.nullcontext()
            )
            with ctx():
                reference_output = self.reference_model.generate(
                    batch[f"prompt_{self.config.reference_mode}_input_ids"],
                    attention_mask=batch[f"prompt_{self.config.reference_mode}_attention_mask"],
                    max_length=self.config.max_length,
                    do_sample=True,
                    pad_token_id=self.tokenizer[self.config.reference_mode].pad_token_id,
                )

        policy_output = pad_to_length(
            policy_output,
            self.config.max_length,
            self.tokenizer[self.config.policy_mode].pad_token_id,
        )
        policy_output = all_gather_if_needed(policy_output, self.rank, self.world_size)
        policy_output_decoded = self.tokenizer[self.config.policy_mode].batch_decode(
            policy_output, skip_special_tokens=True
        )

        if self.config.loss.name in {"dpo", "ipo", "tdpo", "tisdpo", "tisdpo_KDAlign", "KD_tisdpo"}:
            reference_output = pad_to_length(
                reference_output,
                self.config.max_length,
                self.tokenizer[self.config.reference_mode].pad_token_id,
            )
            reference_output = all_gather_if_needed(reference_output, self.rank, self.world_size)
            reference_output_decoded = self.tokenizer[self.config.reference_mode].batch_decode(
                reference_output, skip_special_tokens=True
            )
        else:
            reference_output_decoded = []

        return policy_output_decoded, reference_output_decoded

    def concatenated_forward(
        self, model: nn.Module, batch: Dict, mode: str
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """

        concatenated_batch = concatenated_inputs(batch, mode)
        # dict_keys(['concatenated_weight', f'concatenated_{mode}_input_ids', f'concatenated_{mode}_attention_mask', f'concatenated_{mode}_labels', f'concatenated_{mode}_parent_dict'])
        all_logits = model(
            concatenated_batch[f"concatenated_{mode}_input_ids"],
            attention_mask=concatenated_batch[f"concatenated_{mode}_attention_mask"],
        ).logits.to(torch.float32)
        all_logps = _get_batch_logps(
            all_logits,
            concatenated_batch[f"concatenated_{mode}_labels"],
            concatenated_batch["concatenated_weight"],
            average_log_prob=False,
            token_level=False,
        )
        chosen_logps = all_logps[: batch[f"chosen_{mode}_input_ids"].shape[0]]
        rejected_logps = all_logps[batch[f"chosen_{mode}_input_ids"].shape[0] :]
        return chosen_logps, rejected_logps

    def KD_tisdpo_concatenated_forward(
        self,
        model: nn.Module,
        # reference_model: nn.Module,
        batch: Dict,
        mode: str,
        # reference_mode: str,
    ):
        """Run the policy model and the reference model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = concatenated_inputs(batch, mode)
        # reference_concatenated_batch = concatenated_inputs(batch, reference_mode)
        # dict_keys(
        #     [
        #         "concatenated_weight",
        #         f"concatenated_{mode}_input_ids",
        #         f"concatenated_{mode}_attention_mask",
        #         f"concatenated_{mode}_labels",
        #         f"concatenated_{mode}_parent_dict",
        #     ]
        # )
        # Inside KD_tisdpo_concatenated_forward, before the model call:
        input_ids_to_model = concatenated_batch[f"concatenated_{mode}_input_ids"]
        attention_mask_to_model = concatenated_batch[f"concatenated_{mode}_attention_mask"]
        current_rank = torch.distributed.get_rank()

        print(
            f"[Rank {current_rank}] Device of input_ids before model call: {input_ids_to_model.device}"
        )
        print(
            f"[Rank {current_rank}] Device of attention_mask before model call: {attention_mask_to_model.device}"
        )
        print(f"Device of model's first parameter: {next(model.parameters()).device}")
        print("--- END DEVICE CHECK ---")
        all_logits = model(
            concatenated_batch[f"concatenated_{mode}_input_ids"],
            attention_mask=concatenated_batch[f"concatenated_{mode}_attention_mask"],
        ).logits.to(torch.float32)  # (2*batchsize) * seq len * vocab size

        all_logits = torch.log_softmax(all_logits, dim=-1)
        r = torch.distributed.get_rank()
        print("===================================")
        print(f"[Rank {r}] all_logits shape: {all_logits.shape}")
        print(
            f"[Rank {r}] all_labels shape (before gather): {concatenated_batch[f'concatenated_{mode}_labels'].shape}"
        )
        print(
            f"[Rank {r}] all_labels min value: {concatenated_batch[f'concatenated_{mode}_labels'].min()}"
        )
        print(
            f"[Rank {r}] all_labels max value: {concatenated_batch[f'concatenated_{mode}_labels'].max()}"
        )
        print(f"[Rank {r}] Expected max label index: {all_logits.shape[-1] - 1}")  # vocab_size - 1
        print("==================================")

        all_no_prompt_logits, all_no_prompt_labels = prompt_remove(
            all_logits,
            concatenated_batch[f"concatenated_{mode}_labels"],
            concatenated_batch[f"concatenated_{mode}_input_ids"],
        )
        print("==================================")
        print(f"[Rank {r}] no_prompt_logits shape: {all_no_prompt_logits.shape}")
        print(f"[Rank {r}] no_prompt_labels shape (before gather): {all_no_prompt_labels.shape}")
        print(f"[Rank {r}] no_prompt_labels min value: {all_no_prompt_labels.min()}")
        print(f"[Rank {r}] no_prompt_labels max value: {all_no_prompt_labels.max()}")
        print(
            f"[Rank {r}] Expected max label (no_prompt) index: {all_no_prompt_logits.shape[-1] - 1}"
        )  # vocab_size - 1
        print("==================================")

        all_ptoken_logps = get_ptoken_logps(
            all_no_prompt_logits,
            all_no_prompt_labels,
            concatenated_batch[f"concatenated_{mode}_parent_list"],
        )

        chosen_ptoken_logps = all_ptoken_logps[: batch[f"chosen_{mode}_input_ids"].shape[0]]
        rejected_ptoken_logps = all_ptoken_logps[batch[f"chosen_{mode}_input_ids"].shape[0] :]
        padded_chosen_weight = concatenated_batch["concatenated_weight"][
            : batch[f"chosen_{mode}_input_ids"].shape[0]
        ]
        padded_rejected_weight = concatenated_batch["concatenated_weight"][
            batch[f"chosen_{mode}_input_ids"].shape[0] :
        ]

        return (
            chosen_ptoken_logps,
            rejected_ptoken_logps,
            padded_chosen_weight,
            padded_rejected_weight,
        )

        # with torch.no_grad():
        #     reference_all_logits = reference_model(
        #         reference_concatenated_batch[f"concatenated_{reference_mode}_input_ids"],
        #         attention_mask=reference_concatenated_batch[f"concatenated_{reference_mode}_attention_mask"],
        #     ).logits.to(torch.float32)

        # reference_all_logits = torch.log_softmax(reference_all_logits, dim=-1)
        # reference_all_no_prompt_logits, reference_all_no_prompt_labels = prompt_remove(
        #     reference_all_logits,
        #     reference_concatenated_batch[f"concatenated_{reference_mode}_labels"],
        #     reference_concatenated_batch[f"concatenated_{reference_mode}_input_ids"],
        # )

        # reference_all_ptoken_logps = get_ptoken_logps(
        #     reference_all_no_prompt_logits,
        #     reference_all_no_prompt_labels,
        #     reference_concatenated_batch[f"concatenated_{reference_mode}_parent_dict"],
        # )

        # all_logps_margin, all_logps = _get_batch_logps_KDtisdpo(
        #     all_ptoken_logps, reference_all_ptoken_logps, concatenated_batch["concatenated_weight"]
        # )

        # chosen_logps_margin = all_logps_margin[: batch[f"chosen_{mode}_input_ids"].shape[0]]
        # rejected_logps_margin = all_logps_margin[batch[f"chosen_{mode}_input_ids"].shape[0] :]
        # # chosen_position_kl = all_position_kl[:batch['chosen_input_ids'].shape[0]]
        # # rejected_position_kl = all_position_kl[batch['chosen_input_ids'].shape[0]:]

        # chosen_logps = all_logps[: batch[f"chosen_{mode}_input_ids"].shape[0]].detach()
        # rejected_logps = all_logps[batch[f"chosen_{mode}_input_ids"].shape[0] :].detach()

        # return chosen_logps_margin, rejected_logps_margin, chosen_logps, rejected_logps

    def KD_tisdpo_DSKD_concatenated_forward(
        self,
        model: nn.Module,
        # reference_model: nn.Module,
        batch: Dict,
        mode: str,
        # reference_mode: str,
    ):
        """Run the policy model and the reference model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = concatenated_inputs(batch, mode)
        # reference_concatenated_batch = concatenated_inputs(batch, reference_mode)
        # dict_keys(
        #     [
        #         "concatenated_weight",
        #         f"concatenated_{mode}_input_ids",
        #         f"concatenated_{mode}_attention_mask",
        #         f"concatenated_{mode}_labels",
        #         f"concatenated_{mode}_parent_dict",
        #     ]
        # )
        all_out = model(
            concatenated_batch[f"concatenated_{mode}_input_ids"],
            attention_mask=concatenated_batch[f"concatenated_{mode}_attention_mask"],
            output_hidden_states=True,
        )  # (2*batchsize) * seq len * vocab size

        all_logits = all_out.logits.to(torch.float32)  # (2*batchsize) * seq len * vocab size
        all_logits = torch.log_softmax(all_logits, dim=-1)
        all_no_prompt_logits, all_no_prompt_labels = prompt_remove(
            all_logits,
            concatenated_batch[f"concatenated_{mode}_labels"],
            concatenated_batch[f"concatenated_{mode}_input_ids"],
        )

        all_ptoken_logps = get_ptoken_logps(
            all_no_prompt_logits,
            all_no_prompt_labels,
            concatenated_batch[f"concatenated_{mode}_parent_list"],
        )

        chosen_ptoken_logps = all_ptoken_logps[: batch[f"chosen_{mode}_input_ids"].shape[0]]
        rejected_ptoken_logps = all_ptoken_logps[batch[f"chosen_{mode}_input_ids"].shape[0] :]
        padded_chosen_weight = concatenated_batch["concatenated_weight"][
            : batch[f"chosen_{mode}_input_ids"].shape[0]
        ]
        padded_rejected_weight = concatenated_batch["concatenated_weight"][
            batch[f"chosen_{mode}_input_ids"].shape[0] :
        ]

        chosen_logits = all_logits[: batch[f"chosen_{mode}_input_ids"].shape[0]]
        chosen_hidden_state = all_out.hidden_states[-1][
            : batch[f"chosen_{mode}_input_ids"].shape[0]
        ]

        return (
            chosen_ptoken_logps,
            rejected_ptoken_logps,
            padded_chosen_weight,
            padded_rejected_weight,
            chosen_logits,
            chosen_hidden_state,
        )

    def get_batch_metrics(
        self, batch: Dict[str, Union[List, torch.LongTensor]], loss_config, train=True
    ):
        """Compute the SFT or DPO loss and other metrics for the given batch of inputs."""

        metrics = {}
        train_test = "train" if train else "eval"

        if loss_config.name in {"dpo", "ipo"}:
            policy_chosen_logps, policy_rejected_logps = self.concatenated_forward(
                self.policy, batch, mode=self.config.policy_mode
            )
            with torch.no_grad():
                reference_chosen_logps, reference_rejected_logps = self.concatenated_forward(
                    self.reference_model, batch, mode=self.config.reference_mode
                )

            if loss_config.name == "dpo":
                loss_kwargs = {
                    "beta": loss_config.beta,
                    "reference_free": loss_config.reference_free,
                    "label_smoothing": loss_config.label_smoothing,
                    "ipo": False,
                }
            elif loss_config.name == "ipo":
                loss_kwargs = {"beta": loss_config.beta, "ipo": True}
            else:
                raise ValueError(f"unknown loss {loss_config.name}")

            losses, chosen_rewards, rejected_rewards = preference_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
                **loss_kwargs,
            )

            reward_accuracies = (chosen_rewards > rejected_rewards).float()

            chosen_rewards = all_gather_if_needed(chosen_rewards, self.rank, self.world_size)
            rejected_rewards = all_gather_if_needed(rejected_rewards, self.rank, self.world_size)
            reward_accuracies = all_gather_if_needed(reward_accuracies, self.rank, self.world_size)

            metrics[f"rewards_{train_test}/chosen"] = chosen_rewards.cpu().numpy().tolist()
            metrics[f"rewards_{train_test}/rejected"] = rejected_rewards.cpu().numpy().tolist()
            metrics[f"rewards_{train_test}/accuracies"] = reward_accuracies.cpu().numpy().tolist()
            metrics[f"rewards_{train_test}/margins"] = (
                (chosen_rewards - rejected_rewards).cpu().numpy().tolist()
            )

            policy_rejected_logps = all_gather_if_needed(
                policy_rejected_logps.detach(), self.rank, self.world_size
            )
            metrics[f"logps_{train_test}/rejected"] = policy_rejected_logps.cpu().numpy().tolist()
        # elif loss_config.name == 'tdpo':
        #     chosen_logps_margin, rejected_logps_margin, chosen_position_kl, rejected_position_kl, policy_chosen_logps, policy_rejected_logps\
        #         = self.tdpo_concatenated_forward(self.policy, self.reference_model, batch)
        #     losses, chosen_rewards, rejected_rewards = tdpo_loss(chosen_logps_margin, rejected_logps_margin,
        #                                                          chosen_position_kl, rejected_position_kl,
        #                                                          beta=loss_config.beta, alpha=loss_config.alpha, if_tdpo2=loss_config.if_tdpo2)

        #     reward_accuracies = (chosen_rewards > rejected_rewards).float()

        #     chosen_rewards = all_gather_if_needed(chosen_rewards, self.rank, self.world_size)
        #     rejected_rewards = all_gather_if_needed(rejected_rewards, self.rank, self.world_size)
        #     reward_accuracies = all_gather_if_needed(reward_accuracies, self.rank, self.world_size)

        #     metrics[f'rewards_{train_test}/chosen'] = chosen_rewards.cpu().numpy().tolist()
        #     metrics[f'rewards_{train_test}/rejected'] = rejected_rewards.cpu().numpy().tolist()
        #     metrics[f'rewards_{train_test}/accuracies'] = reward_accuracies.cpu().numpy().tolist()
        #     metrics[f'rewards_{train_test}/margins'] = (chosen_rewards - rejected_rewards).cpu().numpy().tolist()

        #     all_device_chosen_position_kl = all_gather_if_needed(chosen_position_kl.detach(), self.rank, self.world_size)
        #     all_device_rejected_position_kl = all_gather_if_needed(rejected_position_kl.detach(), self.rank, self.world_size)

        #     metrics[f'kl_{train_test}/chosen'] = all_device_chosen_position_kl.cpu().numpy().tolist()
        #     metrics[f'kl_{train_test}/rejected'] = all_device_rejected_position_kl.cpu().numpy().tolist()
        #     metrics[f'kl_{train_test}/margin'] = (all_device_chosen_position_kl - all_device_rejected_position_kl).cpu().numpy().tolist()

        #     policy_rejected_logps = all_gather_if_needed(policy_rejected_logps.detach(), self.rank, self.world_size)
        #     metrics[f'logps_{train_test}/rejected'] = policy_rejected_logps.cpu().numpy().tolist()
        elif loss_config.name == "KD_tisdpo":
            (
                policy_chosen_ptoken_logps,
                policy_rejected_ptoken_logps,
                policy_chosen_weights,
                policy_rejected_weights,
            ) = self.KD_tisdpo_concatenated_forward(
                self.policy, batch, mode=self.config.policy_mode
            )
            with torch.no_grad():
                (
                    reference_chosen_ptoken_logps,
                    reference_rejected_ptoken_logps,
                    reference_chosen_weights,
                    reference_rejected_weights,
                ) = self.KD_tisdpo_concatenated_forward(
                    self.reference_model, batch, mode=self.config.reference_mode
                )
            print("===DEBUGGING===")

            print(f"policy chosen ptoken logps: {policy_chosen_ptoken_logps.shape}")
            print(f"reference chosen ptoken logps: {reference_chosen_ptoken_logps.shape}")
            print(f"policy chosen weight: {policy_chosen_weights.shape}")
            print(f"reference chosen weight: {reference_chosen_weights.shape}")

            chosen_logps_margin, policy_chosen_logps = _get_batch_logps_KDtisdpo(
                policy_chosen_ptoken_logps, reference_chosen_ptoken_logps, policy_chosen_weights
            )
            print("===DEBUGGING===")
            print(f"policy rejected ptoken logps: {policy_rejected_ptoken_logps.shape}")
            print(f"reference rejected ptoken logps: {reference_rejected_ptoken_logps.shape}")
            print(f"policy rejected weight: {policy_rejected_weights.shape}")
            print(f"reference rejected weight: {reference_rejected_weights.shape}")

            rejected_logps_margin, policy_rejected_logps = _get_batch_logps_KDtisdpo(
                policy_rejected_ptoken_logps,
                reference_rejected_ptoken_logps,
                policy_rejected_weights,
            )

            losses, chosen_rewards, rejected_rewards = token_level_dpo_loss(
                chosen_logps_margin, rejected_logps_margin, beta=loss_config.beta
            )

            reward_accuracies = (chosen_rewards > rejected_rewards).float()

            chosen_rewards = all_gather_if_needed(chosen_rewards, self.rank, self.world_size)
            rejected_rewards = all_gather_if_needed(rejected_rewards, self.rank, self.world_size)
            reward_accuracies = all_gather_if_needed(reward_accuracies, self.rank, self.world_size)

            metrics[f"rewards_{train_test}/chosen"] = chosen_rewards.cpu().numpy().tolist()
            metrics[f"rewards_{train_test}/rejected"] = rejected_rewards.cpu().numpy().tolist()
            metrics[f"rewards_{train_test}/accuracies"] = reward_accuracies.cpu().numpy().tolist()
            metrics[f"rewards_{train_test}/margins"] = (
                (chosen_rewards - rejected_rewards).cpu().numpy().tolist()
            )

            # all_device_chosen_position_kl = all_gather_if_needed(chosen_position_kl.detach(), self.rank, self.world_size)
            # all_device_rejected_position_kl = all_gather_if_needed(rejected_position_kl.detach(), self.rank, self.world_size)

            # metrics[f'kl_{train_test}/chosen'] = all_device_chosen_position_kl.cpu().numpy().tolist()
            # metrics[f'kl_{train_test}/rejected'] = all_device_rejected_position_kl.cpu().numpy().tolist()
            # metrics[f'kl_{train_test}/margin'] = (all_device_chosen_position_kl - all_device_rejected_position_kl).cpu().numpy().tolist()

            policy_rejected_logps = all_gather_if_needed(
                policy_rejected_logps.detach(), self.rank, self.world_size
            )
            metrics[f"logps_{train_test}/rejected"] = policy_rejected_logps.cpu().numpy().tolist()

        elif loss_config.name == "tisdpo_KDAlign":
            (
                policy_chosen_ptoken_logps,
                policy_rejected_ptoken_logps,
                policy_chosen_weights,
                policy_rejected_weights,
                student_chosen_logits,
                student_chosen_last_hidden_state,
            ) = self.KD_tisdpo_DSKD_concatenated_forward(
                self.policy, batch, mode=self.config.policy_mode
            )
            with torch.no_grad():
                (
                    reference_chosen_ptoken_logps,
                    reference_rejected_ptoken_logps,
                    reference_chosen_weights,
                    reference_rejected_weights,
                    teacher_chosen_logits,
                    teacher_chosen_last_hidden_state,
                ) = self.KD_tisdpo_DSKD_concatenated_forward(
                    self.reference_model, batch, mode=self.config.reference_mode
                )

            chosen_logps_margin, policy_chosen_logps = _get_batch_logps_KDtisdpo(
                policy_chosen_ptoken_logps, reference_chosen_ptoken_logps, policy_chosen_weights
            )

            rejected_logps_margin, policy_rejected_logps = _get_batch_logps_KDtisdpo(
                policy_rejected_ptoken_logps,
                reference_rejected_ptoken_logps,
                policy_rejected_weights,
            )

            dpo_losses, chosen_rewards, rejected_rewards = token_level_dpo_loss(
                chosen_logps_margin, rejected_logps_margin, beta=loss_config.beta
            )

            reward_accuracies = (chosen_rewards > rejected_rewards).float()

            chosen_rewards = all_gather_if_needed(chosen_rewards, self.rank, self.world_size)
            rejected_rewards = all_gather_if_needed(rejected_rewards, self.rank, self.world_size)
            reward_accuracies = all_gather_if_needed(reward_accuracies, self.rank, self.world_size)

            metrics[f"rewards_{train_test}/chosen"] = chosen_rewards.cpu().numpy().tolist()
            metrics[f"rewards_{train_test}/rejected"] = rejected_rewards.cpu().numpy().tolist()
            metrics[f"rewards_{train_test}/accuracies"] = reward_accuracies.cpu().numpy().tolist()
            metrics[f"rewards_{train_test}/margins"] = (
                (chosen_rewards - rejected_rewards).cpu().numpy().tolist()
            )

            # all_device_chosen_position_kl = all_gather_if_needed(chosen_position_kl.detach(), self.rank, self.world_size)
            # all_device_rejected_position_kl = all_gather_if_needed(rejected_position_kl.detach(), self.rank, self.world_size)

            # metrics[f'kl_{train_test}/chosen'] = all_device_chosen_position_kl.cpu().numpy().tolist()
            # metrics[f'kl_{train_test}/rejected'] = all_device_rejected_position_kl.cpu().numpy().tolist()
            # metrics[f'kl_{train_test}/margin'] = (all_device_chosen_position_kl - all_device_rejected_position_kl).cpu().numpy().tolist()

            policy_rejected_logps = all_gather_if_needed(
                policy_rejected_logps.detach(), self.rank, self.world_size
            )
            metrics[f"logps_{train_test}/rejected"] = policy_rejected_logps.cpu().numpy().tolist()

            model_outputs = {
                "logits": student_chosen_logits,
                "last_hidden_states": student_chosen_last_hidden_state,
            }
            reference_outputs = {
                "logits": teacher_chosen_logits,
                "last_hidden_states": teacher_chosen_last_hidden_state,
            }
            dskd_log = {}

            dskd_loss, dskd_log = self.distiller(
                self.criterion,
                model_outputs=model_outputs,
                teacher_model_outputs=reference_outputs,
                batch=batch,
                logging_output=dskd_log,
            )

            losses = dpo_losses + (loss_config.alpha) * dskd_loss
            for k, v in dskd_log.items():
                metrics[f"dskd_{train_test}/{k}"] = v.cpu().numpy().tolist()

        elif loss_config.name == "sft":
            policy_chosen_logits = self.policy(
                batch[f"chosen_{self.config.policy_mode}_input_ids"],
                attention_mask=batch[f"chosen_{self.config.policy_mode}_attention_mask"],
            ).logits.to(torch.float32)
            policy_chosen_logps = _get_batch_logps(
                policy_chosen_logits,
                batch[f"chosen_{self.config.policy_mode}_labels"],
                average_log_prob=False,
                token_level=False,
            )

            losses = -policy_chosen_logps

        policy_chosen_logps = all_gather_if_needed(
            policy_chosen_logps.detach(), self.rank, self.world_size
        )
        metrics[f"logps_{train_test}/chosen"] = policy_chosen_logps.cpu().numpy().tolist()

        all_devices_losses = all_gather_if_needed(losses.detach(), self.rank, self.world_size)
        metrics[f"loss/{train_test}"] = all_devices_losses.cpu().numpy().tolist()

        return losses.mean(), metrics

    def train(self):
        """Begin either SFT or DPO training, with periodic evaluation."""

        rank0_print(f"Using {self.config.optimizer} optimizer")
        self.optimizer = getattr(torch.optim, self.config.optimizer)(
            self.policy.parameters(), lr=self.config.lr
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(1.0, (step + 1) / (self.config.warmup_steps + 1)),
        )

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        if self.config.loss.name in {"dpo", "ipo", "tdpo", "tisdpo", "KD_tisdpo"}:
            self.reference_model.eval()

        self.example_counter = 0
        self.batch_counter = 0
        last_log = None

        for epoch in range(self.config.n_epochs):
            for batch in self.train_iterator:
                #### BEGIN EVALUATION ####
                if self.batch_counter % self.config.eval_every == 0 and (
                    self.batch_counter > 0 or self.config.do_first_eval
                ):
                    rank0_print(f"Running evaluation after {self.example_counter} train examples")
                    self.policy.eval()

                    all_eval_metrics = defaultdict(list)
                    if self.config.sample_during_eval:
                        all_policy_samples, all_reference_samples = [], []
                        policy_text_table = wandb.Table(columns=["step", "prompt", "sample"])
                        if self.config.loss.name in {"dpo", "ipo", "tdpo", "tisdpo", "KD_tisdpo"}:
                            reference_text_table = wandb.Table(columns=["step", "prompt", "sample"])

                    for eval_batch in (
                        tqdm.tqdm(self.eval_batches, desc="Computing eval metrics")
                        if self.rank == 0
                        else self.eval_batches
                    ):
                        local_eval_batch = slice_and_move_batch_for_device(
                            eval_batch, self.rank, self.world_size, self.rank
                        )  ## needs further testing
                        with torch.no_grad():
                            _, eval_metrics = self.get_batch_metrics(
                                local_eval_batch, self.config.loss, train=False
                            )

                        for k, v in eval_metrics.items():
                            all_eval_metrics[k].extend(v)

                    if self.config.sample_during_eval:
                        if self.config.n_eval_model_samples < self.config.eval_batch_size:
                            rank0_print(
                                f"Warning: n_eval_model_samples ({self.config.n_eval_model_samples}) < eval_batch_size ({self.config.eval_batch_size}). Sampling from the first complete eval batch of prompts."
                            )
                            sample_batches = self.eval_batches[:1]
                        else:
                            n_sample_batches = (
                                self.config.n_eval_model_samples // self.config.eval_batch_size
                            )
                            sample_batches = self.eval_batches[:n_sample_batches]
                        for eval_batch in (
                            tqdm.tqdm(sample_batches, desc="Generating samples...")
                            if self.rank == 0
                            else sample_batches
                        ):
                            local_eval_batch = slice_and_move_batch_for_device(
                                eval_batch, self.rank, self.world_size, self.rank
                            )
                            policy_samples, reference_samples = self.get_batch_samples(
                                local_eval_batch
                            )

                            all_policy_samples.extend(policy_samples)
                            all_reference_samples.extend(reference_samples)

                            for prompt, sample in zip(eval_batch["prompt"], policy_samples):
                                policy_text_table.add_data(self.example_counter, prompt, sample)
                            if self.config.loss.name in {"dpo", "ipo", "tdpo", "tisdpo"}:
                                for prompt, sample in zip(eval_batch["prompt"], reference_samples):
                                    reference_text_table.add_data(
                                        self.example_counter, prompt, sample
                                    )

                    mean_eval_metrics = {k: sum(v) / len(v) for k, v in all_eval_metrics.items()}
                    rank0_print(
                        f"eval after {self.example_counter}: {formatted_dict(mean_eval_metrics)}"
                    )
                    if self.config.sample_during_eval:
                        rank0_print(json.dumps(all_policy_samples[:10], indent=2))
                        if self.config.loss.name in {"dpo", "ipo", "tdpo", "tisdpo"}:
                            rank0_print(json.dumps(all_reference_samples[:10], indent=2))

                    if self.config.wandb.enabled and self.rank == 0:
                        wandb.log(mean_eval_metrics, step=self.example_counter)

                        if self.config.sample_during_eval:
                            wandb.log(
                                {"policy_samples": policy_text_table}, step=self.example_counter
                            )
                            if self.config.loss.name in {"dpo", "ipo", "tdpo", "tisdpo"}:
                                wandb.log(
                                    {"reference_samples": reference_text_table},
                                    step=self.example_counter,
                                )
                    self.save_checkpoint(step=self.batch_counter)
                #### END EVALUATION ####

                #### BEGIN TRAINING ####
                self.policy.train()

                start_time = time.time()
                batch_metrics = defaultdict(list)
                for microbatch_idx in range(self.config.gradient_accumulation_steps):
                    print("============================================")
                    for k in batch:
                        if isinstance(batch[k], torch.Tensor):
                            print(f"{k}_size BATCH SIZE", batch[k].shape)
                    global_microbatch = slice_and_move_batch_for_device(
                        batch, microbatch_idx, self.config.gradient_accumulation_steps, self.rank
                    )
                    for k in global_microbatch:
                        if isinstance(global_microbatch[k], torch.Tensor):
                            print(f"{k}_size GLOBAL MICROBATCH", global_microbatch[k].shape)
                    local_microbatch = slice_and_move_batch_for_device(
                        global_microbatch, self.rank, self.world_size, self.rank
                    )
                    for k in local_microbatch:
                        if isinstance(local_microbatch[k], torch.Tensor):
                            print(f"{k}_size LOCAL MICROBATCH", local_microbatch[k].shape)
                    print("==============================================")
                    # print(f"prompt {local_microbatch['prompt'][0]}")
                    loss, metrics = self.get_batch_metrics(
                        local_microbatch, self.config.loss, train=True
                    )
                    """
                    loss_cot_kd = DualSpaceKDWithCMA_OT_Rationale()
                    loss = loss + alpha*loss_cot_kd
                    """
                    (loss / self.config.gradient_accumulation_steps).backward()

                    for k, v in metrics.items():
                        batch_metrics[k].extend(v)

                grad_norm = self.clip_gradient()
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                step_time = time.time() - start_time
                examples_per_second = self.config.batch_size / step_time
                batch_metrics["examples_per_second"].append(examples_per_second)
                batch_metrics["grad_norm"].append(grad_norm)

                self.batch_counter += 1
                self.example_counter += self.config.batch_size

                if (
                    last_log is None
                    or time.time() - last_log > self.config.minimum_log_interval_secs
                ):
                    mean_train_metrics = {k: sum(v) / len(v) for k, v in batch_metrics.items()}
                    mean_train_metrics["counters/examples"] = self.example_counter
                    mean_train_metrics["counters/updates"] = self.batch_counter
                    rank0_print(
                        f"train stats after {self.example_counter} examples: {formatted_dict(mean_train_metrics)}"
                    )

                    if self.config.wandb.enabled and self.rank == 0:
                        wandb.log(mean_train_metrics, step=self.example_counter)

                    last_log = time.time()
                else:
                    rank0_print(
                        f"skipping logging after {self.example_counter} examples to avoid logging too frequently"
                    )
                #### END TRAINING ####

    def clip_gradient(self):
        """Clip the gradient norm of the parameters of a non-FSDP policy."""
        return torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(), self.config.max_grad_norm
        ).item()

    def write_state_dict(
        self,
        step: int,
        state: Dict[str, torch.Tensor],
        metrics: Dict,
        filename: str,
        dir_name: Optional[str] = None,
    ):
        """Write a checkpoint to disk."""
        if dir_name is None:
            dir_name = os.path.join(self.run_dir, "LATEST")

        os.makedirs(dir_name, exist_ok=True)
        output_path = os.path.join(dir_name, filename)
        rank0_print(f"writing checkpoint to {output_path}...")
        torch.save(
            {
                "step_idx": step,
                "state": state,
                "metrics": metrics if metrics is not None else {},
            },
            output_path,
        )

    def save_checkpoint(self, step: int, output_dir: Optional[str] = None):
        """Save a checkpoint"""
        if output_dir is None:
            model_save_dir = os.path.join(self.run_dir, str(step))
        else:
            model_save_dir = output_dir

        os.makedirs(model_save_dir, exist_ok=True)
        self.policy.save_pretrained(model_save_dir)
        rank0_print(f"Checkpoint saved to {model_save_dir} using save_pretrained")

        self.policy.push_to_hub(
            repo_id=f"{self.config.save_repo}_step_{step}",
            private=False,
            commit_message="Save checkpoint",
            token="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",
        )
        self.tokenizer[self.config.policy_mode].push_to_hub(
            repo_id=f"{self.config.save_repo}_step_{step}",
            private=False,
            commit_message="Save checkpoint",
            token="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",
        )

    def save(self, output_dir: Optional[str] = None, metrics: Optional[Dict] = None):
        """Save policy and tokenizer to disk."""
        if output_dir is None:
            model_save_dir = os.path.join(self.run_dir, "LATEST")
        else:
            model_save_dir = output_dir

        os.makedirs(model_save_dir, exist_ok=True)

        # Save model using transformers save_pretrained
        self.policy.save_pretrained(model_save_dir)
        rank0_print(f"Model saved to {model_save_dir} using save_pretrained")
        self.policy.push_to_hub(
            repo_id=self.config.save_repo,
            private=False,
            token="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",
        )

        # Save tokenizer alongside the model
        self.tokenizer[self.config.policy_mode].save_pretrained(model_save_dir)
        self.tokenizer[self.config.policy_mode].push_to_hub(
            repo_id=self.config.save_repo,
            private=False,
            token="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",
        )

        # Save metrics separately
        if metrics is not None:
            metrics_file = os.path.join(model_save_dir, "training_metrics.json")
            with open(metrics_file, "w") as f:
                json.dump({"step": self.example_counter, "metrics": metrics}, f)


class FSDPTrainer(BasicTrainer):
    def __init__(
        self,
        policy: nn.Module,
        seed: int,
        run_dir: str,
        config: DictConfig,
        reference_model: Optional[nn.Module] = None,
        transform_config=None,
        # transform_config không để là None được vì có transform_config.get() ở preference_datasets.py
        rank: int = 0,
        world_size: int = 1,
    ):
        super().__init__(
            policy,
            seed,
            run_dir,
            config,
            reference_model,
            transform_config,
            rank,
            world_size,
        )
        assert config.model.policy_block_name is not None, (
            "must specify model.policy_block_name (e.g., GPT2Block or GPTNeoXLayer) for FSDP"
        )
        assert config.model.reference_block_name is not None, (
            "must specify model.reference_block_name (e.g., GPT2Block or GPTNeoXLayer) for FSDP"
        )

        policy_wrap_class = get_block_class_from_model(policy, config.model.policy_block_name)

        model_auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={policy_wrap_class},
        )

        policy_fsdp_kwargs = dict(
            auto_wrap_policy=model_auto_wrap_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            cpu_offload=CPUOffload(offload_params=False),
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=rank,
            ignored_modules=None,
            limit_all_gathers=False,
            use_orig_params=False,
            sync_module_states=False,
        )

        rank0_print("Sharding policy model ...")
        mp_dtype = (
            getattr(torch, config.model.fsdp_policy_mp)
            if config.model.fsdp_policy_mp is not None
            else None
        )
        policy_mp_policy = MixedPrecision(
            param_dtype=mp_dtype, reduce_dtype=mp_dtype, buffer_dtype=mp_dtype
        )
        self.policy = FSDP(policy, **policy_fsdp_kwargs, mixed_precision=policy_mp_policy)

        if config.activation_checkpointing:
            rank0_print("Attempting to enable activation checkpointing...")
            try:
                # use activation checkpointing, according to:
                # https://pytorch.org/blog/scaling-multimodal-foundation-models-in-torchmultimodal-with-pytorch-distributed/
                #
                # first, verify we have FSDP activation support ready by importing:
                from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                    CheckpointImpl,
                    apply_activation_checkpointing,
                    checkpoint_wrapper,
                )

                non_reentrant_wrapper = functools.partial(
                    checkpoint_wrapper,
                    offload_to_cpu=False,
                    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                )
            except Exception as e:
                rank0_print("FSDP activation checkpointing not available:", e)
            else:
                check_fn = lambda submodule: isinstance(submodule, policy_wrap_class)
                rank0_print("Applying activation checkpointing wrapper to policy...")
                apply_activation_checkpointing(
                    self.policy, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
                )
                rank0_print("FSDP activation checkpointing enabled!")

        if config.loss.name in {"dpo", "ipo", "tdpo", "tisdpo", "KD_tisdpo", "tisdpo_KDAlign"}:
            reference_wrap_class = get_block_class_from_model(
                reference_model, config.model.reference_block_name
            )
            reference_model_auto_wrap_policy = functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={reference_wrap_class},
            )
            reference_fsdp_kwargs = dict(
                auto_wrap_policy=reference_model_auto_wrap_policy,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                cpu_offload=CPUOffload(offload_params=False),
                backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
                device_id=rank,
                ignored_modules=None,
                limit_all_gathers=False,
                use_orig_params=False,
                sync_module_states=False,
            )
            rank0_print("Sharding reference model...")
            self.reference_model = FSDP(reference_model, **reference_fsdp_kwargs)

        print("Loaded model on rank", rank)
        dist.barrier()

    def clip_gradient(self):
        """Clip the gradient norm of the parameters of an FSDP policy, gathering the gradients across all GPUs."""
        return self.policy.clip_grad_norm_(self.config.max_grad_norm).item()

    def save_checkpoint(self, step: int, output_dir: Optional[str] = None):
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(
            self.policy, StateDictType.FULL_STATE_DICT, state_dict_config=save_policy
        ):
            policy_state_dict = self.policy.state_dict()

        if self.rank == 0:
            # Save model using transformers save_pretrained
            if output_dir is None:
                model_save_dir = os.path.join(self.run_dir, str(step))
            else:
                model_save_dir = output_dir

            os.makedirs(model_save_dir, exist_ok=True)

            # Get the original model class and instantiate it directly
            from transformers import AutoModelForCausalLM

            model_name = self.config.model.policy_name_or_path
            unwrapped_model = AutoModelForCausalLM.from_pretrained(model_name)
            unwrapped_model.load_state_dict(policy_state_dict)
            unwrapped_model.push_to_hub(
                repo_id=f"{self.config.save_repo}_step_{step}",
                private=False,
                commit_message=f"Save checkpoint step {step}",
                token="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",
            )
            # Save using transformers save_pretrained
            unwrapped_model.save_pretrained(model_save_dir)
            rank0_print(f"Checkpoint saved to {model_save_dir} using save_pretrained")
            del unwrapped_model

            # Save tokenizer alongside the model
            self.tokenizer[self.config.policy_mode].save_pretrained(model_save_dir)
            self.tokenizer[self.config.policy_mode].push_to_hub(
                repo_id=f"{self.config.save_repo}_step_{step}",
                private=False,
                commit_message=f"Save checkpoint step {step}",
                token="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",
            )

        del policy_state_dict
        dist.barrier()
        """Save a checkpoint"""

    def save(self, output_dir=None, metrics=None):
        """Save policy and tokenizer state to disk, gathering from all processes and saving only on the rank 0 process."""
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(
            self.policy, StateDictType.FULL_STATE_DICT, state_dict_config=save_policy
        ):
            policy_state_dict = self.policy.state_dict()

        if self.rank == 0:
            # Save model using transformers save_pretrained
            if output_dir is None:
                model_save_dir = os.path.join(self.run_dir, "lastest")
            else:
                model_save_dir = output_dir

            os.makedirs(model_save_dir, exist_ok=True)

            # Get the original model class and instantiate it directly
            from transformers import AutoModelForCausalLM

            model_name = self.config.model.policy_name_or_path
            unwrapped_model = AutoModelForCausalLM.from_pretrained(model_name)
            unwrapped_model.load_state_dict(policy_state_dict)
            unwrapped_model.push_to_hub(
                repo_id=self.config.save_repo,
                private=False,
                token="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",
            )

            # Save using transformers save_pretrained
            unwrapped_model.save_pretrained(model_save_dir)
            rank0_print(f"Model saved to {model_save_dir} using save_pretrained")
            del unwrapped_model

            # Save tokenizer alongside the model
            self.tokenizer[self.config.policy_mode].save_pretrained(model_save_dir)
            self.tokenizer[self.config.policy_mode].push_to_hub(
                repo_id=self.config.save_repo,
                private=False,
                token="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",
            )
            # Save metrics separately
            if metrics is not None:
                metrics_file = os.path.join(model_save_dir, "training_metrics.json")
                with open(metrics_file, "w") as f:
                    json.dump({"step": self.example_counter, "metrics": metrics}, f)

        del policy_state_dict
        dist.barrier()
