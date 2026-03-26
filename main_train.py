import filelock

filelock.FileLock = filelock.SoftFileLock
import argparse
import os
import sys
import warnings
import math
from smart_init_simple import add_discrete_tokens

# import filelock
# filelock.FileLock = filelock.SoftFileLock
# os.environ["HF_DATASETS_CACHE"] = "/tmp/hf_ds_cache"
# os.environ["HF_HOME"] = "/tmp/hf_home"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["FILELOCK_TIMEOUT"] = "-1"
"""
三阶段 Curriculum Learning 训练脚本 - NTL-WAS 版本 (完整优化版)

优化点：
1. LoRA 配置：修复权重共享导致的参数膨胀 (14% -> ~7%)
2. 梯度裁剪：max_grad_norm=1.0 防止梯度爆炸
3. Gradient Checkpointing：降低显存 30-40%
4. compute_loss 中 mask 复用，避免重复计算
5. dataloader 优化：pin_memory + prefetch
6. 多卡同步修复
"""
"""
torchrun --nproc_per_node=4 main_train.py \
  --problem 19-0 \
  --output_dir /c20250502/lyh/LLM-Code/project-model/boqp/19-0



"""


os.environ["NCCL_TIMEOUT"] = "7200"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType

try:
    from utils import *
except ImportError:
    pass


# ============================================================
#  多卡同步辅助函数
# ============================================================


def sync_embeddings_across_ranks(model, local_rank):
    """在所有 rank 之间同步 embed_tokens 和 lm_head 的权重"""
    if not dist.is_initialized():
        return

    embed_weight = model.get_input_embeddings().weight.data
    dist.broadcast(embed_weight, src=0)

    if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
        lm_weight = model.lm_head.weight.data
        if lm_weight.data_ptr() != embed_weight.data_ptr():
            dist.broadcast(lm_weight, src=0)

    if local_rank == 0:
        print("[Sync] Embeddings synchronized across all ranks")


def barrier_if_distributed():
    """安全地调用 barrier"""
    if dist.is_initialized():
        dist.barrier()


class SafeDataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    """只在 assistant 部分计算 loss"""

    def __init__(
        self, response_template, tokenizer, mlm=False, fallback_strategy="last_portion"
    ):
        super().__init__(tokenizer=tokenizer, mlm=mlm)
        self.response_template = response_template
        self.tokenizer = tokenizer
        self.response_template_ids = tokenizer.encode(
            response_template, add_special_tokens=False
        )
        self.fallback_strategy = fallback_strategy

    def torch_call(self, examples):
        batch = super().torch_call(examples)
        for i in range(len(batch["input_ids"])):
            input_ids = batch["input_ids"][i].tolist()
            template_found = False

            for idx in range(len(input_ids) - len(self.response_template_ids) + 1):
                if (
                    input_ids[idx : idx + len(self.response_template_ids)]
                    == self.response_template_ids
                ):
                    template_found = True
                    response_start_idx = idx + len(self.response_template_ids)
                    batch["labels"][i, :response_start_idx] = -100
                    break

            if not template_found:
                if self.fallback_strategy == "last_portion":
                    seq_length = len(input_ids)
                    start_pos = int(0.9 * seq_length)
                    batch["labels"][i, :start_pos] = -100
                elif self.fallback_strategy == "full_example":
                    pass
                elif self.fallback_strategy == "skip":
                    batch["labels"][i, :] = -100
        return batch


class ThreePhaseCurriculumTrainerNTLWAS(Trainer):
    """
    三阶段 Curriculum Learning Trainer - NTL-WAS 版本 (优化版)
    """

    def __init__(self, *args, tokenizer=None, curriculum_config=None, **kwargs):
        super().__init__(*args, tokenizer=tokenizer, **kwargs)
        self.custom_tokenizer = tokenizer
        self.curriculum_config = curriculum_config or {
            "phase1_ratio": 0.15,
            "phase2_ratio": 0.60,
            "ce_min_weight": 0.2,
            "beta_max": 1.0,
            "gamma_max": 0.5,
            "ntl_lambda": 0.3,
        }
        self._build_token_id_sets()
        self._build_value_mappings()
        self._device_initialized = False
        self._initialized_device = None

    def _build_token_id_sets(self):
        tokenizer = self.custom_tokenizer
        vocab_size = len(tokenizer)

        self.is_int_lookup = torch.zeros(vocab_size, dtype=torch.bool)
        self.is_frac_lookup = torch.zeros(vocab_size, dtype=torch.bool)

        self.int_token_ids = set()
        self.int_id_to_value = {}

        for s in [0, 1]:
            sign = 1 if s == 0 else -1
            for i in range(1000):
                token_str = f"<s{s}i{i:03d}>"
                token_id = tokenizer.convert_tokens_to_ids(token_str)
                if token_id != tokenizer.unk_token_id:
                    self.int_token_ids.add(token_id)
                    self.is_int_lookup[token_id] = True
                    int_part = i // 10
                    frac_first = i % 10
                    coarse_value = sign * (int_part + frac_first / 10.0)
                    self.int_id_to_value[token_id] = (sign, coarse_value)

        self.frac_token_ids = set()
        self.frac_id_to_value = {}

        for i in range(1000):
            token_str = f"<d{i:03d}>"
            token_id = tokenizer.convert_tokens_to_ids(token_str)
            if token_id != tokenizer.unk_token_id:
                self.frac_token_ids.add(token_id)
                self.is_frac_lookup[token_id] = True
                self.frac_id_to_value[token_id] = i / 10000.0

        self.int_ids_list = sorted(list(self.int_token_ids))
        self.frac_ids_list = sorted(list(self.frac_token_ids))
        self.int_id_to_idx = {tid: idx for idx, tid in enumerate(self.int_ids_list)}
        self.frac_id_to_idx = {tid: idx for idx, tid in enumerate(self.frac_ids_list)}
        self.num_token_ids = self.int_token_ids | self.frac_token_ids

        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if local_rank <= 0:
            print(
                f"[NTL-WAS Trainer] INT tokens: {len(self.int_token_ids)}, "
                f"FRAC tokens: {len(self.frac_token_ids)}, "
                f"Vocab size: {vocab_size}"
            )

    def _build_value_mappings(self):
        self.int_coarse_values = torch.zeros(len(self.int_ids_list))
        for idx, tid in enumerate(self.int_ids_list):
            _, coarse_val = self.int_id_to_value[tid]
            self.int_coarse_values[idx] = coarse_val

        self.frac_fine_values = torch.zeros(len(self.frac_ids_list))
        for idx, tid in enumerate(self.frac_ids_list):
            self.frac_fine_values[idx] = self.frac_id_to_value[tid]

        vocab_size = len(self.custom_tokenizer)
        self.coarse_target_lookup = torch.zeros(vocab_size)
        for tid, (_, val) in self.int_id_to_value.items():
            self.coarse_target_lookup[tid] = val

        self.fine_target_lookup = torch.zeros(vocab_size)
        for tid, val in self.frac_id_to_value.items():
            self.fine_target_lookup[tid] = val

    def _ensure_device(self, device):
        if self._device_initialized and self._initialized_device == device:
            return

        self.is_int_lookup = self.is_int_lookup.to(device, non_blocking=True)
        self.is_frac_lookup = self.is_frac_lookup.to(device, non_blocking=True)
        self.int_coarse_values = self.int_coarse_values.to(device, non_blocking=True)
        self.frac_fine_values = self.frac_fine_values.to(device, non_blocking=True)
        self.coarse_target_lookup = self.coarse_target_lookup.to(
            device, non_blocking=True
        )
        self.fine_target_lookup = self.fine_target_lookup.to(device, non_blocking=True)

        self.int_ids_tensor = torch.tensor(
            self.int_ids_list, device=device, dtype=torch.long
        )
        self.frac_ids_tensor = torch.tensor(
            self.frac_ids_list, device=device, dtype=torch.long
        )

        self._device_initialized = True
        self._initialized_device = device

    def _get_phase_and_weights(self, progress):
        cfg = self.curriculum_config
        phase1_end = cfg["phase1_ratio"]
        phase2_end = cfg["phase2_ratio"]
        ce_min = cfg["ce_min_weight"]
        beta_max = cfg["beta_max"]
        gamma_max = cfg["gamma_max"]

        if progress < phase1_end:
            return 1, 1.0, 0.0, 0.0
        elif progress < phase2_end:
            t = (progress - phase1_end) / (phase2_end - phase1_end)
            return 2, 1.0 - t * (1.0 - ce_min), t * beta_max, 0.0
        else:
            t = (
                (progress - phase2_end) / (1.0 - phase2_end)
                if (1.0 - phase2_end) > 0
                else 1.0
            )
            return 3, ce_min, beta_max, t * gamma_max

    def _compute_ntl_was_coarse(self, logits, mask_int, labels, device):
        if not mask_int.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        int_positions = mask_int.nonzero(as_tuple=True)
        if len(int_positions[0]) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        selected_logits = logits[int_positions[0], int_positions[1]]
        int_logits = selected_logits[:, self.int_ids_tensor]
        int_probs = F.softmax(int_logits, dim=-1)

        target_ids = labels[int_positions[0], int_positions[1]]
        target_coarse = self.coarse_target_lookup[target_ids]

        abs_diff = torch.abs(
            target_coarse.unsqueeze(-1) - self.int_coarse_values.unsqueeze(0)
        )
        wasserstein_per_position = (int_probs * abs_diff).sum(dim=-1)

        return wasserstein_per_position.mean()

    def _compute_ntl_was_fine(self, logits, labels, mask_int, mask_frac, device):
        batch_size, seq_len, vocab_size = logits.shape
        if seq_len < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        valid_pair_mask = mask_int[:, :-1] & mask_frac[:, 1:]
        if not valid_pair_mask.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        pair_positions = valid_pair_mask.nonzero(as_tuple=True)
        batch_indices = pair_positions[0]
        frac_indices = pair_positions[1] + 1

        selected_logits = logits[batch_indices, frac_indices]
        frac_logits = selected_logits[:, self.frac_ids_tensor]
        frac_probs = F.softmax(frac_logits, dim=-1)

        target_ids = labels[batch_indices, frac_indices]
        target_fine = self.fine_target_lookup[target_ids]

        abs_diff = torch.abs(
            target_fine.unsqueeze(-1) - self.frac_fine_values.unsqueeze(0)
        )
        wasserstein_per_position = (frac_probs * abs_diff).sum(dim=-1)

        return wasserstein_per_position.mean()

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """
        ★ 优化：mask 只计算一次，阶段1和日志记录复用
        """
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        labels = inputs.get("labels")
        device = input_ids.device

        self._ensure_device(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
        )
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        batch_size, seq_len, vocab_size = shift_logits.shape

        total_steps = self.state.max_steps
        current_step = self.state.global_step
        progress = current_step / max(total_steps, 1)

        phase, ce_weight, beta_weight, gamma_weight = self._get_phase_and_weights(
            progress
        )

        flat_logits = shift_logits.view(-1, vocab_size)
        flat_labels = shift_labels.view(-1)
        loss_ce_all = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100)

        # ★ 优化：mask 提前计算一次，所有阶段和日志复用
        valid_mask = shift_labels != -100
        clamped_labels = shift_labels.clamp(min=0, max=len(self.custom_tokenizer) - 1)
        mask_int = self.is_int_lookup[clamped_labels] & valid_mask
        mask_frac = self.is_frac_lookup[clamped_labels] & valid_mask

        if phase == 1:
            total_loss = ce_weight * loss_ce_all
            loss_ntl_was_coarse = torch.tensor(0.0, device=device)
            loss_ntl_was_fine = torch.tensor(0.0, device=device)
        else:
            loss_ntl_was_coarse = (
                self._compute_ntl_was_coarse(
                    shift_logits, mask_int, shift_labels, device
                )
                if beta_weight > 0
                else torch.tensor(0.0, device=device)
            )
            loss_ntl_was_fine = (
                self._compute_ntl_was_fine(
                    shift_logits, shift_labels, mask_int, mask_frac, device
                )
                if gamma_weight > 0
                else torch.tensor(0.0, device=device)
            )
            total_loss = (
                ce_weight * loss_ce_all
                + beta_weight * loss_ntl_was_coarse
                + gamma_weight * loss_ntl_was_fine
            )

        # ★ 优化：日志直接复用上面的 mask，不再重复计算
        if self.state.global_step % self.args.logging_steps == 0:
            self._last_losses = {
                "phase": phase,
                "progress": progress,
                "loss_ce_all": loss_ce_all.item(),
                "loss_ntl_was_coarse": (
                    loss_ntl_was_coarse.item()
                    if isinstance(loss_ntl_was_coarse, torch.Tensor)
                    else loss_ntl_was_coarse
                ),
                "loss_ntl_was_fine": (
                    loss_ntl_was_fine.item()
                    if isinstance(loss_ntl_was_fine, torch.Tensor)
                    else loss_ntl_was_fine
                ),
                "ce_weight": ce_weight,
                "beta_weight": beta_weight,
                "gamma_weight": gamma_weight,
                "num_int_tokens": mask_int.sum().item(),
                "num_frac_tokens": mask_frac.sum().item(),
                "num_valid_tokens": valid_mask.sum().item(),
            }

        if return_outputs:
            return (total_loss, outputs)
        return total_loss

    def log(self, logs, start_time=None):
        if hasattr(self, "_last_losses"):
            logs.update(self._last_losses)
        if start_time is not None:
            super().log(logs, start_time)
        else:
            super().log(logs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Three-Phase Curriculum Learning Trainer with NTL-WAS"
    )

    parser.add_argument("--max_seq_length", type=int, default=12000)
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"]
    )
    parser.add_argument(
        "--model_name", type=str, default="/c20250502/lyh/LLM-Code/Qwen2.5-7B-Instruct"
    )
    parser.add_argument("--problem", type=str, default="boqp")
    parser.add_argument(
        "--data_format", type=str, default="messages", choices=["messages", "alpaca"]
    )

    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument(
        "--bias", type=str, default="lora_only", choices=["none", "all", "lora_only"]
    )
    parser.add_argument("--use_rslora", action="store_true", default=False)

    # ★ 优化：默认开启 gradient checkpointing
    parser.add_argument("--use_gradient_checkpointing", type=str, default="true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--warmup_steps", type=int, default=20)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--save_step", type=int, default=10000)

    parser.add_argument("--train_lm_head", action="store_true", default=False)
    parser.add_argument("--train_embed_tokens", action="store_true", default=False)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--add_discrete_tokens", action="store_true", default=True)

    parser.add_argument("--phase1_ratio", type=float, default=0.15)
    parser.add_argument("--phase2_ratio", type=float, default=0.50)
    parser.add_argument("--ce_min_weight", type=float, default=0.4)
    parser.add_argument("--beta_max", type=float, default=1.0)
    parser.add_argument("--gamma_max", type=float, default=0.5)
    parser.add_argument("--ntl_lambda", type=float, default=0.3)

    # ★ 优化：梯度裁剪（你的日志显示 grad_norm 经常上万）
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument(
        "--local_rank", type=int, default=int(os.getenv("LOCAL_RANK", -1))
    )

    args = parser.parse_args()
    return args


_DATASET_CACHE = {}


# def get_dataset_messages(tokenizer, problem, max_seq_length, local_rank=0):
#     global _DATASET_CACHE
#     cache_key = (problem, max_seq_length, "messages")

#     if cache_key in _DATASET_CACHE:
#         if local_rank <= 0:
#             print(f"[get_dataset] Reusing cached dataset for {cache_key}")
#         return _DATASET_CACHE[cache_key]

#     EOS_TOKEN = tokenizer.eos_token

#     def formatting_prompts_func(examples):
#         texts = []
#         for messages in examples["messages"]:
#             try:
#                 text = tokenizer.apply_chat_template(
#                     messages, tokenize=False, add_generation_prompt=False
#                 )
#             except Exception:
#                 parts = []
#                 for msg in messages:
#                     role = msg["role"]
#                     content = msg["content"]
#                     if role == "system":
#                         parts.append(f"<|im_start|>system\n{content}<|im_end|>")
#                     elif role == "user":
#                         parts.append(f"<|im_start|>user\n{content}<|im_end|>")
#                     elif role == "assistant":
#                         parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
#                 text = "\n".join(parts)
#             if not text.endswith(EOS_TOKEN):
#                 text = text + EOS_TOKEN
#             texts.append(text)
#         return {"text": texts}

#     if local_rank <= 0:
#         print(f"[get_dataset] Loading dataset...")

#     train_dataset = load_dataset(
#         f'./data/{problem}/train',
#         split="train",
#         # cache_dir="/tmp/hf_datasets_cache"
#         cache_dir="/c20250502/lyh/hf_cache"
#     ).shuffle(seed=42)

#     if local_rank <= 0:
#         print(f"[get_dataset] Dataset loaded. Size: {len(train_dataset)}")
#         if len(train_dataset) > 0:
#             sample = train_dataset[0]
#             print(f"[get_dataset] Sample keys: {sample.keys()}")

#     train_dataset = train_dataset.map(formatting_prompts_func, batched=True)

#     def tokenize_function(examples):
#         return tokenizer(
#             examples["text"],
#             truncation=True,
#             max_length=max_seq_length,
#             padding=False,
#         )

#     train_dataset = train_dataset.map(
#         tokenize_function,
#         batched=True,
#         remove_columns=train_dataset.column_names,
#     )

#     barrier_if_distributed()


#     _DATASET_CACHE[cache_key] = train_dataset
#     return train_dataset
def get_dataset_messages(tokenizer, problem, max_seq_length, local_rank=0):
    global _DATASET_CACHE
    cache_key = (problem, max_seq_length, "messages")

    if cache_key in _DATASET_CACHE:
        if local_rank <= 0:
            print(f"[get_dataset] Reusing cached dataset for {cache_key}")
        return _DATASET_CACHE[cache_key]

    EOS_TOKEN = tokenizer.eos_token

    def formatting_prompts_func(examples):
        texts = []
        for messages in examples["messages"]:
            try:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            except Exception:
                parts = []
                for msg in messages:
                    role = msg["role"]
                    content = msg["content"]
                    if role == "system":
                        parts.append(f"<|im_start|>system\n{content}<|im_end|>")
                    elif role == "user":
                        parts.append(f"<|im_start|>user\n{content}<|im_end|>")
                    elif role == "assistant":
                        parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
                text = "\n".join(parts)
            if not text.endswith(EOS_TOKEN):
                text = text + EOS_TOKEN
            texts.append(text)
        return {"text": texts}

    if local_rank <= 0:
        print(f"[get_dataset] Loading dataset...")

    # ★ 关键：cache_dir 用本地 /tmp，keep_in_memory=True 避免写磁盘
    train_dataset = load_dataset(
        f"{problem}/train",
        split="train",
        cache_dir="{problem}/../hf_ds_cache",
        keep_in_memory=True,
    ).shuffle(seed=42, keep_in_memory=True)

    if local_rank <= 0:
        print(f"[get_dataset] Dataset loaded. Size: {len(train_dataset)}")
        if len(train_dataset) > 0:
            sample = train_dataset[0]
            print(f"[get_dataset] Sample keys: {sample.keys()}")

    train_dataset = train_dataset.map(
        formatting_prompts_func, batched=True, keep_in_memory=True
    )

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )

    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        keep_in_memory=True,
    )

    barrier_if_distributed()

    _DATASET_CACHE[cache_key] = train_dataset
    return train_dataset


def get_dataset_alpaca(tokenizer, problem, max_seq_length, local_rank=0):
    global _DATASET_CACHE
    cache_key = (problem, max_seq_length, "alpaca")

    if cache_key in _DATASET_CACHE:
        return _DATASET_CACHE[cache_key]

    alpaca_prompt = (
        "Below is an instruction describing a combinatorial optimization problem. "
        "It is paired with an input that provides the data of the instance. "
        "Your task is to produce a feasible solution that optimizes (minimizes or maximizes) the given objective.\n\n"
        "### Instruction:{}\n\n"
        "### Input:{}\n\n"
        "### Response:{}"
    )
    EOS_TOKEN = tokenizer.eos_token

    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        for instruction, input_text, output in zip(instructions, inputs, outputs):
            text = alpaca_prompt.format(instruction, input_text, output) + EOS_TOKEN
            texts.append(text)
        return {"text": texts}

    train_dataset = load_dataset(
        f"{problem}/train",
        split="train",
        cache_dir="{problem}/../hf_cache",
    ).shuffle(seed=42)

    train_dataset = train_dataset.map(formatting_prompts_func, batched=True)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )

    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names,
    )

    barrier_if_distributed()

    _DATASET_CACHE[cache_key] = train_dataset
    return train_dataset


def train_model(args):
    local_rank = args.local_rank
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_main_process = (local_rank == 0) or (local_rank == -1)

    if args.output_dir is None:
        dir_out = f"output_{args.problem}_ntl_was_lora_r{args.lora_r}_ep{args.num_train_epochs}"
    else:
        dir_out = args.output_dir

    if args.use_wandb and is_main_process:
        import wandb

        wandb.init(
            project=args.model_name.split("/")[-1] + "_" + args.problem + "_ntl_was",
            name=dir_out,
            config=vars(args),
        )
        report_to = "wandb"
    else:
        report_to = "none"

    use_gc = False
    if isinstance(args.use_gradient_checkpointing, str):
        if args.use_gradient_checkpointing.lower() in ["true", "yes", "1", "unsloth"]:
            use_gc = True
    elif isinstance(args.use_gradient_checkpointing, bool):
        use_gc = args.use_gradient_checkpointing

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    # =========================
    #  Load Model & Tokenizer
    # =========================
    if is_main_process:
        print(f"Loading model: {args.model_name} with {dtype}...")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        cache_dir="/gpfs/work4/0/prjs0685/cache",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map=None,
        trust_remote_code=True,
        cache_dir="/gpfs/work4/0/prjs0685/cache",
    )

    # =========================
    #  添加离散化 Token + 同步嵌入
    # =========================
    if args.add_discrete_tokens and args.data_format == "messages":
        print_rank = 0 if local_rank <= 0 else local_rank
        tokenizer, model = add_discrete_tokens(tokenizer, model, print_rank)

        barrier_if_distributed()
        sync_embeddings_across_ranks(model, local_rank)
        barrier_if_distributed()

    if use_gc:
        model.gradient_checkpointing_enable()

    # =========================
    #  ★ 优化1: LoRA Setup - 修复权重共享参数膨胀
    # =========================
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    if args.train_lm_head:
        target_modules.append("lm_head")
    if args.train_embed_tokens:
        target_modules.append("embed_tokens")

    # ★ 关键优化：检查权重共享，避免重复保存
    is_tied = getattr(model.config, "tie_word_embeddings", False)

    if is_tied:
        # 权重共享：只需保存 embed_tokens，lm_head 通过 tying 自动更新
        modules_to_save = ["embed_tokens"]
        if is_main_process:
            print(
                f"[LoRA] tie_word_embeddings=True -> modules_to_save=['embed_tokens'] only"
            )
            print(
                f"[LoRA] This saves ~550M parameters compared to saving both embed_tokens and lm_head"
            )
    else:
        modules_to_save = ["embed_tokens", "lm_head"]
        if is_main_process:
            print(
                f"[LoRA] tie_word_embeddings=False -> modules_to_save=['embed_tokens', 'lm_head']"
            )

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias=args.bias,
        task_type=TaskType.CAUSAL_LM,
        use_rslora=args.use_rslora,
        modules_to_save=modules_to_save,
    )

    model = get_peft_model(model, peft_config)

    # 验证
    if is_main_process:
        model.print_trainable_parameters()
        test_token = "<s0i100>"
        test_id = tokenizer.convert_tokens_to_ids(test_token)
        if test_id != tokenizer.unk_token_id:
            emb_weight = model.get_input_embeddings().weight.data
            norm = emb_weight[test_id].norm().item()
            print(f"[Verify] After PEFT wrap: {test_token} emb norm = {norm:.4f}")
            if norm < 1e-6:
                print(
                    "[WARNING] Token embedding appears to be zero after PEFT wrapping!"
                )

    # =========================
    #  Dataset & Collator
    # =========================
    if args.data_format == "messages":
        train_dataset = get_dataset_messages(
            tokenizer, args.problem, args.max_seq_length, local_rank=local_rank
        )
        response_template = "<|im_start|>assistant\n"
    else:
        train_dataset = get_dataset_alpaca(
            tokenizer, args.problem, args.max_seq_length, local_rank=local_rank
        )
        response_template = "### Response:"

    collator = SafeDataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
        fallback_strategy="full_example",
    )

    # =========================
    #  Curriculum 配置
    # =========================
    curriculum_config = {
        "phase1_ratio": args.phase1_ratio,
        "phase2_ratio": args.phase2_ratio,
        "ce_min_weight": args.ce_min_weight,
        "beta_max": args.beta_max,
        "gamma_max": args.gamma_max,
        "ntl_lambda": args.ntl_lambda,
    }

    if is_main_process:
        print(f"\n{'='*60}")
        print(f"[Three-Phase Curriculum Learning with NTL-WAS]")
        print(f"{'='*60}")
        print(f"Phase 1 (0% ~ {args.phase1_ratio*100:.0f}%): Pure CE")
        print(
            f"Phase 2 ({args.phase1_ratio*100:.0f}% ~ {args.phase2_ratio*100:.0f}%): CE + NTL-WAS Coarse"
        )
        print(f"  CE: 1.0 -> {args.ce_min_weight}, Beta: 0 -> {args.beta_max}")
        print(
            f"Phase 3 ({args.phase2_ratio*100:.0f}% ~ 100%): CE + NTL-WAS Coarse + Fine"
        )
        print(
            f"  CE: {args.ce_min_weight}, Beta: {args.beta_max}, Gamma: 0 -> {args.gamma_max}"
        )
        print(f"World size: {world_size}, Local rank: {local_rank}")
        print(f"Gradient checkpointing: {use_gc}")
        print(f"Max grad norm: {args.max_grad_norm}")
        print(f"{'='*60}\n")

    # =========================
    #  ★ 优化2: TrainingArguments
    # =========================
    training_args = TrainingArguments(
        output_dir=dir_out,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_step,
        save_total_limit=args.save_total_limit,
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        bf16_full_eval=True,
        optim=args.optim,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        seed=args.seed,
        report_to=report_to,
        gradient_checkpointing=use_gc,
        gradient_checkpointing_kwargs={"use_reentrant": False} if use_gc else None,
        # ★ 优化：梯度裁剪（你的 grad_norm 经常上万，必须裁剪）
        max_grad_norm=args.max_grad_norm,
        # DDP 参数
        ddp_find_unused_parameters=False,
        local_rank=local_rank,
        group_by_length=False,
        save_on_each_node=False,
        ddp_broadcast_buffers=False,
        # ★ 优化：dataloader 加速
        # dataloader_num_workers=4 if local_rank >= 0 else 0,
        # dataloader_pin_memory=True,
        dataloader_num_workers=0 if local_rank >= 0 else 0,
        dataloader_pin_memory=False,
        resume_from_checkpoint=False,
    )

    trainer = ThreePhaseCurriculumTrainerNTLWAS(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
        curriculum_config=curriculum_config,
    )

    # =========================
    #  Start Training
    # =========================
    if is_main_process:
        print("\nStarting training...")
        print(f"  Data format: {args.data_format}")
        print(
            f"  Discrete tokens: {args.add_discrete_tokens and args.data_format == 'messages'}"
        )
        print(f"  Weight tying: {is_tied}")

    trainer.train()

    if is_main_process:
        trainer.save_model(dir_out)
        tokenizer.save_pretrained(dir_out)
        print(f"Model and tokenizer saved to {dir_out}")

    return trainer


if __name__ == "__main__":
    args = parse_args()
    train_model(args)
