# Adapted from FakeVLM_extended/utils.py — removed plot_fft_spectra.

import math
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
from torch.utils.data import Sampler
import transformers
from transformers import Trainer
from transformers.trainer import has_length


def rank0_print(*args):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(*args)
    else:
        print(*args)


def find_all_linear_names(
    named_modules: Dict, target_modules: List[str]
) -> List[str]:
    lora_module_names = set()
    for name, module in named_modules.items():
        if not any(mod in name for mod in target_modules):
            continue
        if isinstance(module, torch.nn.Linear):
            lora_module_names.add(name)
    lora_module_names.discard("lm_head")
    return list(lora_module_names)


def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        from deepspeed import zero

        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def safe_save_model_for_hf_trainer(
    trainer: transformers.Trainer, output_dir: str
):
    if getattr(trainer, "deepspeed", None):
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {k: v.cpu() for k, v in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


class NoTextOnlyBatchSampler(Sampler):
    """Ensures no batch is text-only (required for DeepSpeed)."""

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        is_text_only: Optional[List[bool]] = None,
        generator=None,
    ):
        if is_text_only is None:
            raise ValueError("`is_text_only` must be provided.")
        self.batch_size = batch_size
        self.world_size = world_size
        self.is_text_only = is_text_only
        self.generator = generator
        self.mega_batch_size = batch_size * world_size

    def __len__(self):
        return len(self.is_text_only)

    def __iter__(self):
        mm_indices = [
            i for i, text_only in enumerate(self.is_text_only) if not text_only
        ]
        uni_indices = [
            i for i, text_only in enumerate(self.is_text_only) if text_only
        ]
        num_batches = math.ceil(
            (len(mm_indices) + len(uni_indices)) / self.mega_batch_size
        )
        if len(mm_indices) < num_batches:
            raise ValueError(
                f"{len(mm_indices)} multimodal entries, {num_batches} batches. "
                "Not enough multimodal data."
            )

        mm_indices = [
            mm_indices[i]
            for i in torch.randperm(len(mm_indices), generator=None).tolist()
        ]
        uni_indices = [
            uni_indices[i]
            for i in torch.randperm(len(uni_indices), generator=None).tolist()
        ]

        num_uni_per_batch = [len(uni_indices) // num_batches] * num_batches
        for i in range(len(uni_indices) % num_batches):
            num_uni_per_batch[i] += 1

        mega_batches = []
        cur_uni = 0
        cur_mm = 0
        for i, n_uni in enumerate(num_uni_per_batch):
            batch = list(uni_indices[cur_uni : cur_uni + n_uni])
            cur_uni += n_uni
            if i < num_batches - 1:
                inc = self.mega_batch_size - len(batch)
                batch.extend(mm_indices[cur_mm : cur_mm + inc])
                cur_mm += inc
            else:
                batch.extend(mm_indices[cur_mm:])
            mega_batches.append(batch)

        order = torch.randperm(len(mega_batches), generator=self.generator)
        mega_batches = [mega_batches[i] for i in order]
        return iter([i for batch in mega_batches for i in batch])


class TrainerWithCustomSampler(Trainer):
    def _get_train_sampler(self, dataset=None) -> Optional[Sampler]:
        dataset = dataset or self.train_dataset
        if dataset is None or not has_length(dataset):
            return None
        return NoTextOnlyBatchSampler(
            self.args.train_batch_size,
            world_size=self.args.world_size
            * self.args.gradient_accumulation_steps,
            is_text_only=dataset.is_text_only,
        )

    def _get_eval_sampler(self, eval_dataset) -> Optional[Sampler]:
        return NoTextOnlyBatchSampler(
            self.args.eval_batch_size,
            world_size=self.args.world_size,
            is_text_only=eval_dataset.is_text_only,
        )
