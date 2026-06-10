# FakeVLM -- Ablation Fine-Tuning

LoRA fine-tuning of the original [FakeVLM](https://huggingface.co/lingcco/fakeVLM) model **without** the frequency-domain feature branch, to isolate whether improvements come from the spectral features or from LoRA fine-tuning alone.

## Motivation

FakeVLM-Extended adds a parallel FFT branch that concatenates a frequency token with CLIP visual tokens before Vicuna 7B. Both the FFT magnitude and FFT phase variants improve over the baseline, but produce similar results to each other. This ablation tests the null hypothesis: the gains come entirely from LoRA fine-tuning Vicuna, and the frequency features do not contribute.

## Ablation Conditions

| Condition | What is trained | Flag |
|---|---|---|
| **LoRA on Vicuna only** | LoRA adapters on Vicuna's linear layers (~100M params) | `--train_projector False` (default) |
| **LoRA on Vicuna + CLIP projector** | LoRA adapters + CLIP→Vicuna projection MLP | `--train_projector True` |

Hyperparameters match FakeVLM-Extended Stage 2 exactly (3 epochs, LR 2e-5, LoRA r=8/α=16, batch 4, grad accum 4) for a controlled comparison.

## Setup

```bash
cd dgm-2026.1/projects/ed-xai/ed-xai
pip install -r FakeVLM/requirements.txt
```

All commands below assume you are in `dgm-2026.1/projects/ed-xai/ed-xai/`.

## Training

### Condition 1 -- LoRA on Vicuna only

```bash
torchrun --nproc_per_node=1 -m FakeVLM.train \
    --model_hf_path llava-hf/llava-1.5-7b-hf \
    --model_local_path ../models/FakeVLM \
    --data_path ../data/external/FakeClue/data_json/train.json \
    --image_folder ../data/external/FakeClue/train \
    --output_dir ../models/FakeVLM/lora_vicuna \
    --bf16 True \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing True \
    --learning_rate 2e-5 \
    --warmup_steps 558 \
    --weight_decay 0.0 \
    --lr_scheduler_type cosine \
    --eval_split_ratio 0.05 \
    --eval_strategy steps \
    --eval_steps 1000 \
    --logging_steps 50 \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 3 \
    --report_to none \
    --use_lora True \
    --train_projector False \
    --dataloader_num_workers 4 \
    --deepspeed FakeVLM/ds_configs/zero2.json
```

### Condition 2 -- LoRA on Vicuna + CLIP projector

Same as above with `--train_projector True` and a different output directory:

```bash
torchrun --nproc_per_node=1 -m FakeVLM.train \
    --model_hf_path llava-hf/llava-1.5-7b-hf \
    --model_local_path ../models/FakeVLM \
    --data_path ../data/external/FakeClue/data_json/train.json \
    --image_folder ../data/external/FakeClue/train \
    --output_dir ../models/FakeVLM/lora_vicuna_projector \
    --bf16 True \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing True \
    --learning_rate 2e-5 \
    --warmup_steps 558 \
    --weight_decay 0.0 \
    --lr_scheduler_type cosine \
    --eval_split_ratio 0.05 \
    --eval_strategy steps \
    --eval_steps 1000 \
    --logging_steps 50 \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 3 \
    --report_to none \
    --use_lora True \
    --train_projector True \
    --dataloader_num_workers 4 \
    --deepspeed FakeVLM/ds_configs/zero2.json
```

### Quick CPU test

```bash
python -m FakeVLM.train \
    --model_hf_path llava-hf/llava-1.5-7b-hf \
    --model_local_path ../models/FakeVLM \
    --data_path ../data/external/FakeClue/data_json/train.json \
    --image_folder ../data/external/FakeClue/train \
    --output_dir FakeVLM/output/test \
    --per_device_train_batch_size 1 \
    --learning_rate 2e-5 \
    --use_lora True \
    --use_cpu True \
    --max_steps 2
```

## Evaluation

```bash
python -m FakeVLM.eval \
    --model-local-path ../models/FakeVLM \
    --lora-adapter-path ../models/FakeVLM/lora_vicuna \
    --data-path ../data/external/FakeClue/data_json/test.json \
    --image-folder ../data/external/FakeClue/test \
    --output-path ../reports/lora_vicuna/eval_results.json
```

For condition 2, replace `lora_vicuna` with `lora_vicuna_projector`.

### Quick CPU test

```bash
python -m FakeVLM.eval \
    --model-local-path ../models/FakeVLM \
    --lora-adapter-path FakeVLM/output/test \
    --data-path ../data/external/FakeClue/data_json/test.json \
    --image-folder ../data/external/FakeClue/test \
    --output-path FakeVLM/output/test/results.json \
    --device cpu \
    --max-samples 2
```

## Expected comparison

| Model | Training | Frequency features |
|---|---|---|
| Baseline | None | No |
| FakeVLM + LoRA (Vicuna) | LoRA on Vicuna | No |
| FakeVLM + LoRA (Vicuna + projector) | LoRA on Vicuna + CLIP projector | No |
| FakeVLM-Extended (FFT magnitude) | LoRA on Vicuna + freq projector | FFT magnitude |
| FakeVLM-Extended (FFT phase) | LoRA on Vicuna + freq projector | FFT phase |

If the LoRA-only models match or exceed the FFT-extended models, the frequency branch is not contributing to the improvement.

## File overview

```
FakeVLM/
|-- train.py             # Training entrypoint
|-- eval.py              # Evaluation entrypoint
|-- collator.py          # LLaVA data collator (no freq features)
|-- dataset.py           # LazySupervisedDataset (from FakeVLM_extended)
|-- loader.py            # Model loading (from FakeVLM_extended)
|-- arguments.py         # CLI argument dataclasses
|-- utils.py             # Trainer, helpers (from FakeVLM_extended)
+-- ds_configs/
    +-- zero2.json       # DeepSpeed ZeRO-2 configuration
```
