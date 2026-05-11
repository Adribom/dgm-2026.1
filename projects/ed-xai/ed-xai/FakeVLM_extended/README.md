# FakeVLM-Extended

Extends [FakeVLM](https://github.com/) (LLaVA 1.5) with a modular frequency-domain feature branch for improved deepfake detection on FakeClue.

## Architecture

FakeVLM uses CLIP-ViT to produce 576 visual tokens that are projected into Vicuna 7B's embedding space. This extension adds a parallel frequency branch that extracts spectral features from the input image, maps them through a trainable MLP, and concatenates the result as additional token(s) alongside the CLIP tokens.

```
Image --> CLIP-ViT --> 576 x 1024 --> CLIP Projector --+--> 577 x 4096 --> Vicuna 7B
                                                       |
Image --> FreqExtractor --> 3072 --> FreqProjector -----+
                                      1 x 4096
```

The `ExtendedProjector` replaces the original `multi_modal_projector` and runs both branches transparently. The HuggingFace `LlavaForConditionalGeneration` forward pass is unmodified -- it sees 577 visual features instead of 576 and merges them into the text sequence via `masked_scatter`.

## Setup

```bash
cd dgm-2026.1/projects/ed-xai/ed-xai
pip install -r FakeVLM_extended/requirements.txt
```

All commands below assume you are in `dgm-2026.1/projects/ed-xai/ed-xai/` (the parent of `FakeVLM_extended/`), since `python -m` resolves modules relative to the working directory.

## Training

Training follows a two-stage approach mirroring LLaVA's original recipe.

### Stage 1 -- Train frequency projector only

Freezes CLIP, the CLIP projector, and Vicuna. Only the frequency projector MLP (~22M params) is trained. This teaches the MLP to produce a meaningful token from the frequency features.

```bash
torchrun --nproc_per_node=1 -m FakeVLM_extended.train \
    --model_hf_path llava-hf/llava-1.5-7b-hf \
    --model_local_path ../models/FakeVLM \
    --data_path ../data/external/FakeClue/data_json/train_frequency.json \
    --image_folder ../data/external/FakeClue/train \
    --output_dir FakeVLM_extended/output/stage1 \
    --bf16 True \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --learning_rate 1e-3 \
    --freq_extractor_name fft \
    --freq_pool_size 32 \
    --training_stage 1 \
    --use_lora False \
    --deepspeed FakeVLM_extended/ds_configs/zero2.json
```

Output: `FakeVLM_extended/output/stage1/freq_projector.pt`

### Stage 2 -- Fine-tune Vicuna with LoRA + frequency projector

Loads the Stage 1 projector checkpoint, applies LoRA (r=8, alpha=16) to Vicuna's linear layers, and continues training both the LoRA adapters and the frequency projector.

```bash
torchrun --nproc_per_node=1 -m FakeVLM_extended.train \
    --model_hf_path llava-hf/llava-1.5-7b-hf \
    --model_local_path ../models/FakeVLM \
    --data_path ../data/external/FakeClue/data_json/train_frequency.json \
    --image_folder ../data/external/FakeClue/train \
    --output_dir FakeVLM_extended/output/stage2 \
    --bf16 True \
    --num_train_epochs 5 \
    --per_device_train_batch_size 4 \
    --learning_rate 2e-5 \
    --freq_extractor_name fft \
    --freq_pool_size 32 \
    --training_stage 2 \
    --freq_projector_checkpoint FakeVLM_extended/output/stage1/freq_projector.pt \
    --use_lora True \
    --lora_r 8 \
    --lora_alpha 16 \
    --deepspeed FakeVLM_extended/ds_configs/zero2.json
```

Output: LoRA adapter in `FakeVLM_extended/output/stage2/` + `freq_projector.pt`

### Quick CPU test

To verify the pipeline runs end-to-end without a GPU (slow, ~14GB RAM to load the 7B model):

```bash
python -m FakeVLM_extended.train \
    --model_hf_path llava-hf/llava-1.5-7b-hf \
    --model_local_path ../models/FakeVLM \
    --data_path ../data/external/FakeClue/data_json/train_frequency.json \
    --image_folder ../data/external/FakeClue/train \
    --output_dir FakeVLM_extended/output/stage1_test \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-3 \
    --freq_extractor_name fft \
    --freq_pool_size 32 \
    --training_stage 1 \
    --use_lora False \
    --use_cpu True \
    --max_steps 2
```

Stage 2 (requires a Stage 1 checkpoint):

```bash
python -m FakeVLM_extended.train \
    --model_hf_path llava-hf/llava-1.5-7b-hf \
    --model_local_path ../models/FakeVLM \
    --data_path ../data/external/FakeClue/data_json/train_frequency.json \
    --image_folder ../data/external/FakeClue/train \
    --output_dir FakeVLM_extended/output/stage2_test \
    --per_device_train_batch_size 1 \
    --learning_rate 2e-5 \
    --freq_extractor_name fft \
    --freq_pool_size 32 \
    --training_stage 2 \
    --freq_projector_checkpoint FakeVLM_extended/output/stage1_test/freq_projector.pt \
    --use_lora True \
    --lora_r 8 \
    --lora_alpha 16 \
    --use_cpu True \
    --max_steps 2
```

## Evaluation

```bash
python -m FakeVLM_extended.eval \
    --model-local-path ../models/FakeVLM \
    --freq-projector-checkpoint FakeVLM_extended/output/stage2/freq_projector.pt \
    --lora-adapter-path FakeVLM_extended/output/stage2 \
    --data-path ../data/external/FakeClue/data_json/test_frequency.json \
    --image-folder ../data/external/FakeClue/test \
    --output-path FakeVLM_extended/results.json
```

### Quick CPU test

```bash
python -m FakeVLM_extended.eval \
    --model-local-path ../models/FakeVLM \
    --freq-projector-checkpoint FakeVLM_extended/output/stage2_test/freq_projector.pt \
    --lora-adapter-path FakeVLM_extended/output/stage2_test \
    --data-path ../data/external/FakeClue/data_json/test_frequency.json \
    --image-folder ../data/external/FakeClue/test \
    --output-path FakeVLM_extended/output/stage2_test/results.json \
    --device cpu \
    --max-samples 2
```

## Creating a new extractor

The extractor registry makes it straightforward to experiment with different frequency representations. Each extractor is a self-contained `nn.Module` with no trainable parameters (frozen by default).

### 1. Create the extractor class

Add a new file under `extractors/`. Subclass `BaseFrequencyExtractor` and implement three members:

```python
# extractors/dct.py
from typing import List

import torch
from torch import Tensor
from PIL import Image

from .base import BaseFrequencyExtractor


class DCTExtractor(BaseFrequencyExtractor):
    def __init__(self, input_size: int = 224, num_coeffs: int = 1024):
        super().__init__()  # freezes all parameters
        self._input_size = input_size
        self._num_coeffs = num_coeffs

    @property
    def output_dim(self) -> int:
        """Total feature dimensionality after forward()."""
        return self._num_coeffs

    def preprocess(self, images: List[Image.Image]) -> Tensor:
        """Convert PIL images to a batched tensor on CPU.

        Called by the collator during data loading. Must return
        [B, C, H, W] float32.
        """
        # resize, convert to tensor, etc.
        ...

    def forward(self, x: Tensor) -> Tensor:
        """Extract features on the model device.

        Input:  [B, C, H, W]
        Output: [B, output_dim]

        Always called under torch.no_grad().
        """
        # apply DCT, pool, flatten, etc.
        ...
```

The contract:
- `output_dim` tells the `FrequencyProjector` MLP what input dimension to expect.
- `preprocess()` runs on CPU during data collation (before batches hit the GPU).
- `forward()` runs on GPU. The framework wraps it in `torch.no_grad()`.
- The constructor must call `super().__init__()`, which sets `requires_grad_(False)`.

### 2. Register it

Add an entry in `extractors/__init__.py`:

```python
from .dct import DCTExtractor

_EXTRACTORS = {
    "fft": FFTExtractor,
    "dct": DCTExtractor,   # <-- add this
}
```

### 3. Use it

Pass `--freq_extractor_name dct` (and any extractor-specific args) to `train.py` or `eval.py`. The `FrequencyProjector` MLP automatically adapts to the new `output_dim`.

### Adding extractor-specific CLI arguments

If your extractor needs parameters beyond `input_size` and `pool_size`, add them to `FreqArguments` in `arguments.py` and pass them through in `train.py` where `get_extractor()` is called. The `get_extractor()` factory forwards all `**kwargs` to the constructor.

## File overview

```
FakeVLM_extended/
|-- extractors/
|   |-- __init__.py             # Registry: get_extractor(name, **kwargs)
|   |-- base.py                 # BaseFrequencyExtractor ABC
|   +-- fft.py                  # FFTExtractor (log-magnitude FFT spectrum)
|-- model.py                    # ExtendedProjector + extend_model()
|-- frequency_projector.py      # FrequencyProjector MLP (trainable)
|-- collator.py                 # Extended collator (577-token expansion)
|-- dataset.py                  # LazySupervisedDataset (adapted from FakeVLM)
|-- loader.py                   # Model loading (adapted from FakeVLM)
|-- arguments.py                # CLI argument dataclasses
|-- utils.py                    # Trainer, helpers (adapted from FakeVLM)
|-- train.py                    # Training entrypoint (Stage 1 / Stage 2)
|-- eval.py                     # Evaluation entrypoint
+-- requirements.txt
```

## Key design decisions

- **Projector wrapping, not model subclassing.** The `ExtendedProjector` replaces `model.multi_modal_projector` and the HF forward pass is monkey-patched minimally. This preserves compatibility with HF Trainer, DeepSpeed ZeRO-2/3, and PEFT/LoRA without subclassing `LlavaForConditionalGeneration`.
- **Non-legacy masked_scatter path.** Setting `config.image_seq_length = 577` triggers the HF non-legacy merge path, which uses `masked_scatter` to place 577 visual features into the text embedding sequence.
- **Zero-token fallback.** When `freq_pixel_values` is not provided (e.g., using the model on data without frequency preprocessing), the `ExtendedProjector` substitutes a zero token to maintain the expected 577-token shape.
- **Frozen extractors, trainable projector.** The frequency extractor never trains (its job is feature extraction). The projector MLP learns to map those features into the LLM's embedding space. This mirrors the CLIP projector design in LLaVA.
