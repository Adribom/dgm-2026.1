import types

import torch
from torch import nn, Tensor
from transformers import LlavaForConditionalGeneration

from .extractors.base import BaseFrequencyExtractor
from .frequency_projector import FrequencyProjector


class ExtendedProjector(nn.Module):
    """Wraps the original LLaVA multi-modal projector to append frequency tokens.

    During forward, runs the original CLIP projector and concatenates
    frequency-domain tokens produced by the freq_extractor + freq_projector
    pipeline.  The freq_pixel_values tensor is set externally via
    _freq_pixel_values before each forward call (managed by extend_model).
    """

    def __init__(
        self,
        original_projector: nn.Module,
        freq_extractor: BaseFrequencyExtractor,
        freq_projector: FrequencyProjector,
    ):
        super().__init__()
        self.original_projector = original_projector
        self.freq_extractor = freq_extractor
        self.freq_projector = freq_projector
        self._freq_pixel_values = None

    def forward(self, image_features: Tensor) -> Tensor:
        clip_tokens = self.original_projector(image_features)

        if self._freq_pixel_values is not None:
            freq_pv = self._freq_pixel_values
            self._freq_pixel_values = None
            with torch.no_grad():
                freq_feat = self.freq_extractor(
                    freq_pv.to(device=clip_tokens.device)
                )
            freq_tokens = self.freq_projector(
                freq_feat.to(dtype=clip_tokens.dtype)
            )
            return torch.cat([clip_tokens, freq_tokens], dim=1)

        # Fallback when freq_pixel_values is not provided: zero tokens
        # to keep the 577-token shape expected by masked_scatter.
        batch_size = clip_tokens.shape[0]
        zero_tokens = torch.zeros(
            batch_size,
            self.freq_projector.num_tokens,
            clip_tokens.shape[2],
            device=clip_tokens.device,
            dtype=clip_tokens.dtype,
        )
        return torch.cat([clip_tokens, zero_tokens], dim=1)


def extend_model(
    model: LlavaForConditionalGeneration,
    freq_extractor: BaseFrequencyExtractor,
    freq_projector: FrequencyProjector,
    num_freq_tokens: int = 1,
) -> LlavaForConditionalGeneration:
    """Extend a LlavaForConditionalGeneration model with frequency features.

    Replaces multi_modal_projector with ExtendedProjector and patches
    forward / prepare_inputs_for_generation to handle freq_pixel_values.
    """
    # 1. Wrap the projector (lives on model.model in transformers 5.x)
    extended_proj = ExtendedProjector(
        model.model.multi_modal_projector, freq_extractor, freq_projector
    )
    model.model.multi_modal_projector = extended_proj

    # 2. Update config so HF uses the non-legacy masked_scatter path
    #    (image_seq_length=577 makes the check "577 < 577" = False)
    model.config.image_seq_length = 576 + num_freq_tokens

    # 3. Freeze the extractor
    freq_extractor.requires_grad_(False)

    # 4. Patch forward to handle freq_pixel_values kwarg
    # logits_to_keep must be explicit so _supports_logits_to_keep() finds it
    # via inspect.signature; otherwise generate() passes None and the
    # hidden_states[:, None, :] indexing produces 4D logits.
    def _extended_forward(self, freq_pixel_values=None, logits_to_keep=0, **kwargs):
        if freq_pixel_values is not None:
            self.model.multi_modal_projector._freq_pixel_values = freq_pixel_values
        return LlavaForConditionalGeneration.forward(
            self, logits_to_keep=logits_to_keep, **kwargs
        )

    model.forward = types.MethodType(_extended_forward, model)

    # 5. Patch prepare_inputs_for_generation for generate() support
    _orig_prepare = LlavaForConditionalGeneration.prepare_inputs_for_generation

    def _extended_prepare(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        attention_mask=None,
        logits_to_keep=None,
        is_first_iteration=False,
        freq_pixel_values=None,
        **kwargs,
    ):
        model_inputs = _orig_prepare(
            self,
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            logits_to_keep=logits_to_keep,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )
        if model_inputs.get("pixel_values") is not None:
            model_inputs["freq_pixel_values"] = freq_pixel_values
        return model_inputs

    model.prepare_inputs_for_generation = types.MethodType(
        _extended_prepare, model
    )

    return model
