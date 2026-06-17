"""
Tests for binding.sd_generator.

We mock the heavy diffusers pipeline so tests run in <1s without
downloading models or requiring a GPU. The point is to catch
configuration / wiring bugs, not to validate diffusion itself.

What's actually being tested:
  - `params_from_config` correctly extracts hyperparameters from the YAML.
  - `generate_one` calls the underlying pipeline with the right arguments
    (a refactor that silently drops `guidance_scale`, for example, fails here).
  - Empty negative prompt is normalized to None for the pipeline.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from binding.sd_generator import GenerationParams, SDGenerator, params_from_config


def test_params_from_config_extracts_all_fields():
    cfg = {
        "width": 512,
        "height": 512,
        "num_inference_steps": 30,
        "guidance_scale": 7.5,
        "negative_prompt": "",
    }
    p = params_from_config(cfg)
    assert p.width == 512
    assert p.height == 512
    assert p.num_inference_steps == 30
    assert p.guidance_scale == 7.5
    assert p.negative_prompt == ""


def test_params_from_config_handles_missing_negative_prompt():
    """Negative prompt is optional and should default to empty string."""
    cfg = {
        "width": 512, "height": 512,
        "num_inference_steps": 30, "guidance_scale": 7.5,
    }
    p = params_from_config(cfg)
    assert p.negative_prompt == ""


def test_params_frozen():
    """GenerationParams is frozen — protects against accidental mutation mid-run."""
    p = GenerationParams(width=512, height=512, num_inference_steps=30, guidance_scale=7.5)
    with pytest.raises((AttributeError, Exception)):
        p.width = 1024  # type: ignore[misc]


@patch("binding.sd_generator.SDGenerator.__init__", return_value=None)
def test_generate_one_passes_correct_kwargs(mock_init):
    """Verify that generate_one forwards all hyperparameters to the diffusers pipeline."""
    gen = SDGenerator(model_id="dummy")  # __init__ is mocked
    gen.pipe = MagicMock()
    gen.device = "cpu"
    gen.pipe.return_value.images = [MagicMock(name="fake_image")]

    params = GenerationParams(
        width=256, height=256, num_inference_steps=20,
        guidance_scale=5.0, negative_prompt="blurry",
    )
    result = gen.generate_one(prompt="a red apple", seed=7, params=params)

    # Pipeline was called once.
    assert gen.pipe.call_count == 1
    kwargs = gen.pipe.call_args.kwargs

    # Every hyperparameter from `params` reached the pipeline.
    assert kwargs["prompt"] == "a red apple"
    assert kwargs["width"] == 256
    assert kwargs["height"] == 256
    assert kwargs["num_inference_steps"] == 20
    assert kwargs["guidance_scale"] == 5.0
    assert kwargs["negative_prompt"] == "blurry"

    # Generator was provided (so seed handling actually happens).
    assert kwargs["generator"] is not None

    # generate_one returns the first image from the pipeline output.
    assert result is gen.pipe.return_value.images[0]


@patch("binding.sd_generator.SDGenerator.__init__", return_value=None)
def test_generate_one_empty_negative_prompt_becomes_none(mock_init):
    """
    Empty-string negative_prompt should become None for the pipeline.
    diffusers treats "" and None differently in some versions; we
    normalize to None to stay on the documented path.
    """
    gen = SDGenerator(model_id="dummy")
    gen.pipe = MagicMock()
    gen.device = "cpu"
    gen.pipe.return_value.images = [MagicMock()]

    params = GenerationParams(
        width=512, height=512, num_inference_steps=30,
        guidance_scale=7.5, negative_prompt="",
    )
    gen.generate_one(prompt="x", seed=0, params=params)

    assert gen.pipe.call_args.kwargs["negative_prompt"] is None
