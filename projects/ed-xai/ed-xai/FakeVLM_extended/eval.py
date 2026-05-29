import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from .extractors import get_extractor
from .frequency_projector import FrequencyProjector
from .loader import load_model
from .model import extend_model


def build_prompt(question: str) -> str:
    return f"USER: <image>\n{question}\nASSISTANT:"


def infer_single(
    model, processor, image_path, prompt, device,
    freq_extractor=None, num_freq_tokens=0, image_token_id=None,
):
    image = Image.open(image_path).convert("RGB")

    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    generate_kwargs = {}

    if freq_extractor is not None:
        input_ids = inputs["input_ids"]
        mask = input_ids[0] == image_token_id
        last_img_pos = mask.nonzero()[-1].item()
        extra = torch.full((1, num_freq_tokens), image_token_id, dtype=input_ids.dtype, device=device)
        inputs["input_ids"] = torch.cat(
            [input_ids[:, :last_img_pos + 1], extra, input_ids[:, last_img_pos + 1:]], dim=1
        )
        if "attention_mask" in inputs:
            attn = inputs["attention_mask"]
            inputs["attention_mask"] = torch.cat(
                [attn[:, :last_img_pos + 1], torch.ones_like(extra), attn[:, last_img_pos + 1:]], dim=1
            )
        generate_kwargs["freq_pixel_values"] = freq_extractor.preprocess([image]).to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            **generate_kwargs,
            max_new_tokens=256,
        )

    return processor.decode(output[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-local-path", required=True)
    parser.add_argument(
        "--model-hf-path", default="llava-hf/llava-1.5-7b-hf"
    )
    parser.add_argument("--freq-projector-checkpoint", default=None)
    parser.add_argument("--lora-adapter-path", default=None)
    parser.add_argument("--freq-extractor-name", default="fft")
    parser.add_argument("--freq-input-size", type=int, default=224)
    parser.add_argument("--freq-pool-size", type=int, default=32)
    parser.add_argument("--num-freq-tokens", type=int, default=1)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--image-folder", default=None)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--question-key", default="human")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    compute_dtype = torch.float32 if device.type == "cpu" else torch.bfloat16

    # Load base model
    model, tokenizer, processor, config = load_model(
        model_local_path=args.model_local_path,
        model_hf_path=args.model_hf_path,
        compute_dtype=compute_dtype,
    )

    freq_extractor = None

    if args.freq_projector_checkpoint is not None:
        freq_extractor = get_extractor(
            args.freq_extractor_name,
            input_size=args.freq_input_size,
            pool_size=args.freq_pool_size,
        )
        freq_projector = FrequencyProjector(
            input_dim=freq_extractor.output_dim,
            output_dim=config.text_config.hidden_size,
            num_tokens=args.num_freq_tokens,
        )

        state = torch.load(args.freq_projector_checkpoint, map_location="cpu")
        freq_projector.load_state_dict(state)

        model = extend_model(
            model, freq_extractor, freq_projector, args.num_freq_tokens
        )

        if args.lora_adapter_path is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, args.lora_adapter_path)
            model = model.merge_and_unload()

    model = model.to(device=device, dtype=compute_dtype)
    model.eval()

    # Load evaluation data
    with open(args.data_path) as f:
        data = json.load(f)
    if args.max_samples is not None:
        data = data[: args.max_samples]

    results = []
    for entry in tqdm(data, desc="Evaluating"):
        image_path = entry["image"]
        if args.image_folder:
            image_path = str(Path(args.image_folder) / image_path)

        question = entry["conversations"][0]["value"]
        question = question.replace("<image>", "").strip()
        prompt = build_prompt(question)

        response = infer_single(
            model, processor, image_path, prompt, device,
            freq_extractor=freq_extractor,
            num_freq_tokens=args.num_freq_tokens,
            image_token_id=config.image_token_index,
        )

        results.append(
            {
                "image": entry["image"],
                "ground_truth_label": entry.get("label"),
                "question": question,
                "response": response,
            }
        )

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved {len(results)} results to {args.output_path}")


if __name__ == "__main__":
    main()
