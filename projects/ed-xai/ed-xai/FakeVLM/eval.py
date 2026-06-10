import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from .loader import load_model


def build_prompt(question: str) -> str:
    return f"USER: <image>\n{question}\nASSISTANT:"


def infer_single(model, processor, image_path, prompt, device):
    image = Image.open(image_path).convert("RGB")

    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=256)

    return processor.decode(output[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-local-path", required=True)
    parser.add_argument(
        "--model-hf-path", default="llava-hf/llava-1.5-7b-hf"
    )
    parser.add_argument("--lora-adapter-path", default=None)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--image-folder", default=None)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--question-key", default="human")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    compute_dtype = torch.float32 if device.type == "cpu" else torch.bfloat16

    model, tokenizer, processor, config = load_model(
        model_local_path=args.model_local_path,
        model_hf_path=args.model_hf_path,
        compute_dtype=compute_dtype,
    )

    if args.lora_adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.lora_adapter_path)
        model = model.merge_and_unload()

    model = model.to(device=device, dtype=compute_dtype)
    model.eval()

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

        response = infer_single(model, processor, image_path, prompt, device)

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
