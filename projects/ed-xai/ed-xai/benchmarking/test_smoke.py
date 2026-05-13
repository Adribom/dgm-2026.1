"""
Smoke test for the Benchmarker class using dummy models and datasets.
Run from the ed-xai/ed-xai/ directory:

    python -m benchmarking.test_smoke
"""
import random
from typing import List, Tuple

from PIL import Image

from benchmarking import Benchmarker
from benchmarking.metrics import accuracy, auc, avg_precision, css, f1, rouge_l


# ---------------------------------------------------------------------------
# Dummy dataset factories
# ---------------------------------------------------------------------------

_FAKE_RESPONSES = [
    "This image is clearly fake. It shows deepfake artifacts and manipulated facial features.",
    "The image appears synthetic and artificially generated.",
    "This looks like a forged image with altered regions.",
    "Deepfake artifacts are visible in this manipulated image.",
]
_REAL_RESPONSES = [
    "This image looks real and authentic with no signs of manipulation.",
    "The image appears genuine and original.",
    "This is an authentic photograph with no detected artifacts.",
    "The image is real with natural lighting and textures.",
]


def make_loader(n: int = 100, seed: int = 0, balance: float = 0.5):
    """
    Returns a loader_fn that yields (images, labels, references).
    balance: fraction of fake images (label=0).
    """
    rng = random.Random(seed)

    def loader_fn() -> Tuple[List[Image.Image], List[int], List[str]]:
        images, labels, references = [], [], []
        for i in range(n):
            label = 0 if rng.random() < balance else 1
            img = Image.new("RGB", (224, 224), (rng.randint(0, 255),) * 3)
            if label == 0:
                ref = rng.choice(_FAKE_RESPONSES)
            else:
                ref = rng.choice(_REAL_RESPONSES)
            images.append(img)
            labels.append(label)
            references.append(ref)
        return images, labels, references

    return loader_fn


# ---------------------------------------------------------------------------
# Dummy model factories
# ---------------------------------------------------------------------------

def make_weak_model(seed: int = 1):
    """Simulates a ~65% accurate model with noisy scores and vague responses."""
    rng = random.Random(seed)

    def predict(images: List[Image.Image]) -> List[dict]:
        results = []
        for _ in images:
            score = rng.random()
            label = 0 if score > 0.35 else 1
            # Vague responses reduce CSS scores
            response = (
                "The image shows some unusual features that may indicate manipulation."
                if rng.random() > 0.5
                else "The image could be real or synthetic, it is hard to tell."
            )
            results.append({"label": label, "score": score, "response": response})
        return results

    return predict


def make_strong_model(seed: int = 2):
    """Simulates a ~88% accurate model with confident scores and clear responses."""
    rng = random.Random(seed)

    def predict(images: List[Image.Image]) -> List[dict]:
        results = []
        for _ in images:
            score = rng.betavariate(2, 5) if rng.random() > 0.12 else rng.betavariate(5, 2)
            label = 0 if score < 0.5 else 1
            if label == 0:
                response = rng.choice(_FAKE_RESPONSES)
            else:
                response = rng.choice(_REAL_RESPONSES)
            results.append({"label": label, "score": 1.0 - score, "response": response})
        return results

    return predict


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    benchmarker = Benchmarker(
        models=[
            ("FakeVLM-Base",     make_weak_model(seed=1)),
            ("FakeVLM-Extended", make_strong_model(seed=2)),
        ],
        datasets=[
            ("FakeClue-Test",  make_loader(n=200, seed=10)),
            ("FakeClue-Train", make_loader(n=500, seed=20)),
        ],
        metrics=[accuracy, auc, f1, avg_precision, css, rouge_l],
        output_dir="./benchmark_output",
    )

    print("Running benchmark...")
    benchmarker.run()

    print("\n=== Results ===")
    print(benchmarker.results.to_string())

    print("\nExporting...")
    benchmarker.export_xlsx().export_png()

    print("\nDone. Check ./benchmark_output/")
