import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import pandas as pd


class Benchmarker:
    """
    Cross-model, cross-dataset evaluation framework.

    Usage (hybrid — constructor + incremental add_* methods):

        benchmarker = Benchmarker(
            models=[("FakeVLM-Base", base_predict_fn)],
            datasets=[("FakeClue-Test", test_loader_fn)],
            metrics=[accuracy, auc, f1, avg_precision, css, rouge_l],
            output_dir="./results",
        )
        benchmarker.add_model("FakeVLM-Extended", ext_predict_fn)
        benchmarker.run().export_xlsx().export_png()

    predict_fn signature:
        def predict_fn(images: List[PIL.Image]) -> List[dict]:
            # Each dict must have:
            #   "label": int   — 0=fake, 1=real
            #   "score": float — confidence (used for AUC/AP)
            #   "response": str — model text (used for Rouge-L and CSS; optional)

    loader_fn signature:
        def loader_fn() -> Tuple[List[PIL.Image], List[int], List[str]]:
            # Returns: (images, labels, reference_texts)
            # labels:     0=fake, 1=real
            # references: ground-truth explanation text (may be empty strings)
    """

    def __init__(
        self,
        models: Optional[List[Tuple]] = None,
        datasets: Optional[List[Tuple]] = None,
        metrics: Optional[List] = None,
        output_dir: Union[str, Path] = "./benchmark_results",
    ):
        self._models: List[Dict] = []
        self._datasets: List[Dict] = []
        self._metrics: List = []
        self._results: Optional[pd.DataFrame] = None
        self.output_dir = Path(output_dir)

        for item in models or []:
            self.add_model(*item)
        for item in datasets or []:
            self.add_dataset(*item)
        for metric in metrics or []:
            self.add_metric(metric)

    # ------------------------------------------------------------------
    # Builder methods
    # ------------------------------------------------------------------

    def add_model(self, name: str, predict_fn: Callable) -> "Benchmarker":
        self._models.append({"name": name, "predict_fn": predict_fn})
        return self

    def add_model_from_results(
        self, name: str, results_json_path: Union[str, Path]
    ) -> "Benchmarker":
        """
        Register a model from a pre-computed eval.py JSON output.

        Expected JSON format (list of dicts):
            [{"image": "...", "ground_truth_label": "fake"|"real",
              "response": "...", "question": "..."}, ...]
        """
        json_path = Path(results_json_path)

        def _predict_from_json(images):
            with open(json_path) as f:
                records = json.load(f)
            label_map = {"fake": 0, "real": 1}
            preds = []
            for r in records:
                label_str = r.get("ground_truth_label", "real")
                preds.append({
                    "label": label_map.get(label_str.lower(), 1),
                    "score": 0.5,   # score not available from eval.py output
                    "response": r.get("response", ""),
                })
            return preds

        self._models.append({"name": name, "predict_fn": _predict_from_json})
        return self

    def add_dataset(self, name: str, loader_fn: Callable) -> "Benchmarker":
        self._datasets.append({"name": name, "loader_fn": loader_fn})
        return self

    def add_metric(self, metric) -> "Benchmarker":
        self._metrics.append(metric)
        return self

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self) -> "Benchmarker":
        """Run inference across all (model, dataset) pairs and compute metrics."""
        if not self._models:
            raise ValueError("No models registered. Use add_model() first.")
        if not self._datasets:
            raise ValueError("No datasets registered. Use add_dataset() first.")
        if not self._metrics:
            raise ValueError("No metrics registered. Use add_metric() first.")

        records = []
        for model in self._models:
            for dataset in self._datasets:
                images, labels, references = dataset["loader_fn"]()
                preds = model["predict_fn"](images)

                y_pred = [p["label"] for p in preds]
                y_score = [p.get("score", 0.5) for p in preds]
                responses = [p.get("response", "") for p in preds]

                row: Dict = {"Model": model["name"], "Dataset": dataset["name"]}
                for metric in self._metrics:
                    result = metric(
                        y_true=labels,
                        y_pred=y_pred,
                        y_score=y_score,
                        responses=responses,
                        references=references,
                    )
                    if isinstance(result, dict):
                        row.update(result)
                    else:
                        row[metric.name] = result

                records.append(row)

        self._results = pd.DataFrame(records).set_index(["Model", "Dataset"])
        return self

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    @property
    def results(self) -> pd.DataFrame:
        if self._results is None:
            raise RuntimeError("Call run() before accessing results.")
        return self._results

    def export_xlsx(self, path: Optional[Union[str, Path]] = None) -> "Benchmarker":
        from .export import export_xlsx
        out = Path(path) if path else self.output_dir / "benchmark.xlsx"
        export_xlsx(self.results, out)
        print(f"Saved: {out}")
        return self

    def export_png(self, path: Optional[Union[str, Path]] = None) -> "Benchmarker":
        from .export import export_png
        out = Path(path) if path else self.output_dir / "benchmark.png"
        export_png(self.results, out)
        print(f"Saved: {out}")
        return self

    def export(
        self,
        stem: Optional[Union[str, Path]] = None,
    ) -> "Benchmarker":
        """Export both .xlsx and .png. stem may be a path without extension."""
        if stem is None:
            return self.export_xlsx().export_png()
        stem = Path(stem)
        return self.export_xlsx(stem.with_suffix(".xlsx")).export_png(stem.with_suffix(".png"))
