from typing import List

from .base import MetricFn


def _rouge_l(responses: List[str], references: List[str], **kwargs) -> float:
    try:
        from rouge_score import rouge_scorer as rs_module
    except ImportError as e:
        raise ImportError("rouge-score is required: pip install rouge-score") from e

    scorer = rs_module.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    for resp, ref in zip(responses, references):
        if ref and resp:
            score = scorer.score(ref, resp)
            scores.append(score["rougeL"].fmeasure)
    if not scores:
        return 0.0
    return round(sum(scores) / len(scores), 4)


rouge_l = MetricFn(name="Rouge-L", fn=_rouge_l)
