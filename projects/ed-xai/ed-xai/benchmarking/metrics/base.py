from dataclasses import dataclass
from typing import Callable, Dict, List, Union


@dataclass
class MetricFn:
    """
    Wraps a metric function with a display name.

    The underlying callable receives all prediction components and should
    return either a float or a dict[str, float] (for multi-value metrics
    like CSS, which expand into separate columns).
    """

    name: str
    fn: Callable

    def __call__(
        self,
        y_true: List[int],
        y_pred: List[int],
        y_score: List[float],
        responses: List[str],
        references: List[str],
    ) -> Union[float, Dict[str, float]]:
        return self.fn(
            y_true=y_true,
            y_pred=y_pred,
            y_score=y_score,
            responses=responses,
            references=references,
        )
