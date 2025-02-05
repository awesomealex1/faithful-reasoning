from typing import Dict, List

import regex as re
import numpy as np

from src.metrics.utils import best_em, best_subspan_em
from src.metrics.base_metric import BaseMetric
from src.configs import FrameworkConfigs


class TriviaQA(BaseMetric):
    def __init__(self, framework_configs: FrameworkConfigs, **kwargs):
        super().__init__(framework_configs, **kwargs)


    @staticmethod
    def compute_metrics(prediction: str, refs: List[str]):
        scores = {}
        scores["EM"] = best_em(prediction, refs)
        scores["Subspan_EM"] = best_subspan_em(prediction, refs)

        return scores

    def __call__(self, predictions) -> Dict[str, float]:
        em_scores = []
        subspan_em_scores = []
        for sample in predictions:
            refs = [
                ans[0] if type(ans) in [list, tuple] else ans
                for ans in sample["answers"]
            ]

            # Only consider until \n, ., or ,
            prediction = re.split("\n|\.|\,", sample["predicted_answer"])[0]

            scores = self.compute_metrics(prediction, refs)

            em_scores += [scores["EM"]]
            subspan_em_scores += [scores["Subspan_EM"]]

        metrics = {
            "EM": np.mean(em_scores),
            "Subspan_EM": np.mean(subspan_em_scores),
        }
        return metrics
