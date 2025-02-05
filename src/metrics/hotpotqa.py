from typing import List, Dict
import re
import numpy as np
from src.metrics.base_metric import BaseMetric
from src.configs import FrameworkConfigs

class HotpotQA(BaseMetric):
    def __init__(self, framework_configs: FrameworkConfigs, **kwargs):
        super().__init__(framework_configs, **kwargs)

    def compute_metrics(self, prediction: str, refs: List[str]):
        scores = {}
        scores["Subspan_EM"] = self.unnormalised_best_subspan_em(prediction, refs)

        return scores
    
    @staticmethod
    def unnormalised_best_subspan_em(
        prediction: str, ground_truths: List[str]
    ) -> float:
        for ground_truth in ground_truths:
            if ground_truth.lower() in prediction.lower():
                return 1.0
        return 0.0

    def __call__(self, predictions) -> Dict[str, float]:
        subspan_em_scores = []
        for sample in predictions:
            refs = [
                ans[0] if type(ans) in [list, tuple] else ans
                for ans in sample["answers"]
            ]

            prediction = self.answer_extractor(sample["predicted_answer"])

            scores = self.compute_metrics(prediction, refs)

            subspan_em_scores += [scores["Subspan_EM"]]

        metrics = {
            "Subspan_EM": np.mean(subspan_em_scores),
        }
        return metrics