from typing import List, Dict
import re
import numpy as np
from src.metrics.base_metric import BaseMetric
from src.configs import FrameworkConfigs

class WikiMultihopQA(BaseMetric):
    def __init__(self, framework_configs: FrameworkConfigs, **kwargs):
        super().__init__(framework_configs, **kwargs)

    def compute_metrics(self, prediction: str, refs: List[str]):
        scores = {}
        scores["Subspan_EM"] = self.unnormalised_best_subspan_em(prediction, refs)
        scores["F1"] = self.compute_f1(prediction, refs)
        scores["Recall"] = self.compute_recall(prediction, refs)
        return scores
    
    @staticmethod
    def unnormalised_best_subspan_em(
        prediction: str, ground_truths: List[str]
    ) -> float:
        for ground_truth in ground_truths:
            if ground_truth.lower() in prediction.lower():
                return 1.0
        return 0.0

    @staticmethod
    def compute_f1(prediction: str, ground_truths: List[str]) -> float:
        # Tokenize the prediction
        prediction_tokens = set(prediction.lower().split())

        # Flatten the ground_truths list of tokens
        ground_truth_tokens = set(token for gt in ground_truths for token in gt.lower().split())

        # Compute the intersection of prediction tokens with the ground truth tokens
        intersection = prediction_tokens.intersection(ground_truth_tokens)
        precision = len(intersection) / len(prediction_tokens) if prediction_tokens else 0
        recall = len(intersection) / len(ground_truth_tokens) if ground_truth_tokens else 0

        # Compute the F1 score
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return f1_score

    @staticmethod
    def compute_recall(prediction: str, ground_truths: List[str]) -> float:
        # Tokenize the prediction
        prediction_tokens = set(prediction.lower().split())

        # Flatten the ground_truths list of tokens
        ground_truth_tokens = set(token for gt in ground_truths for token in gt.lower().split())

        # Compute the recall score
        intersection = prediction_tokens.intersection(ground_truth_tokens)
        recall = len(intersection) / len(ground_truth_tokens) if ground_truth_tokens else 0
        return recall

    def __call__(self, predictions) -> Dict[str, float]:
        subspan_em_scores = []
        f1_scores = []
        recall_scores = []
        for sample in predictions:
            refs = [
                ans[0] if type(ans) in [list, tuple] else ans
                for ans in sample["answers"]
            ]

            prediction = self.answer_extractor(sample["predicted_answer"])

            scores = self.compute_metrics(prediction, refs)

            subspan_em_scores += [scores["Subspan_EM"]]
            f1_scores.append(scores["F1"])
            recall_scores.append(scores["Recall"])

        metrics = {
            "Subspan_EM": np.mean(subspan_em_scores),
            "F1": np.mean(f1_scores),
            "Recall": np.mean(recall_scores),
        }
        return metrics
