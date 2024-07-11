from typing import Dict

import string
from typing import List

import regex as re
import numpy as np


def normalize_answer(s: str) -> str:
    """Normalization from the SQuAD evaluation script.

    See https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
    """

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def best_em(prediction: str, ground_truths: List[str]) -> float:
    normalized_prediction = normalize_answer(prediction)

    for ground_truth in ground_truths:
        normalized_ground_truth = normalize_answer(ground_truth)
        if normalized_ground_truth.lower() == normalized_prediction.lower():
            return 1.0
    return 0.0


def best_subspan_em(prediction: str, ground_truths: List[str]) -> float:
    normalized_prediction = normalize_answer(prediction)

    for ground_truth in ground_truths:
        normalized_ground_truth = normalize_answer(ground_truth)
        if normalized_ground_truth.lower() in normalized_prediction.lower():
            return 1.0
    return 0.0


class NQSwap:
    def __init__(self):
        pass

    @staticmethod
    def compute_metrics(prediction: str, sub_ref: List[str], org_ref: List[str]):
        scores = {}
        scores["sub_EM"] = best_em(prediction, sub_ref)
        scores["sub_Subspan_EM"] = best_subspan_em(prediction, sub_ref)

        scores["org_EM"] = best_em(prediction, org_ref)
        scores["org_Subspan_EM"] = best_subspan_em(prediction, org_ref)

        return scores

    def __call__(self, predictions) -> Dict[str, float]:
        sub_em_scores = []
        sub_subspan_em_scores = []
        org_em_scores = []
        org_subspan_em_scores = []
        for sample in predictions:
            org_ref = [
                ans[0] if type(ans) in [list, tuple] else ans
                for ans in sample["org_answer"]
            ]
            sub_ref = [
                ans[0] if type(ans) in [list, tuple] else ans
                for ans in sample["sub_answer"]
            ]

            # Only consider until \n, ., or ,
            prediction = re.split("\n|\.|\,", sample["predicted_answer"])[0]

            scores = self.compute_metrics(prediction, sub_ref, org_ref)

            sub_em_scores += [scores["sub_EM"]]
            sub_subspan_em_scores += [scores["sub_Subspan_EM"]]

            org_em_scores += [scores["org_EM"]]
            org_subspan_em_scores += [scores["org_Subspan_EM"]]

        metrics = {
            "sub_EM": np.mean(sub_em_scores),
            "sub_Subspan_EM": np.mean(sub_subspan_em_scores),
            "org_EM": np.mean(org_em_scores),
            "org_Subspan_EM": np.mean(org_subspan_em_scores),
        }
        return metrics
