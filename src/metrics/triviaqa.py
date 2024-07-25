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


class TriviaQA:
    def __init__(self):
        pass

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
