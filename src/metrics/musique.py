from typing import Dict, List

import regex as re
import numpy as np

from src.metrics.utils import best_em, best_subspan_em


class MuSiQue:
    def __init__(self):
        pass

    @staticmethod
    def compute_metrics(prediction: str, refs: List[str]):
        scores = {}
        scores["EM"] = best_em(prediction, refs)
        scores["Subspan_EM"] = best_subspan_em(prediction, refs)

        return scores

    @staticmethod
    def answer_extractor(cot_answer: str) -> str:
        # Adapted from https://github.com/StonyBrookNLP/ircot/blob/main/evaluate.py

        cot_regex = re.compile(".* answer is:? (.*)\\.?")
        match = cot_regex.match(cot_answer)
        if match:
            output = match.group(1)
            if output.endswith("."):
                output = output[:-1]
        else:
            output = cot_answer

        return output

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
            # Extract answer from the CoT reasonings
            prediction = self.answer_extractor(prediction)
            print("refs: ", refs)
            print("prediction: ", prediction)

            scores = self.compute_metrics(prediction, refs)
            print("scores: ", scores)

            em_scores += [scores["EM"]]
            subspan_em_scores += [scores["Subspan_EM"]]

        metrics = {
            "EM": np.mean(em_scores),
            "Subspan_EM": np.mean(subspan_em_scores),
        }
        return metrics
