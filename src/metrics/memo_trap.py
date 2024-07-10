from typing import Dict

import numpy as np


class MemoTrap:
    def __init__(self):
        pass

    def __call__(self, predictions) -> Dict[str, float]:
        scores = []
        for sample in predictions:
            scores_true = sample["scores_true"][0]
            scores_false = sample["scores_false"][0]
            scores += [1.0 if scores_true > scores_false else 0.0]
        metrics = {"acc": np.mean(scores)}
        return metrics
