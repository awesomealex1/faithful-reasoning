from typing import Dict

import numpy as np
from src.metrics.base_metric import BaseMetric
from src.configs import FrameworkConfigs


class MemoTrap(BaseMetric):
    def __init__(self, framework_configs: FrameworkConfigs, **kwargs):
        super().__init__(framework_configs, **kwargs)

    def __call__(self, predictions: dict) -> Dict[str, float]:
        """
        Compute macro and micro accuracy for MemoTrap.

        Args:
            predictions (dict): Predictions from the model.

        Returns:
            Dict[str, float]: Macro and micro accuracy.
        """
        scores = {
            "all": [],
            "proverb_ending": [],
            "proverb_translation": [],
            "hate_speech_ending": [],
            "history_of_science_qa": [],
        }
        for sample in predictions:
            scores_true = sample["scores_true"][0]
            scores_false = sample["scores_false"][0]
            split = sample["split"][0]
            score = 1.0 if scores_true > scores_false else 0.0
            scores[split] += [score]
            scores["all"] += [score]

        metrics = {f"{split}_acc": np.mean(scores[split]) for split in scores.keys()}
        metrics["macro_avg_acc"] = np.mean(list(metrics.values()))
        metrics["micro_avg_acc"] = np.mean(scores["all"])

        return metrics
