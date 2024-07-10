from typing import Dict

import numpy as np


class MemoTrap:
    def __init__(self):
        pass

    def __call__(self, predictions) -> Dict[str, float]:
        scores = {
            "proverb_ending": [],
            "proverb_translation": [],
            "hate_speech_ending": [],
            "history_of_science_qa": [],
        }
        for sample in predictions:
            scores_true = sample["scores_true"][0]
            scores_false = sample["scores_false"][0]
            split = sample["split"][0]
            print(scores_true)
            print(scores_false)
            print(sample["split"])
            scores[split] += [1.0 if scores_true > scores_false else 0.0]

        metrics = {f"{split}_acc": np.mean(scores[split]) for split in scores.keys()}
        metrics["macro_avg_acc"] = np.mean(list(metrics.values()))
        metrics["micro_avg_acc"] = np.mean([metrics[split] for split in scores.keys()])
        return metrics
