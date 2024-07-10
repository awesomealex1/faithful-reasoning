from typing import Dict

import numpy as np


class MemoTrap:
    def __init__(self):
        pass

    @staticmethod
    def compute_metrics(score_true, score_false):
        # Compute metrics by splits
        scores = {}
        scores[""] = 1.0 if score_true > score_false else 0.0
        scores["proverb_ending"] = 
        scores["proverb_translation"] = 
        scores["hate_speech_ending"] = 
        scores["history_of_science_qa"] = 

        return scores

    def __call__(self, predictions) -> Dict[str, float]:
        scores = {}
        for sample in predictions:
            scores_true = sample["scores_true"][0]
            scores_false = sample["scores_false"][0]
            scores[sample["split"]] += [1.0 if scores_true > scores_false else 0.0]
        
        metrics = {
            f"{split}_acc": np.mean(scores[split])
            for split in scores.keys()
        }
        metrics["macro_avg_acc"] = np.mean(list(metrics.values()))
        metrics["micro_avg_acc"] = np.mean(
            [
                metrics[split]
                for split in scores.keys()
            ]
        )
        return metrics
