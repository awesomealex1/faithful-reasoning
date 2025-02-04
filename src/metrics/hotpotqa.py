from typing import List

class HotpotQA:
    def __init__(self):
        pass

    def compute_metrics(self, prediction: str, refs: List[str]):
        scores = {}
        scores["Subspan_EM"] = self.unnormalised_best_subspan_em(prediction, refs)

        return scores
