from typing import Dict, List

import regex as re
import numpy as np

from src.metrics.ifeval.utils import process_results


class IFEval:
    def __init__(self):
        pass

    def __call__(self, predictions) -> Dict[str, float]:
        # TODO: process_results
        doc = {
            "key": predictions["key"],
            "instruction_id_list": predictions["instruction_id_list"],
            "prompt": predictions["prompt"],
            "kwargs": predictions["kwargs"],
        }
        process_results(doc, predictions["predicted_answer"])
