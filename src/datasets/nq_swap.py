import gzip
import json
import math
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import PreTrainedTokenizer

from src.configs import DataConfigs, DecoderConfigs, PromptConfigs
from src.datasets.base_dataset import BaseDataset


class NQSwap(BaseDataset):
    def __init__(
        self,
        data_configs: DataConfigs,
        **kwargs,
    ):
        super().__init__(data_configs, **kwargs)
        self.data_filename = os.path.join(
            self.data_dir, self.available_variations[data_configs.gold_at]
        )

        # Prepare data
        self.data = self.parse_data()

    def parse_data(self) -> dict:
        # Open the gz file, and read the jsonl file
        data = {
            "question": [],
            "answers": [],
            "contexts": [],
        }
        with gzip.open(self.data_filename, "rb") as f:
            for line in f:
                instance = json.loads(line)
                data["question"] += [instance["question"]]
                data["answers"] += [instance["answers"]]
                data["contexts"] += [
                    [context["title"] + context["text"] for context in instance["ctxs"]]
                ]

        return {
            "question": data["question"],
            "answers": data["answers"],
            "contexts": data["contexts"],
        }

    def __getitem__(self, idx):
        return {
            "contexts": self.data["contexts"][idx],
            "question": self.data["question"][idx],
            "answers": self.data["answers"][idx],
        }

    def __len__(self):
        return len(self.data["question"])
