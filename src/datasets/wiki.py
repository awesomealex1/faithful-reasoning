from src.datasets.base_dataset import BaseDataset
from src.configs import DataConfigs
from typing import List
import os
import json

class WikiMultihopQA(BaseDataset):

    def __init__(
        self,
        data_configs: DataConfigs,
        **kwargs,
    ):
        super().__init__(data_configs, **kwargs)
        self.variation = data_configs.variation

        self.data_filename = os.path.join(self.data_dir, "dev_subsampled.jsonl")

        # Prepare data
        self.data = self.parse_data()

    def parse_data(self) -> List[dict]:
        data = []

        with open(self.data_filename, "r") as f:
            for i, line in enumerate(f):
                instance = json.loads(line)
                data.append({
                        "idx": instance["question_id"],
                        "question": instance["question_text"],
                        "answers": instance["answers_objects"][0]["spans"],
                    }
                )

        if self.num_samples > 0:
            data = data[: self.num_samples]

        return data

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample

    def __len__(self):
        return len(self.data)