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

        self.data_filename = os.path.join(self.data_dir, "train.json")

        # Prepare data
        self.data = self.parse_data()
    
    def parse_data(self) -> List[dict]:
        if "jsonl" in self.data_filename:
            return self.parse_data_jsonl()
        else:
            return self.parse_data_json()

    def parse_data_json(self) -> List[dict]:
        data = []
        skipped = 0

        with open(self.data_filename, "r") as f:
            raw_data = json.load(f)
            i = 0

            for _, instance in enumerate(raw_data):
                if "answer" not in instance:
                    skipped += 1
                    continue
                data.append({
                        "idx": i,
                        "question": instance["question"],
                        "answers": [instance["answer"]],
                        "question_id": instance["_id"]
                    }
                )
                i += 1
        
        print(f"Skipped {skipped} unanswerable questions.")
        if self.num_samples > 0:
            data = data[: self.num_samples]
        data = data[167:]   # Temporary while creating ft dataset

        return data

    def parse_data_jsonl(self) -> List[dict]:
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