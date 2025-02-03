from base_dataset import BaseDataset
from configs import DataConfigs
from typing import List
import os

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
        pass

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample

    def __len__(self):
        return len(self.data)