import json
import os
from typing import List

from src.configs import DataConfigs
from src.datasets.base_dataset import BaseDataset


class XSum(BaseDataset):
    def __init__(
        self,
        data_configs: DataConfigs,
        **kwargs,
    ):
        super().__init__(data_configs, **kwargs)
        self.variation = data_configs.variation

        self.data_filename = os.path.join(self.data_dir, "xsum-1000.jsonl")

        # Prepare data
        self.data = self.parse_data()

    def parse_data(self) -> List[dict]:
        # Open the gz file, and read the jsonl file
        data = []

        with open(self.data_filename, "r") as f:
            for i, line in enumerate(f):
                instance = json.loads(line)
                data += [
                    {
                        "idx": instance["id"],
                        "document": instance["document"],
                        "summary": instance["summary"],
                    }
                ]

        if self.num_samples > 0:
            data = data[: self.num_samples]

        return data

    def build_prompt(self, context):
        verbalised_contexts = f"Article: {context}\n\n"
        verbalised_question = (
            f"Generate a summary comprising of 1 sentence for the article.\n"
        )
        answer_prefix = "Summary: "
        if self.kwargs["use_chat_template"]:
            input_text_prompt = [
                [f"{verbalised_contexts}{verbalised_question}{answer_prefix}"]
            ]
        else:
            input_text_prompt = (
                f"{verbalised_contexts}{verbalised_question}{answer_prefix}"
            )
        return {
            "verbalised_instruction": "",
            "verbalised_icl_demo": "",
            "verbalised_contexts": verbalised_contexts,
            "verbalised_question": verbalised_question,
            "verbalised_answer_prefix": answer_prefix,
            "prompted_question": input_text_prompt,
        }

    def __getitem__(self, idx):
        sample = self.data[idx]

        prompt = self.build_prompt(sample["document"])

        # For attention analysis
        sample["verbalised_instruction"] = prompt["verbalised_instruction"]
        sample["verbalised_icl_demo"] = prompt["verbalised_icl_demo"]
        sample["verbalised_contexts"] = prompt["verbalised_contexts"]
        sample["verbalised_question"] = prompt["verbalised_question"]
        sample["verbalised_answer_prefix"] = prompt["verbalised_answer_prefix"]

        sample["prompted_question"] = prompt["prompted_question"]

        return sample

    def __len__(self):
        return len(self.data)
