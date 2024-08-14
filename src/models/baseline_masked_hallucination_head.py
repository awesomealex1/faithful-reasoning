import json
import os
from typing import List, Optional, Tuple

import numpy as np
import torch

from src.configs import DecoderConfigs, ModelConfigs
from src.models.base_model import BaseModel


class BaselineMaskedHallucinationHead(BaseModel):
    def __init__(
        self,
        model_configs: ModelConfigs,
        decoder_configs: DecoderConfigs,
    ):
        super().__init__(model_configs, decoder_configs)

        self._load_hallucination_heads()
        print("Hallucination heads: ", self.hallucination_heads)
        print("Hallucination head scores: ", self.hallucination_head_scores)

    def _load_hallucination_heads(self):
        self.num_hallucination_heads = (
            self.decoder_configs.configs.num_hallucination_heads
        )

        model_base_name = self.model_configs.configs.model_name_or_path.split("/")[1]
        with open(
            os.path.join(
                self.decoder_configs.configs.hallucination_heads_dir,
                f"{model_base_name}.json",
            )
        ) as file:
            head_list = json.loads(file.readline())

        hallucination_heads = {}
        for feature_name in self.decoder_configs.configs.feature_names:
            if feature_name in head_list:
                for layer_head, value in head_list[feature_name].items():
                    if layer_head not in hallucination_heads:
                        hallucination_heads[layer_head] = []
                    hallucination_heads[layer_head].append(value)

        if self.decoder_configs.configs.aggregation_method == "mean":
            hallucination_heads = {
                k: np.mean(v) for k, v in hallucination_heads.items()
            }
        elif self.decoder_configs.configs.aggregation_method == "max":
            hallucination_heads = {k: np.max(v) for k, v in hallucination_heads.items()}
        else:
            raise ValueError(
                f"Unknown hallucination head aggregation method: {self.decoder_configs.configs.aggregation_method}"
            )

        hallucination_heads = sorted(hallucination_heads.items(), key=lambda x: x[1])
        self.hallucination_heads = [
            [int(ll) for ll in l[0].split("-")] for l in hallucination_heads
        ][: self.num_hallucination_heads]
        self.hallucination_head_scores = [l[1] for l in hallucination_heads][
            : self.num_hallucination_heads
        ]

    def generate(
        self,
        inputs,
        return_attentions: bool = False,
    ) -> dict:
        return self._generate(
            inputs,
            return_attentions=return_attentions,
            block_list=self.hallucination_heads,
        )

    def lm_score(
        self,
        prompt,
        answer,
    ):
        prompted_question = prompt["prompted_question"][0]

        if len(prompt["verbalised_instruction"][0]):
            use_system_prompt = True
        else:
            use_system_prompt = False

        with torch.no_grad():
            if type(prompted_question) == list:
                input_text = prompted_question + [answer]
            else:
                input_text = prompted_question + answer
            input_ids = self._verbalise_input(
                input_text,
                use_system_prompt=use_system_prompt,
                add_generation_prompt=False,
            ).to(self.model.device)
            prefix_ids = self._verbalise_input(
                prompted_question, use_system_prompt=use_system_prompt
            ).to(self.model.device)
            continue_ids = input_ids[0, prefix_ids.shape[-1] :]

            outputs = self.model(input_ids, block_list=self.hallucination_heads)[0]
            outputs = outputs.squeeze(0).log_softmax(-1)  # logits to log probs

            # skip tokens in the prompt -- we only care about the answer
            outputs = outputs[prefix_ids.shape[-1] - 1 : -1, :]

            # get logprobs for each token in the answer
            log_probs = outputs[range(outputs.shape[0]), continue_ids].sum().item()

        return log_probs
