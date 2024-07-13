import json
import os
from typing import List, Optional, Tuple

import numpy as np
import torch

from src.configs import DecoderConfigs, ModelConfigs
from src.models.base_model import BaseModel


class BaselineMaskedRetrievalHead(BaseModel):
    def __init__(
        self,
        model_configs: ModelConfigs,
        decoder_configs: DecoderConfigs,
    ):
        super().__init__(model_configs, decoder_configs)

        self._load_retrieval_heads()
        print(self.retrieval_heads)

    def _load_retrieval_heads(self):
        self.num_retrieval_heads = self.decoder_configs.configs.num_retrieval_heads

        model_base_name = self.model_configs.configs.model_name_or_path.split("/")[1]

        with open(
            os.path.join(
                self.decoder_configs.configs.retrieval_heads_dir,
                f"{model_base_name}.json",
            )
        ) as file:
            head_list = json.loads(file.readline())

        stable_block_list = [(l[0], np.mean(l[1])) for l in head_list.items()]
        stable_block_list = sorted(stable_block_list, key=lambda x: x[1], reverse=True)
        self.retrieval_heads = [
            [int(ll) for ll in l[0].split("-")] for l in stable_block_list
        ][: self.num_retrieval_heads]

    def generate(
        self,
        inputs,
    ) -> str:
        self.model.eval()

        inputs = self._verbalise_input(inputs).to(self.model.device)

        # Predict
        with torch.inference_mode():
            output = self.model.generate(
                inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                block_list=self.retrieval_heads,
            )
            decoded_text = self.tokenizer.decode(
                output[0, inputs.size(1) :], skip_special_tokens=True
            )

        return decoded_text

    def lm_score(
        self,
        prompt,
        answer,
    ):
        with torch.no_grad():
            if type(prompt) == list:
                input_text = prompt + [answer]
            else:
                input_text = prompt + answer
            input_ids = self._verbalise_input(input_text).to(self.model.device)
            prefix_ids = self._verbalise_input(prompt).to(self.model.device)
            continue_ids = input_ids[0, prefix_ids.shape[-1] :]

            outputs = self.model(input_ids, block_list=self.retrieval_heads)[0]
            outputs = outputs.squeeze(0).log_softmax(-1)  # logits to log probs

            # skip tokens in the prompt -- we only care about the answer
            outputs = outputs[prefix_ids.shape[-1] - 1 : -1, :]

            # get logprobs for each token in the answer
            log_probs = outputs[range(outputs.shape[0]), continue_ids].sum().item()

        return log_probs
