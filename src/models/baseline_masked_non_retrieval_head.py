import json
import os
import random
from typing import List, Optional, Tuple

import numpy as np
import torch

from src.configs import DecoderConfigs, ModelConfigs
from src.models.base_model import BaseModel


class BaselineMaskedNonRetrievalHead(BaseModel):
    def __init__(
        self,
        model_configs: ModelConfigs,
        decoder_configs: DecoderConfigs,
    ):
        super().__init__(model_configs, decoder_configs)

        self._load_retrieval_heads()
        self.num_retrieval_heads = self.decoder_configs.configs.num_retrieval_heads
        assert (
            self.num_retrieval_heads < 0,
            "Number of retrieval heads should be negative",
        )  # negative number of retrieval heads to signify selecting random heads
        self.random_heads = self._construct_random_head(-self.num_retrieval_heads)
        print("Random heads: ", self.random_heads)

    def _load_retrieval_heads(self):
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
        ][:100]

    def _construct_random_head(self, n):
        results = []
        seed_list = [
            i for i in range(32)
        ]  # FIXME: 32 is hardcoded, copied from Retrieval_Head repo
        random.shuffle(seed_list)
        while len(results) < n:
            l, h = random.choices(seed_list, k=2)
            if (l, h) in results or (l, h) in self.retrieval_heads:
                continue
            else:
                results.append((l, h))
        return results

    def generate(
        self,
        inputs,
    ) -> dict:
        self.model.eval()

        inputs = self._verbalise_input(inputs).to(self.model.device)

        # Predict
        with torch.inference_mode():
            input_logits = self.model(
                input_ids=inputs[:, :-1], use_cache=True, return_dict=True
            )
            generated_ids = []
            last_input_token = inputs[:, -1]
            past_kv = input_logits.past_key_values
            for _ in range(self.max_new_tokens):
                last_input_token = last_input_token.view(1, 1)
                outputs = self.model(
                    input_ids=last_input_token,
                    past_key_values=past_kv,
                    use_cache=True,
                    attn_mode="torch",
                    block_list=self.random_heads,
                )
                past_kv = outputs.past_key_values
                last_input_token = outputs.logits[0, -1].argmax()
                generated_ids.append(last_input_token.item())
                if last_input_token.item() == self.tokenizer.eos_token_id:
                    break
            decoded_text = self.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )

        return {"decoded_text": decoded_text}

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

            outputs = self.model(input_ids, block_list=self.random_heads)[0]
            outputs = outputs.squeeze(0).log_softmax(-1)  # logits to log probs

            # skip tokens in the prompt -- we only care about the answer
            outputs = outputs[prefix_ids.shape[-1] - 1 : -1, :]

            # get logprobs for each token in the answer
            log_probs = outputs[range(outputs.shape[0]), continue_ids].sum().item()

        return log_probs
