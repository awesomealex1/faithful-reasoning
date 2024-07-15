from typing import List, Optional, Tuple

import os
import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.configs import DecoderConfigs, ModelConfigs

from src.models.base_model import BaseModel


class DeCoReVanilla(BaseModel):
    def __init__(
        self,
        model_configs: ModelConfigs,
        decoder_configs: DecoderConfigs,
    ):
        super().__init__(model_configs, decoder_configs)

        self._load_retrieval_heads()
        print("Retrieval heads: ", self.retrieval_heads)

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
            generated_tokens = torch.tensor(
                [[]], dtype=torch.long, device=self.model.device
            )
            for forward_pass in range(self.max_new_tokens):
                base_output = self.model.generate(
                    input_ids=torch.cat([inputs, generated_tokens], dim=1),
                    max_new_tokens=1,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                hallucinated_output = self.model.generate(
                    input_ids=torch.cat([inputs, generated_tokens], dim=1),
                    max_new_tokens=1,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    block_list=self.retrieval_heads,
                )

                base_score = base_output["scores"][0][0].cpu()
                hallucinated_score = hallucinated_output["scores"][0][0].cpu()

                # Aggregate teacher LM and student LM scores

                next_token_score = (
                    (1 + self.decoder_configs.configs.alpha) * base_score
                    - self.decoder_configs.configs.alpha * hallucinated_score
                )

                _, indices = torch.topk(next_token_score, 1)
                generated_tokens = torch.cat(
                    [generated_tokens, indices.unsqueeze(dim=0).to(self.model.device)],
                    dim=1,
                )

            decoded_text = self.tokenizer.decode(
                generated_tokens[0], skip_special_tokens=True
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

            base_outputs = self.model(input_ids)[0]
            hallucinated_outputs = self.model(
                input_ids, block_list=self.retrieval_heads
            )[0]

            base_logits = base_outputs[0, prefix_ids.shape[-1] - 1 : -1, :]
            hallucinated_logits = hallucinated_outputs[
                0, prefix_ids.shape[-1] - 1 : -1, :
            ]

            base_logits = base_logits.log_softmax(dim=-1)
            hallucinated_logits = hallucinated_logits.log_softmax(dim=-1)
            diff_logits = base_logits - hallucinated_logits

            # diff_logits = diff_logits.log_softmax(dim=-1)

            log_probs = (
                diff_logits[range(diff_logits.shape[0]), continue_ids].sum().item()
            )

        return log_probs
