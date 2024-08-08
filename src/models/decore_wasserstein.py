from typing import List, Optional, Tuple

import copy
import os
import json
import torch
from torch.nn import functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.configs import DecoderConfigs, ModelConfigs

from src.models.base_model import BaseModel


class DeCoReWasserstein(BaseModel):
    def __init__(
        self,
        model_configs: ModelConfigs,
        decoder_configs: DecoderConfigs,
    ):
        super().__init__(model_configs, decoder_configs)

        self._load_retrieval_heads()
        print("Retrieval heads: ", self.retrieval_heads)

        self.alpha_cap = decoder_configs.configs.get("alpha_cap", None)

        self.scale_alpha = decoder_configs.configs.get("scale_alpha", False)

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

    def _calculate_wasserstein(self, ori_logits, hallucinated_logits):
        ori_probs = torch.softmax(ori_logits, dim=-1)
        hallucinated_probs = torch.softmax(hallucinated_logits, dim=-1)

        ori_cdf = torch.cumsum(ori_probs, dim=-1)
        hallucinated_cdf = torch.cumsum(hallucinated_probs, dim=-1)

        return torch.sum(torch.abs(ori_cdf - hallucinated_cdf), dim=-1)

    def generate(
        self,
        inputs,
        return_attentions: bool = False,
    ) -> dict:
        assert (
            not return_attentions
        ), "Return attentions not supported for DeCoReVanilla"
        self.model.eval()

        prompt = inputs["prompted_question"][0]
        inputs = self._verbalise_input(prompt).to(self.model.device)

        # Predict
        with torch.inference_mode():
            input_logits = self.model(
                input_ids=inputs[:, :-1], use_cache=True, return_dict=True
            )
            generated_ids = []
            last_input_token = inputs[:, -1]
            base_past_kv = copy.deepcopy(input_logits.past_key_values)
            hallucinated_past_kv = copy.deepcopy(input_logits.past_key_values)
            alphas = []
            for _ in range(self.max_new_tokens):
                last_input_token = last_input_token.view(1, 1)

                base_outputs = self.model(
                    input_ids=last_input_token,
                    past_key_values=base_past_kv,
                    use_cache=True,
                    attn_mode="torch",
                )
                hallucinated_outputs = self.model(
                    input_ids=last_input_token,
                    past_key_values=hallucinated_past_kv,
                    use_cache=True,
                    attn_mode="torch",
                    block_list=self.retrieval_heads,
                )

                base_past_kv = base_outputs.past_key_values
                hallucinated_past_kv = hallucinated_outputs.past_key_values

                alpha = self._calculate_wasserstein(
                    base_outputs.logits[0, -1], hallucinated_outputs.logits[0, -1]
                )

                if self.alpha_cap:
                    # If the entropy is too high, cap the alpha with the entropy cap
                    alpha = torch.min(
                        alpha, torch.tensor(self.alpha_cap).to(alpha.device)
                    )

                alphas += [alpha.item()]

                next_token_logits = (1 + alpha) * base_outputs.logits[
                    0, -1
                ] - alpha * hallucinated_outputs.logits[0, -1]

                last_input_token = next_token_logits.argmax()
                generated_ids.append(last_input_token.item())
                if last_input_token.item() == self.tokenizer.eos_token_id:
                    break
            decoded_text = self.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )

        return {"decoded_text": decoded_text, "attentions": {}, "alphas": alphas}

    def lm_score(
        self,
        prompt,
        answer,
    ):
        prompt = prompt["prompted_question"][0]
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

            # TODO: Probably should take the mean entropy of all tokens to be fair
            entropies = []
            for i in range(base_logits.shape[0]):
                entropies += [
                    self._calculate_wasserstein(
                        base_logits[i, :], hallucinated_logits[i, :]
                    )
                ]
            alpha = torch.max(torch.stack(entropies))

            if self.alpha_cap:
                # If the entropy is too high, cap the alpha with the entropy cap
                alpha = torch.min(alpha, torch.tensor(self.alpha_cap).to(alpha.device))

            base_logits = base_logits.log_softmax(dim=-1)
            hallucinated_logits = hallucinated_logits.log_softmax(dim=-1)

            diff_logits = (1 + alpha) * base_logits - alpha * hallucinated_logits

            if self.decoder_configs.configs.post_softmax:
                diff_logits = diff_logits.log_softmax(dim=-1)

            log_probs = (
                diff_logits[range(diff_logits.shape[0]), continue_ids].sum().item()
            )

        return log_probs
