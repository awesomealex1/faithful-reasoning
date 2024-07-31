from typing import List, Optional, Tuple

import copy
import os
import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.configs import DecoderConfigs, ModelConfigs

from src.models.base_model import BaseModel


class DeCoReGuided(BaseModel):
    def __init__(
        self,
        model_configs: ModelConfigs,
        decoder_configs: DecoderConfigs,
    ):
        super().__init__(model_configs, decoder_configs)

        self._load_retrieval_heads()
        print("Retrieval heads: ", self.retrieval_heads)

        self._load_classifier()

    def _load_classifier(self):
        self.classifier = torch.load(self.decoder_configs.configs.classifier_path)[
            "model"
        ]
        self.classifier_max_sequence_length = (
            self.decoder_configs.configs.classifier_max_sequence_length
        )
        self.classifier_num_samples = (
            self.decoder_configs.configs.classifier_num_samples
        )

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

    def _prepare_lookback_ratios(self, lookback_ratios):
        sample = torch.cat(
            [
                lookback_ratios["bos"][
                    :, :, : self.classifier_max_sequence_length
                ].unsqueeze(-1),
                lookback_ratios["instruction"][
                    :, :, : self.classifier_max_sequence_length
                ].unsqueeze(-1),
                lookback_ratios["icl_demo"][
                    :, :, : self.classifier_max_sequence_length
                ].unsqueeze(-1),
                lookback_ratios["contexts"][
                    :, :, : self.classifier_max_sequence_length
                ].unsqueeze(-1),
                lookback_ratios["question"][
                    :, :, : self.classifier_max_sequence_length
                ].unsqueeze(-1),
                lookback_ratios["answer_prefix"][
                    :, :, : self.classifier_max_sequence_length
                ].unsqueeze(-1),
                lookback_ratios["new_tokens"][
                    :, :, : self.classifier_max_sequence_length
                ].unsqueeze(-1),
            ],
            dim=-1,
        )
        sample = sample.numpy()

        if (
            np.isnan(sample[:, :, 0, :]).all()
            and np.isnan(sample[:, :, 0, :]).all()
            and np.isnan(sample[:, :, 0, :]).all()
            and np.isnan(sample[:, :, 0, :]).all()
        ):
            sample = sample[:, :, 1:, :]

        # Padding
        height, width, sequence_length, features = sample.shape
        if sequence_length < self.classifier_max_sequence_length:
            padding = np.zeros(
                (
                    height,
                    width,
                    self.classifier_max_sequence_length - sequence_length,
                    features,
                )
            )
            sample = np.concatenate((sample, padding), axis=2)
        elif sequence_length > self.classifier_max_sequence_length:
            sample = sample[:, :, : self.classifier_max_sequence_length, :]

        # Flatten
        sample = sample.reshape(-1)

        return np.array([sample])

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
        tokenised_inputs = self._verbalise_input(prompt).to(self.model.device)

        # Calculate the length of each component
        component_lengths = self._get_component_lengths(inputs, tokenised_inputs)

        # Predict
        with torch.inference_mode():
            input_logits = self.model(
                input_ids=tokenised_inputs[:, :-1], use_cache=True, return_dict=True
            )
            generated_ids = []
            generation_start_id = tokenised_inputs.size(1)
            last_input_token = tokenised_inputs[:, -1]
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

                hallucinated_past_kvs = [hallucinated_past_kv]
                hallucinated_logits = []
                hallucinated_attention_maps = []
                hallucinated_last_input_token = last_input_token
                for _ in range(self.classifier_max_sequence_length):
                    hallucinated_output = self.model(
                        input_ids=hallucinated_last_input_token,
                        past_key_values=hallucinated_past_kvs[-1],
                        use_cache=True,
                        output_attentions=True,
                        attn_mode="torch",
                        block_list=self.retrieval_heads,
                    )
                    hallucinated_past_kvs += [hallucinated_output.past_key_values]
                    hallucinated_logits += [hallucinated_output.logits]
                    hallucinated_attention_maps += [hallucinated_output.attentions]
                    hallucinated_last_input_token = hallucinated_output.logits[
                        0, -1
                    ].argmax()

                lookback_ratios = self.get_lookback_ratios(
                    hallucinated_attention_maps, component_lengths, generation_start_id
                )
                lookback_ratios = self._prepare_lookback_ratios(lookback_ratios)
                # Get index 0 which means most likely to be incorrect
                alpha = self.classifier.predict_proba(lookback_ratios)[0, 0]

                next_token_logits = (1 + alpha) * base_outputs.logits[
                    0, -1
                ] - alpha * hallucinated_logits[0].logits[0, -1]

                base_past_kv = base_outputs.past_key_values
                hallucinated_past_kv = hallucinated_past_kvs[0]

                last_input_token = next_token_logits.argmax()
                generated_ids.append(last_input_token.item())
                alphas.append(alpha)
                if last_input_token.item() == self.tokenizer.eos_token_id:
                    break
            decoded_text = self.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )

        return {"decoded_text": decoded_text, "alphas": alphas, "attentions": {}}

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

            # base_logits = base_logits.log_softmax(dim=-1)
            # hallucinated_logits = hallucinated_logits.log_softmax(dim=-1)
            diff_logits = (
                (1 + self.decoder_configs.configs.alpha) * base_logits
                - self.decoder_configs.configs.alpha * hallucinated_logits
            )

            diff_logits = diff_logits.log_softmax(dim=-1)

            log_probs = (
                diff_logits[range(diff_logits.shape[0]), continue_ids].sum().item()
            )

        return log_probs
