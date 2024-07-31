from typing import List, Optional, Tuple, Union

import torch
import numpy as np
import copy

from src.configs import DecoderConfigs, ModelConfigs
from src.models.base_model import BaseModel


class BaselineGuided(BaseModel):
    def __init__(
        self,
        model_configs: ModelConfigs,
        decoder_configs: DecoderConfigs,
    ):
        super().__init__(model_configs, decoder_configs)

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

        return sample

    def generate(
        self,
        inputs,
        return_attentions: bool = False,
    ) -> dict:
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
            attentions = []
            generation_start_id = tokenised_inputs.size(1)
            last_input_token = tokenised_inputs[:, -1]
            past_kv = input_logits.past_key_values

            while len(generated_ids) < self.max_new_tokens:
                sample_hallucination_probas = []
                sample_outputs = []
                sample_past_kv = copy.deepcopy(past_kv)
                for _ in range(self.classifier_num_samples):
                    window_attention_maps = []
                    temp_outputs = []
                    for _ in range(self.classifier_max_sequence_length):
                        last_input_token = last_input_token.view(1, 1)
                        outputs = self.model(
                            input_ids=last_input_token,
                            past_key_values=sample_past_kv,
                            use_cache=True,
                            output_attentions=True,
                            attn_mode="torch",
                        )
                        sample_past_kv = outputs.past_key_values
                        last_input_token = outputs.logits[0, -1].argmax()

                        window_attention_maps += [outputs.attentions]
                        temp_outputs += [outputs]

                        if last_input_token.item() == self.tokenizer.eos_token_id:
                            break

                    lookback_ratios = self.get_lookback_ratios(
                        window_attention_maps, component_lengths, generation_start_id
                    )
                    lookback_ratios = self._prepare_lookback_ratios(lookback_ratios)
                    print("lookback_ratios: ", lookback_ratios)
                    print("lookback_ratios.shape: ", lookback_ratios.shape)
                    sample_hallucination_probas += [
                        self.classifier.predict_proba(lookback_ratios)
                    ]
                    sample_outputs += [temp_outputs]

                # Get the argmin hallucination probabilities
                sample_hallucination_probas = np.array(sample_hallucination_probas)
                min_hallucination_idx = np.argmin(sample_hallucination_probas)

                final_outputs = sample_outputs[min_hallucination_idx]

                past_kv = final_outputs[-1].past_key_values
                for final_output in final_outputs:
                    attentions += [final_output.attentions]
                    last_input_token = final_output.logits[0, -1].argmax()
                    generated_ids.append(last_input_token.item())
                    if last_input_token.item() == self.tokenizer.eos_token_id:
                        break
            decoded_text = self.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )

        generation_output = {"decoded_text": decoded_text, "attentions": {}}
        if return_attentions:
            generation_output["attentions"] = self.get_lookback_ratios(
                attentions, component_lengths, generation_start_id
            )

        return generation_output

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

            outputs = self.model(input_ids)[0].squeeze(0)
            outputs = outputs.log_softmax(-1)  # logits to log probs

            # skip tokens in the prompt -- we only care about the answer
            outputs = outputs[prefix_ids.shape[-1] - 1 : -1, :]

            # get logprobs for each token in the answer
            log_probs = outputs[range(outputs.shape[0]), continue_ids].sum().item()

        return log_probs
