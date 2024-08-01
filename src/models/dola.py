from typing import List, Optional, Tuple, Union

import torch
from torch.nn import functional as F

from src.configs import DecoderConfigs, ModelConfigs
from src.models.base_model import BaseModel


class DoLa(BaseModel):
    def __init__(
        self,
        model_configs: ModelConfigs,
        decoder_configs: DecoderConfigs,
    ):
        super().__init__(model_configs, decoder_configs)

        self.dola_layers = self.decoder_configs.configs.dola_layers

        self.post_softmax = self.decoder_configs.configs.post_softmax

        if self.dola_layers == "low":
            self.candidate_premature_layers = list(range(0, 16, 2)) + [
                32
            ]  # FIXME: 16,32 is hard-coded for llama3-8b
        elif self.dola_layers == "high":
            self.candidate_premature_layers = list(range(16, 32, 2)) + [
                32
            ]  # FIXME: 16,32 is hard-coded for llama3-8b
        self.mature_layer = self.candidate_premature_layers[-1]

    def generate(
        self,
        inputs,
        return_attentions: bool = False,
    ) -> dict:
        self.model.eval()

        prompt = inputs["prompted_question"][0]
        tokenised_inputs = self._verbalise_input(prompt).to(self.model.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                tokenised_inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                dola_layers=self.dola_layers,
            )
            decoded_text = self.tokenizer.decode(
                outputs[0, tokenised_inputs.size(1) :], skip_special_tokens=True
            )
        return {"decoded_text": decoded_text, "attentions": {}}

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

            premature_layer_dist = {l: 0 for l in self.candidate_premature_layers}
            picked_logits = []
            result_dict = {}
            premature_layers = []

            dict_outputs, outputs = self.model(
                input_ids=input_ids,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
                early_exit_layers=self.candidate_premature_layers + [self.mature_layer],
            )

            for seq_i in range(prefix_ids.shape[-1] - 1, input_ids.shape[-1] - 1):
                # Pick the less like layer to contrast with
                # 1. Stacking all premature_layers into a new dimension
                stacked_premature_layers = torch.stack(
                    [
                        dict_outputs[i][:, seq_i, :]
                        for i in self.candidate_premature_layers
                    ],
                    dim=0,
                )

                # 2. Calculate the softmax values for mature_layer and all premature_layers
                softmax_mature_layer = F.softmax(
                    dict_outputs[self.mature_layer][:, seq_i, :], dim=-1
                )  # shape: (batch_size, num_features)
                softmax_premature_layers = F.softmax(
                    stacked_premature_layers, dim=-1
                )  # shape: (num_premature_layers, batch_size, num_features)

                # 3. Calculate M, the average distribution
                M = 0.5 * (
                    softmax_mature_layer[None, :, :] + softmax_premature_layers
                )  # shape: (num_premature_layers, batch_size, num_features)

                # 4. Calculate log-softmax for the KL divergence
                log_softmax_mature_layer = F.log_softmax(
                    dict_outputs[self.mature_layer][:, seq_i, :], dim=-1
                )  # shape: (batch_size, num_features)
                log_softmax_premature_layers = F.log_softmax(
                    stacked_premature_layers, dim=-1
                )  # shape: (num_premature_layers, batch_size, num_features)

                # 5. Calculate the KL divergences and then the JS divergences
                kl1 = F.kl_div(
                    log_softmax_mature_layer[None, :, :], M, reduction="none"
                ).mean(
                    -1
                )  # shape: (num_premature_layers, batch_size)
                kl2 = F.kl_div(log_softmax_premature_layers, M, reduction="none").mean(
                    -1
                )  # shape: (num_premature_layers, batch_size)
                js_divs = 0.5 * (kl1 + kl2)  # shape: (num_premature_layers, batch_size)

                # 6. Reduce the batchmean
                js_divs = js_divs.mean(-1)  # shape: (num_premature_layers,)
                premature_layer = self.candidate_premature_layers[
                    int(js_divs.argmax().cpu().item())
                ]
                premature_layer_dist[premature_layer] += 1

                premature_layers.append(premature_layer)

            base_logits = torch.zeros_like(
                dict_outputs[self.mature_layer][0, prefix_ids.shape[-1] - 1 : -1]
            )
            for i, l in enumerate(premature_layers):
                base_logits[i] = dict_outputs[l][0, prefix_ids.shape[-1] - 1 + i]
            final_logits = dict_outputs[self.mature_layer][
                0, prefix_ids.shape[-1] - 1 : -1
            ]
            final_logits = final_logits.log_softmax(dim=-1)
            base_logits = base_logits.log_softmax(dim=-1)
            diff_logits = final_logits - base_logits
            if self.post_softmax:
                diff_logits = diff_logits.log_softmax(dim=-1)

            log_probs = (
                diff_logits[range(diff_logits.shape[0]), continue_ids].sum().item()
            )

        return log_probs
