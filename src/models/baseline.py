from typing import List, Optional, Tuple

import torch

from src.configs import DecoderConfigs, ModelConfigs
from src.models.base_model import BaseModel


class Baseline(BaseModel):
    def __init__(
        self,
        model_configs: ModelConfigs,
        decoder_configs: DecoderConfigs,
    ):
        super().__init__(model_configs, decoder_configs)

    def generate(
        self,
        inputs,
    ) -> str:
        self.model.eval()

        print(inputs)
        if self.model_configs.model_type == "instruct":
            inputs = [
                p[0] for p in inputs
            ]  # Quirky data loader behaviour to make things as tuple
        print(inputs)

        inputs = self._verbalise_input(inputs).to(self.model.device)

        # Predict
        with torch.inference_mode():
            output = self.model.generate(
                inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
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
            if self.model_configs.model_type == "instruct":
                input_text = [p[0] for p in prompt] + [answer]
            elif self.model_configs.model_type == "base":
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
