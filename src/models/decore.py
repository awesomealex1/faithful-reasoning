from typing import List, Optional, Tuple

import os
import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.configs import DecoderConfigs, ModelConfigs

from src.models.base_model import BaseModel


class DeCoRe(BaseModel):
    def __init__(
        self,
        model_configs: ModelConfigs,
        decoder_configs: DecoderConfigs,
    ):
        super().__init__(model_configs, decoder_configs)

        model_base_name = model_configs.configs.model_name_or_path.split("/")[1]

        with open(
            os.path.join(
                decoder_configs.configs.retrieval_heads_dir, f"{model_base_name}.json"
            )
        ) as file:
            head_list = json.loads(file.readline())

        ## use the average retrieval score and ranking
        head_score_list = [
            ([int(ll) for ll in l[0].split("-")], np.mean(l[1]))
            for l in head_list.items()
        ]
        head_score_list = sorted(head_score_list, key=lambda x: x[1], reverse=True)
        top_retrieval_heads = [
            [l[0], round(np.mean(l[1]), 2)] for l in head_score_list
        ][:10]
        print(top_retrieval_heads)

    def generate(
        self,
        inputs,
    ) -> str:
        self.model.eval()

        lm_input = self._verbalise_task_input(inputs)
        hallucinated_lm_input = self._verbalise_task_input(
            inputs, self.teacher_tokenizer
        )

        slm_input = slm_input.to(self.model.device)
        tlm_input = tlm_input.to(self.model.device)

        # Predict
        with torch.inference_mode():
            generated_tokens = torch.tensor(
                [[]], dtype=torch.long, device=self.model.device
            )
            for forward_pass in range(self.max_new_tokens):
                slm_output = self.model.generate(
                    input_ids=torch.cat([slm_input, generated_tokens], dim=1),
                    max_new_tokens=1,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                slm_score = slm_output["scores"][0][0].cpu()

                # Forward pass
                tlm_output = self.model.generate(
                    input_ids=torch.cat([tlm_input, generated_tokens], dim=1),
                    max_new_tokens=1,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                tlm_score = tlm_output["scores"][0][0].cpu()

                # Aggregate teacher LM and student LM scores

                next_token_score = (
                    1 + self.decoder_configs.configs.alpha
                ) * tlm_score - self.decoder_configs.configs.alpha * slm_score

                _, indices = torch.topk(next_token_score, 1)
                generated_tokens = torch.cat(
                    [generated_tokens, indices.unsqueeze(dim=0).to(self.model.device)],
                    dim=1,
                )

            decoded_text = self.tokenizer.decode(
                generated_tokens[0], skip_special_tokens=True
            )

        return decoded_text
