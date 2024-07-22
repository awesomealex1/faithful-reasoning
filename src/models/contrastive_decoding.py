from typing import List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.configs import DecoderConfigs, ModelConfigs

from src.models.base_model import BaseModel


class ContrastiveDecoding(BaseModel):
    def __init__(
        self,
        model_configs: ModelConfigs,
        decoder_configs: DecoderConfigs,
    ):
        super().__init__(model_configs, decoder_configs)

        self.teacher_model = AutoModelForCausalLM.from_pretrained(
            decoder_configs.configs.teacher_model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        self.teacher_tokenizer = AutoTokenizer.from_pretrained(
            decoder_configs.configs.teacher_model_name_or_path
        )

    def generate(
        self,
        inputs,
    ) -> dict:
        self.model.eval()

        slm_input = self._verbalise_task_input(inputs)
        tlm_input = self._verbalise_task_input(inputs, self.teacher_tokenizer)

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

                if indices == self.tokenizer.eos_token_id:
                    break

            decoded_text = self.tokenizer.decode(
                generated_tokens[0], skip_special_tokens=True
            )

        return {"decoded_text": decoded_text}
