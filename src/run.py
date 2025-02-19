import gzip
import json
import math
import os

import huggingface_hub
import hydra
import pandas as pd
import torch
import wandb
from huggingface_hub import HfApi
from omegaconf import OmegaConf
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.configs import RunnerConfigs
from src.factories import get_dataset, get_metrics, get_model, get_framework
from transformers import DataCollatorForLanguageModeling, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig


class Run:
    def __init__(self, configs: RunnerConfigs):
        self.configs = configs

        self.hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        self.output_dir = self.hydra_cfg["runtime"]["output_dir"]

        self._load_dataloaders()
        self._load_pipeline()
        self._load_metrics()

        self._setup_run()

    def _load_dataloaders(self) -> None:
        self.dataloaders = {}

        dataset = get_dataset(
            self.configs.data,
            use_chat_template=self.configs.model.model_type == "instruct",
        )
        print(dataset)
        print(self.configs.data)
        self.dataloaders = DataLoader(
            dataset,
            shuffle=False,
            **self.configs.data_loader,
        )

    def _load_pipeline(self) -> None:
        self.model = get_model(self.configs.model, self.configs.decoder)

        if self.configs.framework:
            self.model = get_framework(self.configs.framework, self.configs.data, self.model)

    def _load_metrics(self):
        self.metrics = get_metrics(self.configs.data, self.configs.framework)

    def _setup_run(self):
        # Naming by model name
        self.run_name = f"{self.configs.model.name}__{self.configs.decoder.name}"

        if self.configs.framework:
            self.run_name += f"__{self.configs.framework.name}"

        if not self.configs.debug:
            self.group_name = self.configs.data.name
            wandb.init(
                project=self.configs.wandb_project,
                entity=self.configs.wandb_entity,
                name=self.run_name,
                group=self.group_name,
                config=OmegaConf.to_container(self.configs),
            )

    def test(self):
        """
        Test the model on the dataset and log the predictions and metrics to WandB
        """
        predictions = []

        prediction_filename = f"pred_{self.configs.data.name}_{self.run_name}"

        prediction_filepath = os.path.join(
            self.output_dir, f"{prediction_filename}.json"
        )
        attentions_filepath = os.path.join(
            self.output_dir, f"att_{self.configs.data.name}.pt"
        )

        # To save WandB space, just return attentions for the Baseline model
        # Mainly for Logistic Regression purposes

        for step, batch in enumerate(tqdm(self.dataloaders)):
            # Predict
            prediction = self.model.generate(batch)
            batch["predicted_answer"] = prediction["decoded_text"]
            if "alphas" in prediction:
                # Handle for DeCoRe guided, to analyse the changes in alpha value throughout generation steps
                batch["alphas"] = prediction["alphas"]

            if self.configs.data.name in ["TruthfulQA", "MemoTrap"]:
                scores_true = []
                scores_false = []
                for temp_ans in batch["prompted_ref_true"]:
                    ans = temp_ans[0] if type(temp_ans) in [list, tuple] else temp_ans
                    log_probs = self.model.lm_score(batch, ans)
                    scores_true.append(log_probs)

                for temp_ans in batch["prompted_ref_false"]:
                    ans = temp_ans[0] if type(temp_ans) in [list, tuple] else temp_ans
                    log_probs = self.model.lm_score(batch, ans)
                    scores_false.append(log_probs)

                batch["scores_true"] = scores_true
                batch["scores_false"] = scores_false

            if self.configs.data.name == "MemoTrap":
                batch["answer_index"] = int(batch["answer_index"].cpu().numpy()[0])

            # Brute force normalisation for IFEval, some values were casted as tensors by collator
            if self.configs.data.name == "IFEval":
                batch["kwargs"] = [
                    {
                        k: int(v.cpu().numpy()[0]) if type(v) == torch.Tensor else v
                        for k, v in kwargs_.items()
                    }
                    for kwargs_ in batch["kwargs"]
                ]

            predictions.append(batch)

            values_to_normalised = ["idx"]
            if self.configs.data.name == "PopQA":
                values_to_normalised += [
                    "s_pop",
                    "o_pop",
                ]
            for key in values_to_normalised:
                try:
                    batch[key] = int(batch[key].cpu().numpy()[0])
                except:
                    batch[key] = str(batch[key][0])

            # Save the predictions to a JSONL file after each batch
            with open(prediction_filepath, "a") as f:
                f.write(json.dumps(batch) + "\n")

        # Evaluate
        metrics = self.metrics(predictions)

        # Log
        if not self.configs.debug:
            wandb.log(metrics)

            pred_artifact = wandb.Artifact(prediction_filename, type="prediction")
            pred_artifact.add_file(prediction_filepath)
            wandb.log_artifact(pred_artifact)
        else:
            print(metrics)
    
    def finetune(self):
        sft_config = SFTConfig(
            run_name='test-run-llama',
            output_dir="test-run-llama-checkpoints",
            report_to="wandb",
            logging_steps=25,
            dataset_text_field='text',
        )
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
        )

        if self.model.model.model:
            model = self.model.model.model  # Framework -> Decoder -> Model
            tokenizer = self.model.model.tokenizer
        else:
            model = self.model.model    # Decoder -> Model
            tokenizer = self.model.tokenizer
        model.train()

        model = get_peft_model(model, peft_config)

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # Set to False for causal language modeling
        )

        prompt = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
        (1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
        (2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
        (3) Finish[answer], which returns the answer and finishes the task."""

        def prompt_format(example):
            example['text'] = f"{prompt}\nQuestion: {example['question'].strip()}\n{example['trajectory']}"
            return example
        
        dataset = load_dataset("xz56/react-llama")['train']
        dataset = dataset.remove_columns(['correct_answer', 'id'])
        dataset = dataset.map(prompt_format)

        def tokenize_function(example):
            return tokenizer(example['text'], truncation=True, padding='max_length', max_length=512)
        tokenized_dataset = dataset.map(tokenize_function, batched=True)

        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            peft_config=peft_config,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )

        trainer.train()

