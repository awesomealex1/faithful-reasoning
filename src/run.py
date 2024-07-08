import gzip
import json
import math
import os

import huggingface_hub
import hydra
import pandas as pd
import torch
from accelerate import Accelerator
from accelerate.tracking import WandBTracker
from huggingface_hub import HfApi
from omegaconf import OmegaConf
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from src.configs import RunnerConfigs
from src.factories import get_dataset, get_model, get_metrics


class Run:
    def __init__(self, configs: RunnerConfigs):
        self.configs = configs

        self.hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        self.output_dir = self.hydra_cfg["runtime"]["output_dir"]

        self._load_dataloaders()
        self._load_pipeline()
        self._load_accelerator()

        if not configs.debug:
            self._setup_run()

    def _load_dataloaders(self) -> None:
        self.dataloaders = {}

        dataset = get_dataset(
            self.configs.data,
        )
        self.dataloaders = DataLoader(
            dataset,
            shuffle=True,
            **self.configs.data_loader,
        )

    def _load_pipeline(self) -> None:
        self.model = get_model(self.configs.model, self.configs.decoder)

    def _load_accelerator(self) -> None:
        self.accelerator = Accelerator(log_with="wandb")
        (self.model, self.dataloaders) = self.accelerator.prepare(
            self.model, self.dataloaders
        )

    def _load_metrics(self):
        self.metrics = get_metrics(self.configs.data)

    def _setup_run(self):

        # Naming by model name
        self.wandb_run_name = self.configs.model.name

        self.wandb_tracker = None
        if self.accelerator:
            if self.accelerator.is_main_process:
                self.accelerator.init_trackers(
                    project_name=self.configs.wandb_project,
                    init_kwargs={
                        "wandb": {
                            "entity": self.configs.wandb_entity,
                            "name": self.wandb_run_name,
                        }
                    },
                )
                self.wandb_tracker: WandBTracker = self.accelerator.get_tracker("wandb")
                self.accelerator.wait_for_everyone()
        else:
            wandb.init(
                project=self.configs.wandb_project,
                entity=self.configs.wandb_entity,
                name=self.wandb_run_name,
            )

    def train(self):
        pass

    def test(self):
        predictions = []

        prediction_filepath = os.path.join(
            self.output_dir, f"pred_{self.configs.data.name}.csv"
        )

        for step, batch in enumerate(tqdm(self.dataloaders)):
            # Predict
            prediction = self.model.generate(batch["prompted_question"][0])

            if self.configs.data.name == "TruthfulQA":
                scores_true = []
                scores_false = []
                for temp_ans in batch["prompted_ref_true"]:
                    # append the current answer choice to the prompt
                    log_probs = self.model.lm_score(
                        batch["prompted_question"][0], temp_ans[0]
                    )
                    print("prompted_ref_true")
                    print(log_probs)
                    scores_true.append(log_probs)

                for temp_ans in batch["prompted_ref_false"]:
                    # append the current answer choice to the prompt
                    log_probs = self.model.lm_score(
                        batch["prompted_question"][0], temp_ans[0]
                    )
                    print("prompted_ref_false")
                    print(log_probs)
                    scores_false.append(log_probs)

            batch["predicted_answer"] = prediction
            batch["scores_true"] = scores_true
            batch["scores_false"] = scores_false

            predictions.append(batch)

            print(batch)
            for key, value in batch.items():
                print(key, ":", value, type(value))

            # Save the predictions to a JSONL file after each batch
            with open(prediction_filepath, "a") as f:
                f.write(json.dumps(batch) + "\n")

            # if step >= 10:
            #     break

        # Evaluate
        metrics = self.metrics(predictions)

        # Log
        print(metrics)
        if self.accelerator:
            self.accelerator.log(metrics)
        else:
            wandb.log(metrics)

        # Save predictions as artifacts
        if self.accelerator:
            self.accelerator.save_artifacts(prediction_filepath)
        else:
            wandb.save(prediction_filepath)
