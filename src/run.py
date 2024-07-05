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
from src.factories import get_dataset, get_decoder
from src.model import HFModel


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
        self.model = HFModel(
            self.configs.model, self.configs.prompt
        )
        self.decoder = get_decoder(self.model, self.configs.decoder)

    def _load_accelerator(self) -> None:
        self.accelerator = Accelerator(log_with="wandb")
        (
            self.model,
            self.dataloaders
        ) = self.accelerator.prepare(
            self.model,
            self.dataloaders
        )

    @staticmethod
    def compute_metrics(predictions: pd.DataFrame, split: str):
        groundtruth_answers = predictions["groundtruth_answers"]
        predicted_answers = predictions["predicted_answers"]

        total_em = 0
        for predicted_answer, groundtruth_answer in zip(
            predicted_answers, groundtruth_answers
        ):
            total_em += best_subspan_em(predicted_answer, groundtruth_answer)

        return {f"{split}/em": total_em / len(predicted_answers)}

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

    def test(self, log_metrics: bool = True):
        predictions_df = pd.DataFrame(
            columns=[
                "contexts",
                "question",
                "groundtruth_answers",
                "predicted_answers",
            ]
        )

        for step, batch in enumerate(tqdm(self.dataloaders)):
            # Predict
            prediction = self.pipeline.generate(batch)

            batch_df = pd.DataFrame(
                {
                    "contexts": [batch["contexts"]],
                    "question": [batch["question"]],
                    "groundtruth_answers": [[answer[0] for answer in batch["answers"]]],
                    "predicted_answers": [prediction],
                }
            )

            # Append the batch DataFrame to the overall predictions DataFrame
            predictions_df = pd.concat([predictions_df, batch_df], ignore_index=True)

            # Save the updated DataFrame to a CSV file after each batch
            predictions_df.to_csv(
                os.path.join(self.output_dir, f"predictions_{split}.csv"), index=False
            )

            # if step >= 10:
            #     break

        # Evaluate
        metrics = self.compute_metrics(predictions_df, split=split)

        # Log
        print(metrics)
        if self.accelerator:
            self.accelerator.log(
                metrics
                | {f"{split}_prediction_df": wandb.Table(dataframe=predictions_df)}
            )
        else:
            wandb.log(
                metrics
                | {f"{split}_prediction_df": wandb.Table(dataframe=predictions_df)}
            )
