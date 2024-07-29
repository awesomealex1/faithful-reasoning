import argparse
import logging
import os
import sys

sys.path.append(os.path.dirname(os.getcwd()))
import torch

import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.linear_model import (
    LogisticRegression,
    LogisticRegressionCV,
    ElasticNet,
    ElasticNetCV,
)

from torch.utils.data import DataLoader, Dataset
import numpy as np
from transformers import AutoTokenizer

from src.metrics.nq import best_subspan_em


class AttentionMapDataset(Dataset):
    def __init__(
        self,
        data: list,
        max_sequence_length: int,
        model_name_or_path: str = "meta-llama/Meta-Llama-3-8b",
    ):
        self.data = data

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        newline_token_ids = [
            self.tokenizer("\n")["input_ids"][-1],
            self.tokenizer("\n\n")["input_ids"][-1],
        ]
        self.finish_tokens = [
            self.tokenizer.pad_token_id,
            self.tokenizer.eos_token_id,
        ] + newline_token_ids

        self.max_sequence_length = max_sequence_length
        self.x, self.x_ori_lengths = self.pad_or_truncate_channels(
            [x["attentions"] for x in self.data],
            [x["predicted_answer"] for x in self.data],
        )
        self.y = self.get_label_em()

    def get_label_em(self):
        labels = []
        for sample in self.data:
            labels += [
                best_subspan_em(
                    sample["predicted_answer"], [ans[0] for ans in sample["answers"]]
                )
            ]
        return labels

    def pad_or_truncate_channels(self, attentions, answers):
        data = []
        lengths = []

        tokenized_answers = self.tokenizer(answers)["input_ids"]
        for att, tokens in zip(attentions, tokenized_answers):
            length = 0
            for token in tokens:
                if token in self.finish_tokens:
                    break
                else:
                    length += 1

            sample = torch.cat(
                [
                    att["bos"][:, :, :length].unsqueeze(-1),
                    att["instruction"][:, :, :length].unsqueeze(-1),
                    att["icl_demo"][:, :, :length].unsqueeze(-1),
                    att["contexts"][:, :, :length].unsqueeze(-1),
                    att["question"][:, :, :length].unsqueeze(-1),
                    att["answer_prefix"][:, :, :length].unsqueeze(-1),
                    att["new_tokens"][:, :, :length].unsqueeze(-1),
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

            height, width, sequence_length, features = sample.shape
            if sequence_length < self.max_sequence_length:
                padding = np.zeros(
                    (
                        height,
                        width,
                        self.max_sequence_length - sequence_length,
                        features,
                    )
                )
                sample = np.concatenate((sample, padding), axis=2)
            elif sequence_length > self.max_sequence_length:
                sample = sample[:, :, : self.max_sequence_length, :]
            data.append(sample)
            lengths.append(length)
        data = np.array(data)

        return data, lengths

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.x[idx]  # .transpose((2, 0, 1))
        y = self.y[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(
            y, dtype=torch.float32
        )


def plot_sequence_length(
    dataset: AttentionMapDataset, split: str, outputs_dir: str = "./outputs/lr_analyses"
):
    min_ori_length = min(dataset.x_ori_lengths)
    max_ori_length = max(dataset.x_ori_lengths)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    sns.histplot(dataset.x_ori_lengths, discrete=True, ax=ax)
    ax.set_title(
        f"{split.capitalize()} Num Tokens: Min={min_ori_length}, Max={max_ori_length}"
    )

    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, "sequence_length.png"))


def print_label_distribution(train_dataset, test_dataset):
    train_pos = sum(train_dataset.y)
    train_neg = len(train_dataset.y) - sum(train_dataset.y)
    test_pos = sum(test_dataset.y)
    test_neg = len(test_dataset.y) - sum(test_dataset.y)

    print(f"Train Pos: {train_pos}, Train Neg: {train_neg}")
    print(f"Test Pos: {test_pos}, Test Neg: {test_neg}")


FEATURES_LIST = [
    "BOS",
    "Instruction",
    "ICL Demonstration",
    "Contexts",
    "Question",
    "Answer Prefix",
    "New Tokens",
]


def print_metrics(model, dataset, split):
    flattened_x = dataset.x.reshape(dataset.x.shape[0], -1)
    y_pred = model.predict_proba(flattened_x)
    precision = precision_score(
        dataset.y, [1 if pred[1] >= 0.5 else 0 for pred in y_pred]
    )
    recall = recall_score(dataset.y, [1 if pred[1] >= 0.5 else 0 for pred in y_pred])
    f1 = f1_score(dataset.y, [1 if pred[1] >= 0.5 else 0 for pred in y_pred])
    auroc = roc_auc_score(dataset.y, y_pred[:, 1])
    auprc = average_precision_score(dataset.y, y_pred[:, 1])
    print(
        f"{split.capitalize()}:\nPrecision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}"
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auroc,
        "auprc": auprc,
    }


def fit_logreg_all_features(
    train_dataset,
    test_dataset,
    max_sequence_length,
    regulariser,
    C=1.0,
    outputs_dir="./outputs/lr_analyses",
):
    assert len(FEATURES_LIST) == train_dataset.x.shape[-1]
    assert len(FEATURES_LIST) == test_dataset.x.shape[-1]

    num_train_data, num_layers, num_heads, max_sequence_length, num_features = (
        train_dataset.x.shape
    )

    flattened_train_x = train_dataset.x.reshape(num_train_data, -1)
    print(flattened_train_x.shape)
    flattened_test_x = test_dataset.x.reshape(test_dataset.x.shape[0], -1)
    print(flattened_test_x.shape)

    if regulariser == "l1":
        model = LogisticRegressionCV(
            penalty="l1",
            solver="liblinear",
            cv=5,
            class_weight="balanced",
            random_state=1234,
            n_jobs=-1,
        )
    elif regulariser == "l2":
        model = LogisticRegressionCV(
            penalty="l2",
            solver="liblinear",
            cv=5,
            class_weight="balanced",
            random_state=1234,
            n_jobs=-1,
        )
    elif regulariser == "elastic":
        model = LogisticRegressionCV(
            penalty="elasticnet",
            solver="saga",
            cv=5,
            class_weight="balanced",
            random_state=1234,
            l1_ratios=[0.5] * 5,
            n_jobs=-1,
        )

    model.fit(flattened_train_x, train_dataset.y)

    train_metrics = print_metrics(model, train_dataset, "train")
    test_metrics = print_metrics(model, test_dataset, "test")

    # Reconstruct the model weights
    coef = model.coef_[0].reshape(
        num_layers, num_heads, max_sequence_length, num_features
    )

    # Heatmap for each sequence position and feature
    for seq_id in range(max_sequence_length):
        fig, axs = plt.subplots(
            1, len(FEATURES_LIST), figsize=(len(FEATURES_LIST) * 4, 4)
        )  # Adjusted figsize for better display
        for feature_id, feature_name in enumerate(FEATURES_LIST):
            axs[feature_id].set_title(f"Position: {seq_id}; Feature: {feature_name}")
            sns.heatmap(
                coef[:, :, seq_id, feature_id],
                ax=axs[feature_id],
                center=0,
                cmap="coolwarm",
                cbar=True,
            )
            axs[feature_id].set_xlabel("Head")
            axs[feature_id].set_ylabel("Layer")
        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.show()
    # Save the plot
    plt.savefig(
        os.path.join(
            outputs_dir, f"coef__lr_{regulariser}__seqlen_{max_sequence_length}.png"
        )
    )

    return model, train_metrics, test_metrics


def fit_logreg_per_feature(
    train_dataset, test_dataset, max_sequence_length, regulariser, C=1.0
):
    assert len(FEATURES_LIST) == train_dataset.x.shape[-1]
    assert len(FEATURES_LIST) == test_dataset.x.shape[-1]

    all_models = {}
    all_train_metrics = {}
    all_test_metrics = {}

    for idx, feature in enumerate(FEATURES_LIST):
        print(f"Training logitic regression with {feature} feature")
        train_feature_dataset = train_dataset.x[:, :, :, :, idx]
        test_feature_dataset = test_dataset.x[:, :, :, :, idx]
        num_train_data, num_layers, num_heads, max_sequence_length = (
            train_feature_dataset.shape
        )

        flattened_train_x = train_feature_dataset.reshape(num_train_data, -1)
        print(flattened_train_x.shape)

        flattened_test_x = test_feature_dataset.reshape(
            test_feature_dataset.shape[0], -1
        )
        print(flattened_test_x.shape)

        if regulariser == "l1":
            model = LogisticRegressionCV(
                penalty="l1",
                solver="liblinear",
                cv=5,
                class_weight="balanced",
                random_state=1234,
                n_jobs=-1,
            )
        elif regulariser == "l2":
            model = LogisticRegressionCV(
                penalty="l2",
                solver="liblinear",
                cv=5,
                class_weight="balanced",
                random_state=1234,
                n_jobs=-1,
            )
        elif regulariser == "elastic":
            model = LogisticRegressionCV(
                penalty="elasticnet",
                solver="saga",
                cv=5,
                class_weight="balanced",
                random_state=1234,
                l1_ratios=[0.5] * 5,
                n_jobs=-1,
            )

        model.fit(flattened_train_x, train_dataset.y)

        train_metrics = print_metrics(model, train_feature_dataset, "train")
        test_metrics = print_metrics(model, test_feature_dataset, "test")

        # Reconstruct the model weights
        coef = model.coef_[0].reshape(num_layers, num_heads, max_sequence_length)

        # Heatmap for each sequence position and feature
        for seq_id in range(max_sequence_length):
            fig, axs = plt.subplots(
                1, len(FEATURES_LIST), figsize=(len(FEATURES_LIST) * 4, 4)
            )  # Adjusted figsize for better display
            for feature_id, feature_name in enumerate(FEATURES_LIST):
                axs[feature_id].set_title(
                    f"Position: {seq_id}; Feature: {feature_name}"
                )
                sns.heatmap(
                    coef[:, :, seq_id, feature_id],
                    ax=axs[feature_id],
                    center=0,
                    cmap="coolwarm",
                    cbar=True,
                )
                axs[feature_id].set_xlabel("Head")
                axs[feature_id].set_ylabel("Layer")
            plt.tight_layout()  # Adjust layout to prevent overlap
            plt.show()

        all_models[feature] = model
        all_train_metrics[feature] = train_metrics
        all_test_metrics[feature] = test_metrics

    return all_models, all_train_metrics, all_test_metrics


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../attention_maps")
    parser.add_argument("--outputs_dir", type=str, default="./outputs/lr_analyses")
    parser.add_argument("--max_sequence_length", type=int, default=4)
    parser.add_argument("--regulariser", type=str, default="l1")
    return parser.parse_args()


def main():
    args = argument_parser()

    outputs_dir = args.outputs_dir
    os.makedirs(outputs_dir, exist_ok=True)
    data_dir = args.data_dir

    oracle_case_filepath = "att_NQ_oracle_llama_3_8b_instruct.pt"

    litm_case_filepaths = {
        "gold_at_0": "att_NQ_gold_at_0_llama_3_8b_instruct.pt",
        "gold_at_4": "att_NQ_gold_at_4_llama_3_8b_instruct.pt",
        "gold_at_9": "att_NQ_gold_at_9_llama_3_8b_instruct.pt",
    }

    oracle_samples = torch.load(
        os.path.join(data_dir, oracle_case_filepath), map_location=torch.device("cpu")
    )

    for i, sample in enumerate(oracle_samples):
        sample["em"] = best_subspan_em(
            sample["predicted_answer"], [ans[0] for ans in sample["answers"]]
        )

    litm_correct__oracle_correct = {}
    litm_correct__oracle_incorrect = {}
    litm_incorrect__oracle_correct = {}
    litm_incorrect__oracle_incorrect = {}

    for litm_case_name, litm_case_filepath in litm_case_filepaths.items():
        litm_samples = torch.load(
            os.path.join(data_dir, litm_case_filepath), map_location=torch.device("cpu")
        )

        split_litm_correct__oracle_correct = []
        split_litm_correct__oracle_incorrect = []
        split_litm_incorrect__oracle_correct = []
        split_litm_incorrect__oracle_incorrect = []

        for i, litm_sample in enumerate(litm_samples):
            litm_em = best_subspan_em(
                litm_sample["predicted_answer"],
                [ans[0] for ans in litm_sample["answers"]],
            )
            oracle_sample = oracle_samples[i]
            assert litm_sample["question"] == oracle_sample["question"]

            oracle_em = oracle_sample["em"]
            if litm_em and oracle_em:
                split_litm_correct__oracle_correct.append(litm_sample)
            elif litm_em and not oracle_em:
                split_litm_correct__oracle_incorrect.append(litm_sample)
            elif not litm_em and oracle_em:
                split_litm_incorrect__oracle_correct.append(litm_sample)
            elif not litm_em and not oracle_em:
                split_litm_incorrect__oracle_incorrect.append(litm_sample)

        print(
            f"split_{litm_case_name}_correct__oracle_correct: ",
            len(split_litm_correct__oracle_correct),
        )
        print(
            f"split_{litm_case_name}_correct__oracle_incorrect: ",
            len(split_litm_correct__oracle_incorrect),
        )
        print(
            f"split_{litm_case_name}_incorrect__oracle_correct: ",
            len(split_litm_incorrect__oracle_correct),
        )
        print(
            f"split_{litm_case_name}_incorrect__oracle_incorrect: ",
            len(split_litm_incorrect__oracle_incorrect),
        )
        print("=====================================")

        litm_correct__oracle_correct[litm_case_name] = (
            split_litm_correct__oracle_correct
        )
        litm_correct__oracle_incorrect[litm_case_name] = (
            split_litm_correct__oracle_incorrect
        )
        litm_incorrect__oracle_correct[litm_case_name] = (
            split_litm_incorrect__oracle_correct
        )
        litm_incorrect__oracle_incorrect[litm_case_name] = (
            split_litm_incorrect__oracle_incorrect
        )

    train_splits = [
        "gold_at_0",
        "gold_at_4",
    ]
    test_splits = [
        "gold_at_9",
    ]

    train_samples = []
    for train_split in train_splits:
        train_samples += (
            litm_incorrect__oracle_correct[train_split]
            + litm_correct__oracle_correct[train_split]
        )

    test_samples = []
    for test_split in test_splits:
        test_samples += (
            litm_incorrect__oracle_correct[test_split]
            + litm_correct__oracle_correct[test_split]
        )

    print(f"Train samples: {len(train_samples)}")
    print(f"Test samples: {len(test_samples)}")

    train_dataset = AttentionMapDataset(
        train_samples,
        max_sequence_length=args.max_sequence_length,
        model_name_or_path="meta-llama/Meta-Llama-3-8b-Instruct",
    )
    test_dataset = AttentionMapDataset(
        test_samples,
        max_sequence_length=args.max_sequence_length,
        model_name_or_path="meta-llama/Meta-Llama-3-8b-Instruct",
    )

    plot_sequence_length(train_dataset, "train", outputs_dir)
    plot_sequence_length(test_dataset, "test", outputs_dir)

    print_label_distribution(train_dataset, test_dataset)

    model, train_metrics, test_metrics = fit_logreg_all_features(
        train_dataset,
        test_dataset,
        args.max_sequence_length,
        regulariser=args.regulariser,
    )

    model_per_feature, train_metrics_per_feature, test_metrics_per_feature = (
        fit_logreg_per_feature(
            train_dataset,
            test_dataset,
            args.max_sequence_length,
            regulariser=args.regulariser,
        )
    )

    # Save all outputs
    torch.save(
        {
            "model": model,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "model_per_feature": model_per_feature,
            "train_metrics_per_feature": train_metrics_per_feature,
            "test_metrics_per_feature": test_metrics_per_feature,
        },
        os.path.join(outputs_dir, "lr_analysis_outputs.pt"),
    )


if __name__ == "__main__":
    main()
