import argparse
import logging
import os
import sys
import json

sys.path.append(os.getcwd())

import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ignore warnings
import warnings

warnings.filterwarnings("ignore")


def plot_all(metrics_dict, outputs_dir):
    metrics_names = [
        metrics_name
        for metrics_name in list(metrics_dict["l1"].keys())
        if metrics_name != "sequence_length"
    ]
    fig, axs = plt.subplots(1, len(metrics_names), figsize=(5 * len(metrics_names), 5))
    for i, metric_name in enumerate(metrics_names):
        sequence_length = [int(x) for x in metrics_dict["l1"]["sequence_length"]]

        l1_test_metric_values = metrics_dict["l1"][metric_name]
        l2_test_metric_values = metrics_dict["l2"][metric_name]
        elastic_test_metric_values = metrics_dict["elastic"][metric_name]

        l1_test_metric_values = [
            0 if np.isnan(metric) else metric for metric in l1_test_metric_values
        ]
        l2_test_metric_values = [
            0 if np.isnan(metric) else metric for metric in l2_test_metric_values
        ]
        elastic_test_metric_values = [
            0 if np.isnan(metric) else metric for metric in elastic_test_metric_values
        ]

        sns.lineplot(x=sequence_length, y=l1_test_metric_values, label="L1", ax=axs[i])
        sns.lineplot(x=sequence_length, y=l2_test_metric_values, label="L2", ax=axs[i])
        sns.lineplot(
            x=sequence_length,
            y=elastic_test_metric_values,
            label="Elastic",
            ax=axs[i],
        )
        axs[i].set_ylim(0, 1)
        axs[i].set_xlabel("Max Sequence Length")
        axs[i].set_title(f"Test {metric_name.upper()}")
        plt.legend()
    plt.savefig(os.path.join(outputs_dir, "metrics_all_features_by_seq_length.png"))


def plot_per_feature(metrics_dict, outputs_dir):
    feature_names = list(metrics_dict.keys())
    metrics_names = [
        metrics_name
        for metrics_name in list(metrics_dict["BOS"]["l1"].keys())
        if metrics_name != "sequence_length"
    ]
    fig, axs = plt.subplots(
        len(feature_names),
        len(metrics_names),
        figsize=(5 * len(metrics_names), len(feature_names) * 5),
    )
    for i, feature_name in enumerate(feature_names):
        for j, metric_name in enumerate(metrics_names):
            sequence_length = [
                int(x) for x in metrics_dict[feature_name]["l1"]["sequence_length"]
            ]

            l1_test_metric_values = metrics_dict[feature_name]["l1"][metric_name]
            l2_test_metric_values = metrics_dict[feature_name]["l2"][metric_name]
            elastic_test_metric_values = metrics_dict[feature_name]["elastic"][
                metric_name
            ]

            l1_test_metric_values = [
                0 if np.isnan(metric) else metric for metric in l1_test_metric_values
            ]
            l2_test_metric_values = [
                0 if np.isnan(metric) else metric for metric in l2_test_metric_values
            ]
            elastic_test_metric_values = [
                0 if np.isnan(metric) else metric
                for metric in elastic_test_metric_values
            ]

            sns.lineplot(
                x=sequence_length, y=l1_test_metric_values, label="L1", ax=axs[i][j]
            )
            sns.lineplot(
                x=sequence_length, y=l2_test_metric_values, label="L2", ax=axs[i][j]
            )
            sns.lineplot(
                x=sequence_length,
                y=elastic_test_metric_values,
                label="Elastic",
                ax=axs[i][j],
            )
            axs[i][j].set_ylim(0, 1)
            axs[i][j].set_xlabel("Max Sequence Length")
            axs[i][j].set_title(f"({feature_name}) Test {metric_name.upper()}")
            plt.legend()
    plt.savefig(os.path.join(outputs_dir, "metrics_per_feature_by_seq_length.png"))


def main():
    data_dir = "./outputs/lr_analyses"

    metrics_all = {}
    metrics_per_feature = {}

    for file in os.listdir(data_dir):
        if file.endswith(".pt"):
            print(file)

            split_names = file.replace(".pt", "").split("__")
            regulariser_name = split_names[1].split("_")[1]
            seqlen = split_names[2].split("_")[1]

            data = torch.load(os.path.join(data_dir, file))
            print("All")
            print(f"Test: {data['test_metrics']}")
            print("Per Feature")
            print(f"Test: {data['test_metrics_per_feature']}")

            if regulariser_name not in metrics_all:
                metrics_all[regulariser_name] = {}
            if "sequence_length" not in metrics_all[regulariser_name]:
                metrics_all[regulariser_name]["sequence_length"] = []
            metrics_all[regulariser_name]["sequence_length"] += [seqlen]
            for key in data["test_metrics"]:
                if key.upper() not in metrics_all[regulariser_name]:
                    metrics_all[regulariser_name][key.upper()] = []
                metrics_all[regulariser_name][key.upper()] += [
                    data["test_metrics"][key]
                ]

            for feature in data["test_metrics_per_feature"]:
                if feature not in metrics_per_feature:
                    metrics_per_feature[feature] = {}
                if regulariser_name not in metrics_per_feature[feature]:
                    metrics_per_feature[feature][regulariser_name] = {}
                if (
                    "sequence_length"
                    not in metrics_per_feature[feature][regulariser_name]
                ):
                    metrics_per_feature[feature][regulariser_name][
                        "sequence_length"
                    ] = []
                metrics_per_feature[feature][regulariser_name]["sequence_length"] += [
                    seqlen
                ]
                for key in data["test_metrics_per_feature"][feature]:
                    if (
                        key.upper()
                        not in metrics_per_feature[feature][regulariser_name]
                    ):
                        metrics_per_feature[feature][regulariser_name][key.upper()] = []
                    metrics_per_feature[feature][regulariser_name][key.upper()] += [
                        data["test_metrics_per_feature"][feature][key]
                    ]

    # All

    plot_all(metrics_all, data_dir)
    plot_per_feature(metrics_per_feature, data_dir)


if __name__ == "__main__":
    main()
