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


FEATURES_LIST = [
    "BOS",
    "Instruction",
    "ICL Demonstration",
    "Contexts",
    "Question",
    "Answer Prefix",
    "New Tokens",
]


def get_top_k_heads(
    model,
    max_sequence_length,
    k=10,
    num_layers=32,
    num_heads=32,
    num_features=len(FEATURES_LIST),
):
    coef = model.coef_[0].reshape(
        num_layers, num_heads, max_sequence_length, num_features
    )
    coef = np.sum(coef, axis=-2)
    coef = np.sum(coef, axis=-1)
    top_k_pos_layers, top_k_pos_heads = np.unravel_index(
        np.argsort(coef.flatten())[-k:],
        coef.shape,
    )
    top_k_pos_ids = [(l, h) for l, h in zip(top_k_pos_layers, top_k_pos_heads)]
    top_k_neg_layers, top_k_neg_heads = np.unravel_index(
        np.argsort(coef.flatten())[:k],
        coef.shape,
    )
    top_k_neg_ids = [(l, h) for l, h in zip(top_k_neg_layers, top_k_neg_heads)]
    print(f"Top Aggregated Coefficient")
    print("Positive Coef Values")
    for pos_id in top_k_pos_ids:
        print(f"Layer {pos_id[0]}, Head {pos_id[1]}: {coef[pos_id[0], pos_id[1]]}")

    print("Negative Coef Values")
    for neg_id in top_k_neg_ids:
        print(f"Layer {neg_id[0]}, Head {neg_id[1]}: {coef[neg_id[0], neg_id[1]]}")
    return top_k_pos_ids, top_k_neg_ids


def save_coef_to_json(
    model,
    max_sequence_length,
    num_layers=32,
    num_heads=32,
    num_features=len(FEATURES_LIST),
    outputs_dir="./hallucination_heads",
):
    coef = model.coef_[0].reshape(
        num_layers, num_heads, max_sequence_length, num_features
    )

    abs_coef = np.absolute(coef)
    max_abs_indices = np.argmax(abs_coef, axis=-2)

    num_layers, num_heads, _, num_features = coef.shape
    rows, cols, feats = np.ogrid[:num_layers, :num_heads, :num_features]
    coef = coef[rows, cols, max_abs_indices, feats]

    head_coefs = {}
    for feature_id, feature_name in enumerate(FEATURES_LIST):
        head_coefs[feature_name] = {}
        for layer_id in range(num_layers):
            for head_id in range(num_heads):
                head_coefs[feature_name][f"{layer_id}-{head_id}"] = coef[
                    layer_id, head_id, feature_id
                ]

    print(head_coefs)
    with open(os.path.join(outputs_dir, f"Meta-Llama-3-8B_Instruct.json"), "w") as file:
        json.dump(head_coefs, file)


def plot_aggregate_coef(
    model,
    regulariser,
    max_sequence_length,
    retrieval_heads,
    k=10,
    num_layers=32,
    num_heads=32,
    num_features=len(FEATURES_LIST),
    outputs_dir="./outputs/lr_analyses",
):
    coef = model.coef_[0].reshape(
        num_layers, num_heads, max_sequence_length, num_features
    )

    # Count non zero coefficients
    non_zero_count = np.count_nonzero(coef)
    print(
        f"Non-zero count: {non_zero_count} ({non_zero_count/(num_layers*num_heads*max_sequence_length*num_features)*100:.2f}%)"
    )

    # Max on sequence
    abs_coef = np.absolute(coef)
    max_abs_indices = np.argmax(abs_coef, axis=-2)

    num_layers, num_heads, _, num_features = coef.shape
    rows, cols, feats = np.ogrid[:num_layers, :num_heads, :num_features]
    coef = coef[rows, cols, max_abs_indices, feats]

    # # Mean on sequence dim
    # coef = np.mean(coef, axis=-2)

    fig, axs = plt.subplots(
        1, len(FEATURES_LIST), figsize=(len(FEATURES_LIST) * 5, 5)
    )  # Adjusted figsize for better display
    for feature_id, feature_name in enumerate(FEATURES_LIST):
        axs[feature_id].set_title(f"Feature: {feature_name}")
        sns.heatmap(
            coef[:, :, feature_id],
            ax=axs[feature_id],
            center=0,
            cmap="coolwarm",
            cbar=True,
        )
        axs[feature_id].set_xlabel("Head")
        axs[feature_id].set_ylabel("Layer")

        for coord in retrieval_heads:
            rect = Rectangle(
                (coord[1], coord[0]), 1, 1, fill=False, edgecolor="red", linewidth=2
            )
            axs[feature_id].add_patch(rect)

        top_k_pos_layers, top_k_pos_heads = np.unravel_index(
            np.argsort(coef[:, :, feature_id].flatten())[-k:],
            coef[:, :, feature_id].shape,
        )
        top_k_pos_ids = [(l, h) for l, h in zip(top_k_pos_layers, top_k_pos_heads)]
        top_k_neg_layers, top_k_neg_heads = np.unravel_index(
            np.argsort(coef[:, :, feature_id].flatten())[:k],
            coef[:, :, feature_id].shape,
        )
        top_k_neg_ids = [(l, h) for l, h in zip(top_k_neg_layers, top_k_neg_heads)]

        print(f"Top Coefficient for {feature_name}")
        print("Positive Coef Values")
        for pos_id in top_k_pos_ids:
            print(
                f"Layer {pos_id[0]}, Head {pos_id[1]}: {coef[pos_id[0], pos_id[1], feature_id]}"
            )

        print("Negative Coef Values")
        for neg_id in top_k_neg_ids:
            print(
                f"Layer {neg_id[0]}, Head {neg_id[1]}: {coef[neg_id[0], neg_id[1], feature_id]}"
            )
        print("========================")
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig(
        os.path.join(
            outputs_dir,
            f"agg_coef__lr_{regulariser}__seqlen_{max_sequence_length}__all_features_heatmap.png",
        )
    )


def plot_aggregate_coef_one_feature(
    model,
    regulariser,
    max_sequence_length,
    retrieval_heads,
    feature_name,
    k=10,
    num_layers=32,
    num_heads=32,
    num_features=len(FEATURES_LIST),
    outputs_dir="./outputs/lr_analyses",
):
    coef = model.coef_[0].reshape(num_layers, num_heads, max_sequence_length)

    # Count non zero coefficients
    non_zero_count = np.count_nonzero(coef)
    print(
        f"Non-zero count: {non_zero_count} ({non_zero_count/(num_layers*num_heads*max_sequence_length)*100:.2f}%)"
    )

    # Max on sequence
    abs_coef = np.absolute(coef)
    max_abs_indices = np.argmax(abs_coef, axis=-1)

    num_layers, num_heads, _ = coef.shape
    rows, cols = np.ogrid[:num_layers, :num_heads]
    coef = coef[rows, cols, max_abs_indices]

    # # Mean on sequence dim
    # coef = np.mean(coef, axis=-2)

    fig, ax = plt.subplots(figsize=(5, 5))  # Adjusted figsize for better display
    ax.set_title(f"Feature: {feature_name}")
    sns.heatmap(
        coef,
        ax=ax,
        center=0,
        cmap="coolwarm",
        cbar=True,
    )
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")

    for coord in retrieval_heads:
        rect = Rectangle(
            (coord[1], coord[0]), 1, 1, fill=False, edgecolor="red", linewidth=2
        )
        ax.add_patch(rect)

    top_k_pos_layers, top_k_pos_heads = np.unravel_index(
        np.argsort(coef.flatten())[-k:],
        coef.shape,
    )
    top_k_pos_ids = [(l, h) for l, h in zip(top_k_pos_layers, top_k_pos_heads)]
    top_k_neg_layers, top_k_neg_heads = np.unravel_index(
        np.argsort(coef.flatten())[:k],
        coef.shape,
    )
    top_k_neg_ids = [(l, h) for l, h in zip(top_k_neg_layers, top_k_neg_heads)]

    print(f"Top Coefficient for {feature_name}")
    print("Positive Coef Values")
    for pos_id in top_k_pos_ids:
        print(f"Layer {pos_id[0]}, Head {pos_id[1]}: {coef[pos_id[0], pos_id[1]]}")

    print("Negative Coef Values")
    for neg_id in top_k_neg_ids:
        print(f"Layer {neg_id[0]}, Head {neg_id[1]}: {coef[neg_id[0], neg_id[1]]}")
    print("========================")
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig(
        os.path.join(
            outputs_dir,
            f"agg_coef__lr_{regulariser}__seqlen_{max_sequence_length}__feature_{feature_name}_heatmap.png",
        )
    )


def main():
    data_dir = "./outputs/lr_analyses"

    with open("retrieval_heads/Meta-Llama-3-8B-Instruct.json") as file:
        head_list = json.loads(file.readline())

    stable_block_list = [(l[0], np.mean(l[1])) for l in head_list.items()]
    stable_block_list = sorted(stable_block_list, key=lambda x: x[1], reverse=True)
    retrieval_heads = [
        [int(ll) for ll in l[0].split("-")] for l in stable_block_list if l[1] >= 0.5
    ]
    retrieval_scores = [l[1] for l in stable_block_list if l[1] >= 0.5]
    for head, score in zip(retrieval_heads, retrieval_scores):
        print(f"Head: {head}, Score: {score}")
    print(len(retrieval_heads))

    for regulariser in ["l1", "l2", "elastic"]:
        print(f">>>>> Regulariser: {regulariser}")
        model_filepath = f"lr_analysis_outputs__lr_{regulariser}__seqlen_4.pt"
        state_dict = torch.load(os.path.join(data_dir, model_filepath))
        model = state_dict["model"]
        plot_aggregate_coef(
            model,
            regulariser=regulariser,
            max_sequence_length=4,
            retrieval_heads=retrieval_heads,
        )
        get_top_k_heads(
            model,
            max_sequence_length=4,
            k=20,
        )
        # for feature_name in FEATURES_LIST:
        #     model = state_dict["model_per_feature"][feature_name]
        #     plot_aggregate_coef_one_feature(
        #         model,
        #         regulariser=regulariser,
        #         max_sequence_length=4,
        #         retrieval_heads=retrieval_heads,
        #         feature_name=feature_name,
        #     )
    l2_model = torch.load(
        os.path.join(data_dir, "lr_analysis_outputs__lr_l2__seqlen_4.pt")
    )
    save_coef_to_json(l2_model["model"], 4)


if __name__ == "__main__":
    main()
