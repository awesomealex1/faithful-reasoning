import torch
import pandas as pd
import numpy as np
from pyvene import embed_to_distrib, top_vals, format_token
from pyvene import (
    IntervenableModel,
    VanillaIntervention,
    RepresentationConfig,
    IntervenableConfig,
    ConstantSourceIntervention,
    LocalistRepresentationIntervention,
)
from pyvene import create_llama2, create_llama

from plotnine import ggplot, geom_tile, aes, theme, element_text, xlab, ylab, ggsave
from plotnine.scales import scale_y_reverse, scale_fill_cmap
from tqdm import tqdm

titles = {
    "block_output": "single restored layer in LLAMA-3-8B",
    "mlp_activation": "center of interval of 10 patched mlp layer",
    "attention_output": "center of interval of 10 patched attn layer",
}

colors = {
    "block_output": "Purples",
    "mlp_activation": "Greens",
    "attention_output": "Reds",
}

from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

device = "cuda:0" if torch.cuda.is_available() else "cpu"
# config, tokenizer, llama = create_llama2(name="llama2-xl")
model_name = "meta-llama/Meta-Llama-3-8B"
config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
llama = AutoModelForCausalLM.from_pretrained(
    model_name,
    config=config,
    torch_dtype=torch.bfloat16,  # save memory
    device_map="auto",
)
# llama.to(device)

base = "The Space Needle is in downtown"
inputs = [
    tokenizer(base, return_tensors="pt").to(device),
]
print(base)
res = llama(**inputs[0], return_dict=True, output_hidden_states=True)
distrib = embed_to_distrib(llama, res.hidden_states[-1], logits=False)
print("Before corruption")
top_vals(tokenizer, distrib[0][-1], n=10)


class NoiseIntervention(ConstantSourceIntervention, LocalistRepresentationIntervention):
    def __init__(self, embed_dim, noise_level=0.01, **kwargs):
        super().__init__()
        self.interchange_dim = embed_dim
        rs = np.random.RandomState(1)
        prng = lambda *shape: rs.randn(*shape)
        self.noise = torch.from_numpy(prng(1, 4, embed_dim)).to(device)
        # noise level gpt2-xl = 0.13462981581687927
        self.noise_level = noise_level

    def forward(self, base, source=None, subspaces=None):
        base[..., : self.interchange_dim] += self.noise * self.noise_level
        return base

    def __str__(self):
        return f"NoiseIntervention(embed_dim={self.embed_dim})"


def corrupted_config(model_type):
    config = IntervenableConfig(
        model_type=model_type,
        representations=[
            RepresentationConfig(
                0,  # layer
                "block_input",  # intervention type
            ),
        ],
        intervention_types=NoiseIntervention,
    )
    return config


print("After corruption")
base = tokenizer("The Space Needle is in downtown", return_tensors="pt").to(device)
config = corrupted_config(type(llama))
intervenable = IntervenableModel(config, llama)
_, counterfactual_outputs = intervenable(
    base, unit_locations={"base": ([[[0, 1, 2, 3]]])}
)

distrib = embed_to_distrib(
    llama, counterfactual_outputs.hidden_states[-1], logits=False
)
top_vals(tokenizer, distrib[0][-1], n=10)


def restore_corrupted_with_interval_config(
    layer, stream="mlp_activation", window=10, num_layers=32
):
    start = max(0, layer - window // 2)
    end = min(num_layers, layer - (-window // 2))
    config = IntervenableConfig(
        representations=[
            RepresentationConfig(
                0,  # layer
                "block_input",  # intervention type
            ),
        ]
        + [
            RepresentationConfig(
                i,  # layer
                stream,  # intervention type
            )
            for i in range(start, end)
        ],
        intervention_types=[NoiseIntervention] + [VanillaIntervention] * (end - start),
    )
    return config


# should finish within 1 min with a standard 12G GPU
token = tokenizer.encode(" Seattle")[1]  # first token is sos
print(token)

import os

os.makedirs("./tutorial_data", exist_ok=True)

for stream in ["block_output", "mlp_activation", "attention_output"]:
    data = []
    for layer_i in tqdm(range(llama.config.num_hidden_layers)):
        for pos_i in range(7):
            config = restore_corrupted_with_interval_config(
                layer_i, stream, window=1 if stream == "block_output" else 10
            )
            n_restores = len(config.representations) - 1
            intervenable = IntervenableModel(config, llama)
            _, counterfactual_outputs = intervenable(
                base,
                [None] + [base] * n_restores,
                {
                    "sources->base": (
                        [None] + [[[pos_i]]] * n_restores,
                        [[[0, 1, 2, 3]]] + [[[pos_i]]] * n_restores,
                    )
                },
            )
            distrib = embed_to_distrib(
                llama, counterfactual_outputs.hidden_states[-1], logits=False
            )
            prob = distrib[0][-1][token].detach().cpu().item()
            data.append({"layer": layer_i, "pos": pos_i, "prob": prob})
    df = pd.DataFrame(data)
    df.to_csv(f"./tutorial_data/pyvene_rome_llama3_8b_{stream}.csv")


for stream in ["block_output", "mlp_activation", "attention_output"]:
    df = pd.read_csv(f"./tutorial_data/pyvene_rome_llama3_8b_{stream}.csv")
    df["layer"] = df["layer"].astype(int)
    df["pos"] = df["pos"].astype(int)
    df["p(Seattle)"] = df["prob"].astype(float)

    custom_labels = ["The*", "Space*", "Need*", "le*", "is", "in", "downtown"]
    breaks = [0, 1, 2, 3, 4, 5, 6]

    plot = (
        ggplot(df, aes(x="layer", y="pos"))
        + geom_tile(aes(fill="p(Seattle)"))
        + scale_fill_cmap(colors[stream])
        + xlab(titles[stream])
        + scale_y_reverse(limits=(-0.5, 6.5), breaks=breaks, labels=custom_labels)
        + theme(figure_size=(5, 4))
        + ylab("")
        + theme(axis_text_y=element_text(angle=90, hjust=1))
    )
    ggsave(
        plot, filename=f"./tutorial_data/pyvene_rome_llama3_8b_{stream}.pdf", dpi=200
    )
    print(plot)
