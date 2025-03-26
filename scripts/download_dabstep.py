import os
import json
import huggingface_hub

huggingface_hub.login(token=os.getenv("HF_TOKEN", ""))

CONTEXT_FILENAMES = [
    "data/context/acquirer_countries.csv",
    "data/context/payments-readme.md",
    "data/context/payments.csv",
    "data/context/merchant_category_codes.csv",
    "data/context/fees.json",
    "data/context/merchant_data.json",
    "data/context/manual.md",
]
DATA_DIR = "data/DABStep"
for filename in CONTEXT_FILENAMES:
    huggingface_hub.hf_hub_download(
        repo_id="adyen/DABstep",
        repo_type="dataset",
        filename=filename,
        local_dir=DATA_DIR,
        force_download=True
    )

CONTEXT_FILENAMES = [f"{DATA_DIR}/{filename}" for filename in CONTEXT_FILENAMES]

for file in CONTEXT_FILENAMES:
    if os.path.exists(file):
        print(f"{file} exists.")
    else:
        print(f"{file} does not exist.")