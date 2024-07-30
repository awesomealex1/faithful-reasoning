import logging
import os
import sys

sys.path.append(os.getcwd())

import huggingface_hub

from huggingface_hub import upload_folder, create_repo

from dotenv import load_dotenv

load_dotenv(".env")


def main():
    huggingface_hub.login(token=os.getenv("HF_TOKEN", ""))

    repo_name = "attention_maps"
    username = "aryopg"
    repo_id = f"{username}/{repo_name}"
    create_repo(repo_id=repo_id, repo_type="dataset", private=True)

    upload_folder(
        folder_path="./attention_maps",
        repo_id=repo_id,
        repo_type="dataset",
    )


if __name__ == "__main__":
    main()
