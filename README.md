# DeCoRe: Decoding by Contrasting Retrieval Heads to Mitigate Hallucination


![Overview of the DeCoRe workflow](docs/assets/DeCoRe_arch.png "DeCoRe")

## Setup

### Environment variable

Setup an `.env` file in the root folder

```bash
nano .env
```

```
HF_TOKEN=<your_huggingface_write_access_token>
```

### Required Packages

```bash
pip install -r requirements.txt
```

### Retrieval Heads

The retrieval heads for the models can be found in the [`retriever_heads`](retriever_heads/) folder

To reproduce these, you may go to the [Retrieval_Head](https://github.com/nightdessert/Retrieval_Head) repository to detect the retrieval heads for each model.

```bash
# Llama3-8B-Instruct
python retrieval_head_detection.py  --model_path meta-llama/Meta-Llama-3-8B-Instruct --s 0 --e 5000

# Llama3-70B-Instruct
python retrieval_head_detection.py  --model_path meta-llama/Meta-Llama-3-70B-Instruct --s 0 --e 5000

# Mistral-7B-v0.3-Instruct
python retrieval_head_detection.py  --model_path mistralai/Mistral-7B-Instruct-v0.3 --s 0 --e 5000

# Qwen2-7B-Instruct
CUDA_VISIBLE_DEVICES=0 python retrieval_head_detection.py  --model_path Qwen/Qwen2-7B-Instruct --s 0 --e 5000
```

## Directory Structure

```
.
├── README.md
├── environment.yaml
├── requirements.txt
├── .env.example                     # Example environment file
├── .env                             # Your environment file
├── configs/                         # Hydra configs
│   ├── config.yaml                  # Default config values that will be replaced by experiment config
│   ├── data/                        # Directory containing dataset config files, that will be used in the experiment config files
│   ├── data_loader/                 # Directory containing one default data loader config file
│   ├── decoder/                     # Directory containing decoder config files (e.g., DeCoRe, Baseline, DoLa, ITI), that will be used in the experiment config files
│   ├── experiment/                  # Directory containing experiment config files per decoder
│   └── model/                       # Directory containing model config files, that will be used in the experiment config files
├── data/                            # Directory containing dataset files
├── docs/                            # Directory containing assets for documentation
├── notebooks/                       # Jupyter notebooks directory, only for creating plots
├── retrieval_heads/                 # Directory containing pre-computed retrieval heads
├── scripts/
│   ├── main.py                      # The main script for evaluating the runs
└── src
│   ├── __init__.py
│   ├── configs.py                   # Handle hydra configs
│   ├── datasets/                    # Dataset classes
│   ├── factories.py                 # Factory functions to help with instantiating dataset, model, and metric classes. Called in the run.py
│   ├── metrics/                     # Metrics classes (the name has to be the same as the dataset classes)
│   ├── models/                      # Model classes, instatiating the selected models and decoder method
│   ├── run.py                       # The run manager, handling the selection of dataset, model, and metric classes, initialising WandB, etc.
│   └── utils
│   │   ├── __init__.py
│   │   ├── common_utils.py          # Common utility functions
│   │   ├── modelling_llama.py       # Minimally modified from the Retrieval head repository 
│   │   ├── modelling_mistral.py     # Minimally modified from the Retrieval head repository
│   │   └── modelling_qwen2.py       # Minimally modified from the Retrieval head repository
```

## Evaluation

### TruthfulQA Gen Evaluation

Add OpenAI API key to your `.env` file:
```
OPENAI_API_KEY=<your_openai_api_key >
```

Fine tune `davinci-002` using the data that can be found in [`data/TruthfulQA_eval_fine_tune`](data/TruthfulQA_eval_fine_tune)

Set the fine-tuned model id to the `.env` file

```
GPT_JUDGE_NAME=<your_gpt_judge_fine_tuned_model_id>
GPT_INFO_NAME=<your_gpt_info_fine_tuned_model_id>
```

The ids of both fine-tuned models would usually be prefixed by `ft:davinci-002:...`.

Download the predictions from WandB (if you follow my codebase, it will be in a json format). Amd pass it on to the evaluation script.

```
# Evaluate!

python src/metrics/truthfulqa_gen.py --pred_filepath=path/to/truthfulqa_model_prediction.json
```