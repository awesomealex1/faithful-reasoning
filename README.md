# Faithful Reasoning

## 🛠️ Setup

### 🌏 Environment variable

Setup an `.env` file in the root folder

```bash
nano .env
```

```
HF_TOKEN=<your_huggingface_write_access_token>
```

### 📦 Required Packages

#### 🐍 conda
```bash
conda env create -f environment.yaml
conda activate faithful
```

#### 🐍 pip
```bash
pip install -r requirements.txt
```

For development, we use `black` and `isort`. If you wish to proceed without them and if you are using VSCode, update `.vscode/settings.json` accordingly.

### 🪄 To WandB or not to WandB

If you wish to use WandB, please update the `configs/config.yaml`, specifically the values of `wandb_project` and `wandb_entity`.
We generally recommend using WandB, but if you prefer not to, you can still run the script using the `debug` flag or by setting the value of `debug` in `configs/config.yaml` into `true`. This will bypass the wandb initialisation and logging.

### Download Data

Download the data needed by running:

```bash
sh scripts/download_react_data.sh
```

### Elasticsearch

Go to `src/utils` and install ElasticSearch:

#### Install on Mac
```bash
# source: https://www.elastic.co/guide/en/elasticsearch/reference/current/targz.html
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.10.2-darwin-x86_64.tar.gz
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.10.2-darwin-x86_64.tar.gz.sha512
shasum -a 512 -c elasticsearch-7.10.2-darwin-x86_64.tar.gz.sha512
tar -xzf elasticsearch-7.10.2-darwin-x86_64.tar.gz
cd elasticsearch-7.10.2/
./bin/elasticsearch # start the server
pkill -f elasticsearch # to stop the server
```

#### Install on Linux

```bash
# source: https://www.elastic.co/guide/en/elasticsearch/reference/8.1/targz.html
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.10.2-linux-x86_64.tar.gz
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.10.2-linux-x86_64.tar.gz.sha512
shasum -a 512 -c elasticsearch-7.10.2-linux-x86_64.tar.gz.sha512
tar -xzf elasticsearch-7.10.2-linux-x86_64.tar.gz
cd elasticsearch-7.10.2/
./bin/elasticsearch # start the server
pkill -f elasticsearch # to stop the server
```

To run the server (from root directory):

```bash
./src/utils/elasticsearch-7.10.2/bin/elasticsearch
```

To kill the server: 

```bash
pkill -f elasticsearch
```

After starting the elasticsearch server you need to index the wikipedia corpuses, for which data is downloaded into the corresponding folder in `data`. (Make sure to have run `download_react_data.sh`). First start the retriever server:

```bash
uvicorn serve:app --port 8000 --app-dir src/utils/retriever_server
```

Then index the corpuses (need to do this only once):

```bash
python src/utils/retriever_server/build_index.py {dataset_name} # hotpotqa, 2wikimultihopqa, musique
```

### Run ReAct

Once this is done you can run the scripts to run ReAct! This is an example, just adjust the dataset/decoder/framework/model to your needs:

```bash
python scripts/main.py experiment=musique/baseline/react/qwen2_7b_instruct
```

### Use our evaluation scripts

Our evaluation scripts are in ```/scripts``` and called ```run_experiments_[framework]_[model].py```. This will generate data in the outputs directory which can be used to run the ```calculate_metrics_for_outputs_[framework].py``` scripts to generate metrics for the experiments. Just change the name of the output file to be the one you created!


## 🌲 Directory Structure

```
.
├── README.md
├── environment.yaml
├── requirements.txt
├── .env.example                     # Example environment file
├── .env                             # Your environment file
├── configs/                         # Hydra configs
│   ├── config.yaml                  # Default config values that will be replaced by experiment config
│   ├── data/                        # Directory containing dataset config files, that will be used in the experiment config files
│   ├── data_loader/                 # Directory containing one default data loader config file
│   ├── decoder/                     # Directory containing decoder config files (e.g., DeCoRe, Baseline, DoLa, CAD), that will be used in the experiment config files
│   ├── experiment/                  # Directory containing experiment config files per decoder
|   ├── framework/                   # Directory containing frameowkr config files (e.g., ReAct,..) 
│   └── model/                       # Directory containing model config files, that will be used in the experiment config files
├── data/                            # Directory containing dataset files
├── docs/                            # Directory containing assets for documentation
├── retrieval_heads/                 # Directory containing pre-computed retrieval heads
├── scripts/
│   ├── download_react_data.sh       # Script to download (large) datasets for ReAct
│   ├── main.py                      # The main script for evaluating the runs
│   ├── run_experiments...           # Script to run experiments with seeds
└── src/
    ├── __init__.py
    ├── configs.py                   # Handle Hydra configs
    ├── datasets/                    # Dataset classes
    ├── factories.py                 # Factory functions to help with instantiating dataset, model, and metric classes. Called in the run.py
    ├── metrics/                     # Metrics classes (the name must match the dataset classes)
    ├── models/                      # Model classes, instantiating the selected models and decoder method
    ├── run.py                       # The run manager, handling the selection of dataset, model, and metric classes, initializing WandB, etc.
    └── utils/
        ├── __init__.py
        ├── common_utils.py          # Common utility functions
        ├── modelling_llama.py       # Minimally modified from the Retrieval head repository
        ├── modelling_mistral.py     # Minimally modified from the Retrieval head repository
        └── modelling_qwen2.py       # Minimally modified from the Retrieval head repository
```

## Acknowledgements

Some parts of code are based upon ['DeCoRe: Decoding by Contrasting Retrieval Heads to Mitigate Hallucinations'](https://github.com/aryopg/decore) and ['Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions'](https://github.com/StonyBrookNLP/ircot).