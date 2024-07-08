# DeCoRe: Decoding by Contrasting Retrieval Heads to Mitigate Hallucination

## Metadata
- Paper draft: https://www.overleaf.com/6919357326kssvrkxzmbyz#212775

## Setup

### Required Packages

```bash
pip install -r requirements.txt
```

### Retrieval Heads

The retrieval heads for the models can be found in the [`retriever_heads`](retriever_heads/) folder

To reproduce these, you may go to the [Retrieval_Head](https://github.com/nightdessert/Retrieval_Head) repository to detect the retrieval heads for each model.

```bash
# Llama3-8B
python retrieval_head_detection.py  --model_path meta-llama/Meta-Llama-3-8B --s 0 --e 5000
# Llama3-8B-Instruct
python retrieval_head_detection.py  --model_path meta-llama/Meta-Llama-3-8B-Instruct --s 0 --e 5000

# Llama3-70B
python retrieval_head_detection.py  --model_path meta-llama/Meta-Llama-3-70B --s 0 --e 5000
# Llama3-70B-Instruct
python retrieval_head_detection.py  --model_path meta-llama/Meta-Llama-3-70B-Instruct --s 0 --e 5000

# Mistral-7B-v0.3
python retrieval_head_detection.py  --model_path mistralai/Mistral-7B-v0.3 --s 0 --e 5000
# Mistral-7B-v0.3-Instruct
python retrieval_head_detection.py  --model_path mistralai/Mistral-7B-Instruct-v0.3 --s 0 --e 5000
```