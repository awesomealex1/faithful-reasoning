from typing import Dict, List, Tuple, Any
import json
from src.metrics.ircot_metrics.support_em_f1 import SupportEmF1Metric
from src.metrics.ircot_metrics.answer_support_recall import AnswerSupportRecallMetric
from src.metrics.ircot_metrics.squad_answer_em_f1 import SquadAnswerEmF1Metric
import os

def calculate_metrics(output_file_path: str, gold_file_path: str) -> Dict[str, Any]:
    # Load the output file (predictions)
    with open(output_file_path, 'r') as f:
        outputs = [json.loads(line) for line in f]
    
    # Load the gold file (ground truth)
    with open(gold_file_path, 'r') as f:
        gold_data = [json.loads(line) for line in f]
    
    # Create a dictionary mapping question IDs to gold data for easy lookup
    gold_dict = {item['question_id']: item for item in gold_data}
    
    # Initialize metric calculators
    squad_metric = SquadAnswerEmF1Metric()
    
    for output in outputs:
        # Get the question ID from the output
        question_id = output['idx']

        predicted_answer = output["predicted_answer"]
        
        if question_id in gold_dict:
            gold_item = gold_dict[question_id]
            
            # Extract gold answers
            gold_answers = [ans['spans'][0] for ans in gold_item['answers_objects']]
            
            # Calculate squad EM and F1
            squad_metric(predicted_answer, gold_answers)
    
    # Get the calculated metrics
    squad_metrics = squad_metric.get_metric(reset=True)
    
    # Combine metrics
    return {**squad_metrics}

output_prefix = "outputs/"

outputs = {
    "llama": {
        "hotpot": {
            1234: [
                "2025-03-25/22-28-20",
            ],
            3782: [
                "2025-03-25/22-58-22",
            ],
            9539: [
                "2025-03-25/23-27-46",
            ]
        },
        "wiki": {
            1234: [
                "2025-03-25/22-25-25",   
            ],
            3782: [
                "2025-03-25/22-55-30",
            ],
            9539: [
                "2025-03-25/23-24-56",
            ]
        },
        "musique": {
            1234: [
                "2025-03-25/22-31-00",   
            ],
            3782: [
                "2025-03-25/23-01-04",
            ],
            9539: [
                "2025-03-25/23-30-32",
            ]
        }

    },
    "qwen": {
        "hotpot": {
            1234: [
                "2025-03-25/22-38-36",
            ],
            3782: [
                "2025-03-25/23-08-18",   
            ],
            9539: [
                "2025-03-25/23-38-02",   
            ]
        },
        "wiki": {
            1234: [
                "2025-03-25/22-33-53",
            ],
            3782: [
                "2025-03-25/23-03-46",   
            ],
            9539: [
                "2025-03-25/23-33-22",   
            ]
        },
        "musique": {
            1234: [
                "2025-03-25/22-45-33",   
            ],
            3782: [
                "2025-03-25/23-15-10",   
            ],
            9539: [
                "2025-03-25/23-44-55",   
            ]
        }
    }
}

#output_path = "outputs/2025-03-25/01-55-08/pred_WikiMultihopQA_Qwen2-7b-Instruct__ContextAwareDecoding__ReAct.json"
#output_path = "outputs/2025-03-05/11-51-57/pred_MuSiQue_LLaMA3-8b-Instruct__DoLa__ReAct.json"
output_path = "outputs/2025-03-25/21-43-14/pred_MuSiQue_LLaMA3-8b-Instruct__Baseline__ReAct.json"
wiki_questions_path = "data/2WikiMultiHopQA/test_subsampled.jsonl"
musique_questions_path = "data/MuSiQue/test_subsampled.jsonl"
hotpot_questions_path = "data/HotpotQA/test_subsampled.jsonl"

for model in outputs:
    for dataset in outputs[model]:
        for seed in outputs[model][dataset]:
            for out_dir in outputs[model][dataset][seed]:
                dir_path = os.path.join(output_prefix, out_dir)
                if os.path.exists(dir_path):
                    json_files = [f for f in os.listdir(dir_path) if f.endswith('.json')]
                    if json_files:
                        # Take the first JSON file found
                        json_file = json_files[0]
                        output_path = os.path.join(dir_path, json_file)
                        
                        # Determine which gold data file to use based on the dataset
                        if dataset == "hotpot":
                            gold_file_path = hotpot_questions_path
                        elif dataset == "wiki":
                            gold_file_path = wiki_questions_path
                        elif dataset == "musique":
                            gold_file_path = musique_questions_path
                        
                        if "Baseline" in json_file:
                            decoder = "baseline"
                        elif "ContextAwareDecoding" in json_file:
                            decoder = "cad"
                        elif "DoLa" in json_file:
                            decoder = "dola"
                        elif "DeCoReEntropy" in json_file:
                            decoder = "decore"
                        
                        # Calculate metrics and print results
                        print(f"Model: {model}, Dataset: {dataset}, Decoder: {decoder}, Seed: {seed}, Run: {out_dir}")
                        metrics = calculate_metrics(output_path, gold_file_path)
                        print(metrics)
                        print("-" * 50)
                    else:
                        print(f"No JSON files found in {dir_path}")
                else:
                    print(f"Directory not found: {dir_path}")