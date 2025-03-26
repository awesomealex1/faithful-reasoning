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
    support_metric = SupportEmF1Metric()
    answer_recall_metric = AnswerSupportRecallMetric()
    squad_metric = SquadAnswerEmF1Metric()
    
    for output in outputs:
        # Get the question ID from the output
        question_id = output['idx']

        predicted_answer = output["predicted_answer"]
        
        if question_id in gold_dict:
            gold_item = gold_dict[question_id]
            
            # Extract predicted supporting paragraphs/titles
            # This would depend on your specific output format
            # Assuming your reasoning chain actions identify supporting paragraphs
            predicted_titles, predicted_paragraphs = extract_predicted_titles_paras(output['reasoning_chain'])
            
            # Extract gold supporting paragraphs/titles
            gold_titles = [ctx['title'] for ctx in gold_item['contexts'] if ctx['is_supporting']]
            
            gold_paragraphs = [ctx['paragraph_text'] for ctx in gold_item['contexts'] if ctx['is_supporting']]
            
            # Calculate support metrics (precision, recall, F1, EM)
            support_metric(predicted_titles, predicted_paragraphs, gold_titles, gold_paragraphs)
            
            # Extract gold answers
            gold_answers = [ans['spans'][0] for ans in gold_item['answers_objects']]
            
            # Calculate answer support recall
            answer_recall_metric(predicted_paragraphs, gold_answers)

            # Calculate squad EM and F1
            squad_metric(predicted_answer, gold_answers)
    
    # Get the calculated metrics
    support_metrics = support_metric.get_metric(reset=True)
    answer_recall_metrics = answer_recall_metric.get_metric(reset=True)
    squad_metrics = squad_metric.get_metric(reset=True)
    
    # Combine metrics
    return {**support_metrics, **answer_recall_metrics, **squad_metrics}

def extract_predicted_titles_paras(reasoning_chain: List[str]) -> List[str]:
    titles = []
    paragraphs = []
    for step in reasoning_chain:
        if "Action" in step and "Search[" in step:
            # Extract search queries which might correspond to titles
            query = step.split("Search[")[1].split("]")[0]
            titles.append(query)
        if "Observation" in step and "Could not find" not in step:
            # Extract observation text which contains paragraph content
            observation_text = step.split("Observation")[1].split(":", 1)[1].strip()
            paragraphs.append(observation_text)
    return titles, paragraphs

output_prefix = "outputs/"

outputs = {
    "llama": {
        "hotpot": {
            1234: [
                "2025-03-15/13-32-58",
                "2025-03-15/13-33-14",
                "2025-03-15/15-47-23",
                "2025-03-15/15-47-36",
            ],
            3782: [
                "2025-03-20/14-25-23",
                "2025-03-20/13-29-33",
                "2025-03-20/12-17-05",
                "2025-03-20/04-05-06",
            ],
            9539: [
                "2025-03-21/23-00-19",
                "2025-03-22/15-59-56",
                "2025-03-22/16-57-18",
                "2025-03-22/18-00-03",
            ]
        },
        "wiki": {
            1234: [
                "2025-03-15/15-39-57",   # Sanad
                "2025-03-15/15-48-39",   # Sanad
                "2025-03-16/03-28-29",   # Sanad
                "2025-03-16/03-31-59",   # Sanad
            ],
            3782: [
                "2025-03-20/20-01-57",
                "2025-03-20/18-53-45",
                "2025-03-20/18-07-48",
                "2025-03-20/17-08-18",
            ],
            9539: [
                "2025-03-23/07-27-31",
                "2025-03-23/13-19-43",
                "2025-03-23/14-24-04",
                "2025-03-23/15-25-33",
            ]
        },
        "musique": {
            1234: [
                "2025-03-16/09-39-26",   # Sanad
                "2025-03-16/09-40-48",   # Sanad
                "2025-03-16/13-27-59",   # Sanad
                "2025-03-16/13-28-49",   # Sanad
            ],
            3782: [
                "2025-03-20/01-38-58",
                "2025-03-20/00-26-04",
                "2025-03-19/23-15-21",
                "2025-03-19/21-43-14",
            ],
            9539: [
                "2025-03-23/06-19-33",
                "2025-03-23/05-11-23",
                "2025-03-23/04-04-19",
                "2025-03-23/02-40-39",
            ]
        }

    },
    "qwen": {
        "hotpot": {
            1234: [
                "2025-03-15/18-00-11",
                "2025-03-15/18-00-41",
                "2025-03-15/20-27-18",
                "2025-03-15/20-28-01",
            ],
            3782: [
                "2025-03-22/09-11-38",   # Sanad
                "2025-03-22/09-11-39",   # Sanad
                "2025-03-22/12-33-49",   # Sanad
                "2025-03-22/12-33-50",   # Sanad
            ],
            9539: [
                "2025-03-21/12-44-10",   # Sanad
                "2025-03-21/12-44-11",   # Sanad
                "2025-03-21/16-04-07",   # Sanad
                "2025-03-21/17-37-15",   # Sanad
            ]
        },
        "wiki": {
            1234: [
                "2025-03-17/01-55-08",
                "2025-03-17/01-55-22",
                "2025-03-16/00-57-17",
                "2025-03-16/00-57-20",
            ],
            3782: [
                "2025-03-18/06-08-55",   # Sanad
                "2025-03-18/06-08-56",   # Sanad
                "2025-03-18/08-55-55",   # Sanad
                "2025-03-18/08-55-56",   # Sanad
            ],
            9539: [
                "2025-03-22/03-00-56",   # Sanad
                "2025-03-22/03-00-57",   # Sanad
                "2025-03-22/05-33-35",   # Sanad
                "2025-03-22/05-33-36",   # Sanad
            ]
        },
        "musique": {
            1234: [
                "2025-03-17/01-49-38",   # Sanad
                "2025-03-17/01-51-39",   # Sanad
                "2025-03-17/06-16-39",   # Sanad
                "2025-03-17/06-17-38",   # Sanad
            ],
            3782: [
                "2025-03-17/19-45-29",   # Sanad
                "2025-03-17/19-45-30",   # Sanad
                "2025-03-18/00-28-05",   # Sanad
                "2025-03-18/00-28-06",   # Sanad
            ],
            9539: [
                "2025-03-21/08-32-04",   # Sanad
                "2025-03-21/08-32-05",   # Sanad
                "2025-03-21/04-19-26",   # Sanad
                "2025-03-21/04-19-27",   # Sanad
            ]
        }
    }
}

#output_path = "outputs/2025-03-17/01-55-08/pred_WikiMultihopQA_Qwen2-7b-Instruct__ContextAwareDecoding__ReAct.json"
#output_path = "outputs/2025-03-05/11-51-57/pred_MuSiQue_LLaMA3-8b-Instruct__DoLa__ReAct.json"
output_path = "outputs/2025-03-19/21-43-14/pred_MuSiQue_LLaMA3-8b-Instruct__Baseline__ReAct.json"
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