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
            
            # Extract predicted paragraphs (text content)
            predicted_paragraphs = extract_predicted_paragraphs(output['prompted_question'][1])
            
            # Extract gold answers
            gold_answers = [ans['spans'][0] for ans in gold_item['answers_objects']]
            gold_paragraphs = [ctx['paragraph_text'] for ctx in gold_item['contexts'] if ctx['is_supporting']]
            
            support_metric([], predicted_paragraphs, [], gold_paragraphs)

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

def extract_predicted_paragraphs(prompted_question) -> List[str]:
    # Extract paragraph texts from reasoning chain observations
    paragraphs = []
    if "Context:" in prompted_question:
        # Extract observation text which contains paragraph content
        observation_text = prompted_question.split("Context")[1].split(":", 1)[1].strip()
        paragraphs.append(observation_text)
    return paragraphs

output_prefix = "outputs/"

outputs = {
    "llama": {
        "hotpot": {
            1234: [
                "2025-03-24/10-49-38",
                "2025-03-24/10-54-25",
                "2025-03-24/11-01-16",
                "2025-03-24/11-05-34",
            ],
            3782: [
                "2025-03-24/11-50-39",
                "2025-03-24/11-55-46",
                "2025-03-24/12-03-12",
                "2025-03-24/12-07-31",
            ],
            9539: [
                "2025-03-24/12-53-24",
                "2025-03-24/12-58-15",
                "2025-03-24/13-05-03",
                "2025-03-24/13-09-18",
            ]
        },
        "wiki": {
            1234: [
                "2025-03-24/10-21-01",   
                "2025-03-24/10-27-25",   
                "2025-03-24/10-37-19",   
                "2025-03-24/10-43-27",   
            ],
            3782: [
                "2025-03-24/11-30-46",
                "2025-03-24/11-35-12",
                "2025-03-24/11-41-41",
                "2025-03-24/11-45-30",
            ],
            9539: [
                "2025-03-24/12-33-33",
                "2025-03-24/12-38-05",
                "2025-03-24/12-44-13",
                "2025-03-24/12-48-09",
            ]
        },
        "musique": {
            1234: [
                "2025-03-24/11-11-25",   
                "2025-03-24/11-16-18",   
                "2025-03-24/11-22-16",   
                "2025-03-24/11-25-54",   
            ],
            3782: [
                "2025-03-24/12-13-50",
                "2025-03-24/12-18-23",
                "2025-03-24/12-24-20",
                "2025-03-24/12-27-59",
            ],
            9539: [
                "2025-03-24/13-15-16",
                "2025-03-24/13-25-36",
                "2025-03-24/13-29-26",
                "2025-03-24/13-19-27",
            ]
        }

    },
    "qwen": {
        "hotpot": {
            1234: [
                "2025-03-24/14-54-40",
                "2025-03-24/15-26-40",
                "2025-03-24/15-41-50",
                "2025-03-24/15-52-00",
            ],
            3782: [
                "2025-03-24/17-52-32",   
                "2025-03-24/18-02-06",   
                "2025-03-24/18-18-09",   
                "2025-03-24/18-28-13",   
            ],
            9539: [
                "2025-03-24/20-34-16",   
                "2025-03-24/20-43-07",   
                "2025-03-24/20-57-58",   
                "2025-03-24/21-07-55",   
            ]
        },
        "wiki": {
            1234: [
                "2025-03-24/14-08-32",
                "2025-03-24/14-16-41",
                "2025-03-24/14-30-41",
                "2025-03-24/14-41-33",
            ],
            3782: [
                "2025-03-24/17-06-23",   
                "2025-03-24/17-23-55",   
                "2025-03-24/17-27-49",   
                "2025-03-24/17-38-29",   
            ],
            9539: [
                "2025-03-24/19-48-24",   
                "2025-03-24/19-56-29",   
                "2025-03-24/20-10-28",   
                "2025-03-24/20-21-35",   
            ]
        },
        "musique": {
            1234: [
                "2025-03-24/16-04-45",   
                "2025-03-24/16-15-41",   
                "2025-03-24/16-33-09",   
                "2025-03-24/16-47-25",   
            ],
            3782: [
                "2025-03-24/18-40-45",   
                "2025-03-24/18-53-30",   
                "2025-03-24/19-12-15",   
                "2025-03-24/19-27-38",   
            ],
            9539: [
                "2025-03-24/21-20-30",   
                "2025-03-24/21-31-14",   
                "2025-03-24/21-48-28",   
                "2025-03-24/22-03-00",   
            ]
        }
    }
}

#output_path = "outputs/2025-03-24/01-55-08/pred_WikiMultihopQA_Qwen2-7b-Instruct__ContextAwareDecoding__ReAct.json"
#output_path = "outputs/2025-03-05/11-51-57/pred_MuSiQue_LLaMA3-8b-Instruct__DoLa__ReAct.json"
output_path = "outputs/2025-03-24/21-43-14/pred_MuSiQue_LLaMA3-8b-Instruct__Baseline__ReAct.json"
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