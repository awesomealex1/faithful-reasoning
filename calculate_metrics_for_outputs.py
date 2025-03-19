from typing import Dict, List, Tuple, Any
import json
from src.metrics.ircot_metrics.support_em_f1 import SupportEmF1Metric
from src.metrics.ircot_metrics.answer_support_recall import AnswerSupportRecallMetric

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
    
    for output in outputs:
        # Get the question ID from the output
        question_id = output['idx']
        
        if question_id in gold_dict:
            gold_item = gold_dict[question_id]
            
            # Extract predicted supporting paragraphs/titles
            # This would depend on your specific output format
            # Assuming your reasoning chain actions identify supporting paragraphs
            predicted_support = extract_predicted_support(output['reasoning_chain'])
            
            # Extract gold supporting paragraphs/titles
            gold_support = [ctx['title'] for ctx in gold_item['contexts'] if ctx['is_supporting']]
            
            # Calculate support metrics (precision, recall, F1, EM)
            support_metric(predicted_support, gold_support)
            
            # Extract predicted paragraphs (text content)
            predicted_paragraphs = extract_predicted_paragraphs(output['reasoning_chain'])
            
            # Extract gold answers
            gold_answers = [ans['spans'][0] for ans in gold_item['answers_objects']]
            
            # Calculate answer support recall
            answer_recall_metric(predicted_paragraphs, gold_answers)
    
    # Get the calculated metrics
    support_metrics = support_metric.get_metric(reset=True)
    answer_recall_metrics = answer_recall_metric.get_metric(reset=True)
    
    # Combine metrics
    return {**support_metrics, **answer_recall_metrics}

def extract_predicted_support(reasoning_chain: List[str]) -> List[str]:
    # Extract the titles of supporting paragraphs from reasoning chain
    # This is a placeholder - you'll need to implement based on your specific format
    titles = []
    for step in reasoning_chain:
        if "Action" in step and "Search[" in step:
            # Extract search queries which might correspond to titles
            query = step.split("Search[")[1].split("]")[0]
            titles.append(query)
    return titles

def extract_predicted_paragraphs(reasoning_chain: List[str]) -> List[str]:
    # Extract paragraph texts from reasoning chain observations
    paragraphs = []
    for i, step in enumerate(reasoning_chain):
        if "Observation" in step:
            # Extract observation text which contains paragraph content
            observation_text = step.split("Observation")[1].split(":", 1)[1].strip()
            paragraphs.append(observation_text)
    return paragraphs

output_path = "outputs/2025-03-17/01-55-08/pred_WikiMultihopQA_Qwen2-7b-Instruct__ContextAwareDecoding__ReAct.json"
questions_path = "data/2WikiMultiHopQA/test_subsampled.jsonl"
print(calculate_metrics(output_path, questions_path))