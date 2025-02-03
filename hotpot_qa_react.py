from transformers import AutoTokenizer, AutoModelForCausalLM
from src.frameworks.react import HotpotReAct
from huggingface_hub import login
from transformers import pipeline
import json


pipe = pipeline("text-generation", 
                model="Qwen/Qwen2.5-0.5B"
)
model = lambda x: pipe(
    x,
    max_new_tokens=512, 
    pad_token_id = pipe.tokenizer.eos_token_id,
    return_full_text=False
)
hp_react = HotpotReAct(model=model, max_steps=7)

hotpot_train_path = "src/utils/processed_data/hotpotqa/test_subsampled.jsonl"
n_correct = 0
n_total = 0

with open(hotpot_train_path, 'r', encoding='utf-8') as file:
    for line_number, line in enumerate(file, start=1):
        try:
            sample = json.loads(line)
            question_text = sample["question_text"]
            answers_objects = sample["answers_objects"]
            all_answers = answers_objects[0]["spans"]
            react_answer = hp_react.do_react(question_text)
            if react_answer in all_answers:
                n_correct += 1
            n_total += 1
            print(f"Question: {question_text}")
            print(f"Answer: {all_answers}")
            print(f"ReAct Answer: {react_answer}")
            print(f"Completed {n_total} questions.")
            print(f"EM: {n_correct/n_total}")

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON on line {line_number}: {e}")

print(f"Final EM: {n_correct/n_total}")
