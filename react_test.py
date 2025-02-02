from transformers import AutoTokenizer, AutoModelForCausalLM
from src.frameworks.react import HotpotReAct
from huggingface_hub import login
from transformers import pipeline


#tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
#model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
#pipe = lambda x: model.generate(tokenizer(x))
pipe = pipeline("text-generation", 
                model="Qwen/Qwen2.5-0.5B"
)
model = lambda x: pipe(
    x,
    max_new_tokens=256, 
    pad_token_id = pipe.tokenizer.eos_token_id,
    return_full_text=False
)
model("Hello llama")
hp_react = HotpotReAct(model=model, max_steps=5)
hp_react.do_react("Which dog's ancestors include Gordon and Irish Setters: the Manchester Terrier or the Scotch Collie?")