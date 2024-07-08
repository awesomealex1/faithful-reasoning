# Project updates

# Week 2: July 8th - July 13th

- Further literature review on retrieval heads, induction heads, and ROME
- Baseline experiments on TruthfulQA
- Preliminary run on DeCoRe


# Week 1: July 1st - July 5th

- Onboarded!
- Setting up the machine and connection to external server for experimenting with HuggingFace
- Project definition -> DeCoRe: Decoding by Contrasting Retrieval Heads to Mitigate Hallucination
- Literature Review: CAD, DoLa, ICD
- Selected datasets for benchmarking:
  - TruthfulQA (MC1, MC2, MC3)
  - TruthfulQA (Open Ended)
  - FACTOR
  - HaluEval (QA, Summ, Dial)
  - MemoTrap
  - NQ Swap
- Generated retrieval heads using the [retrieval head code](https://github.com/nightdessert/Retrieval_Head)
  - Stored Llama3 8b, Llama3 70b, Llama3 8b instruct, Llama3 70b instruct, Mistral 7b, Mistral 7b instruct in [retriever_heads](retriever_heads/) folder.