import os

from transformers import AutoConfig, AutoTokenizer

model_path = os.environ["QWEN3_32B_FP8"]

tokenizer = AutoTokenizer.from_pretrained(model_path)
config = AutoConfig.from_pretrained(model_path)

print(tokenizer.model_max_length)
print(config.max_position_embeddings)

prompt = "Hello, world" * 100000
tokenizer(prompt)
