from jinja2 import Environment, FileSystemLoader

env = Environment(
    loader=FileSystemLoader("../../examples/chat_template"),
    trim_blocks=True,
    lstrip_blocks=True,
)
template = env.get_template("tool_chat_template_deepseekv31.jinja")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "4"},
]

output = template.render(
    messages=messages, bos_token="<bos>", add_generation_prompt=True, thinking=True
)
print(output)
