import json

import requests

url = "http://127.0.0.1:8000/v1/chat/completions"

headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer bzl_ascend_vllm_79%ktRH+mS$Kh@KP$MUcIghR",
}

content = "你是一个高准确率的智能职位审核系统，任务是判断职位是否存在“特殊行业职类逃逸”行为。  ## 一、定义 - **职类逃逸**：职位名称、描述或关键词表明该岗位属于某个特殊行业，"


data = {
    "model": "Qwen3-32B",
    "messages": [{"role": "user", "content": content}],
    "top_k": -1,
    "top_p": 0.95,
    "temperature": 0.6,
    "ignore_eos": False,
    "stream": False,
    "chat_template_kwargs": {"enable_thinking": False},
}

response = requests.post(url, headers=headers, json=data)

if response.status_code == 200:
    result = response.json()
    print(json.dumps(result, ensure_ascii=False, indent=2))
else:
    print(f"请求失败，状态码: {response.status_code}")
    print(response.text)
