from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import os
load_dotenv()

client = OpenAI(
    api_key=os.environ.get("MOONSHOT_API_KEY"),
    base_url="https://api.moonshot.cn/v1",
)


file_p = './knowledge/algoxy-zh-cn.pdf'

# file_object = client.files.create(file=Path(file_p), purpose="file-extract")
# print(file_object)
# file_list = client.files.list()
# print(file_list)

print(client.files.retrieve(file_id="cni3u94udu62fber7asg"))

file_content = client.files.content(file_id="cni3u94udu62fber7asg").text

print(len(file_content))
messages=[
    {
        "role": "system",
        "content": "你是 Kimi，由 Moonshot AI 提供的人工智能助手，你更擅长中文和英文的对话。你会为用户提供安全，有帮助，准确的回答。同时，你会拒绝一些涉及恐怖主义，种族歧视，黄色暴力等问题的回答。Moonshot AI 为专有名词，不可翻译成其他语言。",
    },
    {
        "role": "system",
        "content": file_content[:len(file_content)//5],
    },
    {"role": "user", "content": "请根据这个文档生成一篇关于算法的文章，要求格式是markdown，题目是对内容的吸引人的总结。"},
]

# 然后调用 chat-completion, 获取 kimi 的回答
completion = client.chat.completions.create(
  model="moonshot-v1-32k",
  messages=messages,
  temperature=0.3,
)

print(completion.choices[0].message)