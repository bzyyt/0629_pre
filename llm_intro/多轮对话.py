import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"), base_url=os.getenv("DEEPSEEK_BASE_URL")
)

messages = [{"role": "system", "content": "You are a helpful assistant."}]

while True:
    try:
        user_input = input("输入：")
        messages.append({"role": "user", "content": user_input})

        response = client.chat.completions.create(
            model=os.getenv("DEEPSEEK_MODEL"),
            messages=messages,
        )

        assistant_response = response.choices[0].message.content
        messages.append({"role": "assistant", "content": assistant_response})
        print(f"输出：{assistant_response}")
    except EOFError:
        break
