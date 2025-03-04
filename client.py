import requests

response = requests.post(
    'http://127.0.0.1:10095/chat',
    data={"prompt_text": "你是谁"},
    stream=True
)

for line in response.iter_lines(decode_unicode=True):
    if line:
        print(f"接收到事件流数据: {line}")