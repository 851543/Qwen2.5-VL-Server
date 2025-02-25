from swift.llm import InferRequest, InferClient, RequestConfig
from swift.plugin import InferStats


engine = InferClient(host='127.0.0.1', port=8000)
print(f'models: {engine.models}')
metric = InferStats()
request_config = RequestConfig(max_tokens=512, temperature=0)

# 这里使用了3个infer_request来展示batch推理
# 支持传入本地路径、base64和url
infer_requests = [
    InferRequest(messages=[{'role': 'user', 'content': 'who are you?'}]),
    InferRequest(messages=[{'role': 'user', 'content': '<image><image>两张图的区别是什么？'}],
                 images=['http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png',
                         'http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png']),
    InferRequest(messages=[{'role': 'user', 'content': '<video>describe the video'}],
                 videos=['https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/baby.mp4']),
]

resp_list = engine.infer(infer_requests, request_config, metrics=[metric])
print(f'response0: {resp_list[0].choices[0].message.content}')
print(f'response1: {resp_list[1].choices[0].message.content}')
print(f'response2: {resp_list[2].choices[0].message.content}')
print(metric.compute())
metric.reset()

# base64
import base64
import requests
resp = requests.get('https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/baby.mp4')
base64_encoded = base64.b64encode(resp.content).decode('utf-8')
messages = [{'role': 'user', 'content': [
    {'type': 'video', 'video': f'data:video/mp4;base64,{base64_encoded}'},
    {'type': 'text', 'text': 'describe the video'}
]}]
infer_request = InferRequest(messages=messages)
request_config = RequestConfig(max_tokens=512, temperature=0, stream=True)
gen = engine.infer([infer_request], request_config, metrics=[metric])
print(f'response0: ', end='')
for chunk_list in gen:
    chunk = chunk_list[0]
    if chunk is None:
        continue
    print(chunk.choices[0].delta.content, end='', flush=True)
print()
print(metric.compute())