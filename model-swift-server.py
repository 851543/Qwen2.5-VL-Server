import os
from io import BytesIO

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from swift.llm import InferRequest, PtEngine, RequestConfig, load_image
from swift.plugin import InferStats

model = rf'{os.getcwd()}/model/Qwen/Qwen2___5-VL-3B-Instruct'  # m\模型路径
engine = PtEngine(model, max_batch_size=64)
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import StreamingResponse

app = FastAPI()

from PIL import Image

from typing import Optional
import json

from template import get_template

import re


# 病虫害识别
@app.post("/identify")
async def identify(prompt_text: str = Form(), prompt_image: Optional[UploadFile] = File(None)):
    # 处理图片
    image_buffer = BytesIO()
    contents = await prompt_image.read()
    image_buffer.write(contents)
    image = load_image(image_buffer)

    # 调整图片大小
    new_width = 800 if image.width >= 800 else 400
    scale_ratio = new_width / image.width
    new_height = int(image.height * scale_ratio)

    # 构建请求内容
    content = [
        {
            "type": "image",
            "image": image.resize((new_width, new_height), Image.LANCZOS)
        },
        {"type": "text", "text": f"{prompt_text},请帮我分析这张图片中的病虫害情况,需要包含以下信息: 主要病害诊断:诊断可信度(百分之几)、症状概述、主要原因、建议措施 "
                                 f"快速诊断:病名、0（低）|1（中）|2（高）感染、是否需要处理 你的建议:预防建议、注意事项。返回JSON数据格式{get_template()}"}
    ]

    messages = [
        {
            "role": "user",
            "content": content
        }
    ]

    try:
        # 使用非流式响应获取完整结果
        request_config = RequestConfig(max_tokens=512, temperature=0, stream=False)
        response = engine.infer([InferRequest(messages=messages)], request_config)[0]
        # 获取原始响应内容
        content = response.choices[0].message.content
        # 合并前后缀清理操作
        cleaned_content = re.sub(r'^.*```json\n*\n*|```[^\n]*\n*$', '', content, flags=re.DOTALL)
        # 解析JSON
        return json.loads(cleaned_content)
    finally:
        image_buffer.close()


# 聊天
@app.post("/chat")
async def chat(prompt_text: str = Form(), prompt_image: Optional[UploadFile] = File(None)):
    image_buffer = None
    content = [
        {"type": "text", "text": f"{prompt_text}"},
    ]
    if prompt_image is not None:
        image_buffer = BytesIO()
        # 读取文件内容到缓冲区
        contents = await prompt_image.read()
        image_buffer.write(contents)
        image = load_image(image_buffer)
        new_width = 800 if image.width >= 800 else 400
        scale_ratio = new_width / image.width
        new_height = int(image.height * scale_ratio)
        content = [
            {
                "type": "image",
                "image": image.resize((new_width, new_height), Image.LANCZOS)
            },
            {"type": "text", "text": f"{prompt_text}"},
        ]

    messages = [
        {
            "role": "user",
            "content": content
        }
    ]
    return StreamingResponse(infer_stream(InferRequest(messages=messages), image_buffer),
                             media_type="text/event-stream",
                             headers={
                                 "Cache-Control": "no-cache",
                                 "Connection": "keep-alive",
                                 "X-Accel-Buffering": "no",
                                 "Content-Type": "text/event-stream"
                             })


def infer_stream(infer_request: 'InferRequest', image_buffer):
    request_config = RequestConfig(max_tokens=512, temperature=0, stream=True)
    metric = InferStats()
    try:
        gen = engine.infer([infer_request], request_config, metrics=[metric])
        for resp_list in gen:
            if resp_list[0] is None:
                continue
            print(resp_list[0].choices[0])
            # 格式化为SSE事件流格式
            data = {
                "role": resp_list[0].choices[0].delta.role,
                "response": resp_list[0].choices[0].delta.content,
                "finish_reason": resp_list[0].choices[0].finish_reason
            }
            yield f"{json.dumps(data)}\n\n"
    finally:
        if image_buffer is not None:
            image_buffer.close()


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=10095)
