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
from pathlib import Path

input_folder = Path("input-image")

c
@app.post("/chat")
async def chat(prompt_text: str = Form(), prompt_image: UploadFile = File()):
    image_buffer = BytesIO()
    # 读取文件内容到缓冲区
    contents = await prompt_image.read()
    image_buffer.write(contents)
    image = load_image(image_buffer)
    new_width = 800 if image.width >= 800 else 400
    scale_ratio = new_width / image.width
    new_height = int(image.height * scale_ratio)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image.resize((new_width, new_height), Image.LANCZOS)
                },
                {"type": "text", "text": f"{prompt_text}"},
            ],
        }
    ]
    return StreamingResponse(
        infer_stream(InferRequest(messages=messages), image_buffer),
        media_type="text/plain"
    )


def infer_stream(infer_request: 'InferRequest', image_buffer):
    request_config = RequestConfig(max_tokens=128, temperature=0, stream=True)
    metric = InferStats()
    try:
        gen = engine.infer([infer_request], request_config, metrics=[metric])
        for resp_list in gen:
            if resp_list[0] is None:
                continue
            yield str(resp_list[0])
    finally:
        image_buffer.close()


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=10095)
