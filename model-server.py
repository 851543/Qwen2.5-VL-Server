import os
from swift.llm import InferEngine, InferRequest, PtEngine, RequestConfig

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from fastapi import FastAPI, UploadFile, Form, File

app = FastAPI()

from io import BytesIO
from swift.plugin import InferStats
from PIL import Image


def infer_stream(engine: 'InferEngine', infer_request: 'InferRequest'):
    request_config = RequestConfig(max_tokens=128, temperature=0, stream=True)
    metric = InferStats()
    gen = engine.infer([infer_request], request_config, metrics=[metric])

    for resp_list in gen:
        if resp_list[0] is None:
            continue
        print(resp_list[0].choices[0].delta.content, end='', flush=True)


if __name__ == '__main__':
    prompt_text = "描述这张图片"
    image = Image.open(f"{os.getcwd()}/input-image/NSHM_PHOTO_2024_8_19_23_33_19.jpg")
    print(image.width,image.height)
    new_width = 800
    scale_ratio = new_width / image.width
    new_height = int(image.height * scale_ratio)
    print(image.resize((new_width, new_height), Image.LANCZOS))
    # model = rf'{os.getcwd()}/model/Qwen/Qwen2___5-VL-3B-Instruct'  # m\模型路径
    # engine = PtEngine(model, max_batch_size=64)
    # messages = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {
    #                 "type": "image",
    #                 "image": image.resize((new_width, new_height), Image.LANCZOS)
    #             },
    #             {"type": "text", "text": f"{prompt_text}"},
    #         ],
    #     }
    # ]
    # infer_stream(engine, InferRequest(messages=messages))
