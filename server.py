import os
import re
from typing import Literal
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from swift.llm import PtEngine, RequestConfig, BaseArguments, InferRequest, safe_snapshot_download, draw_bbox, \
    load_image

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from fastapi import FastAPI, UploadFile, Form, File
from PIL import Image
from pathlib import Path
from io import BytesIO
import os

new_dir = os.getcwd()

app = FastAPI()

input_folder = Path("input-image")

model_path = rf'{new_dir}/model/Qwen/Qwen2___5-VL-3B-Instruct'  # m\模型路径

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto", device_map="auto")


@app.post("/chat")
async def chat(text: str = Form(), prompt_image: UploadFile = File()):
    # if not input_folder.exists():
    #     # 如果不存在则创建
    #     input_folder.mkdir(parents=True, exist_ok=True)
    # file_path = input_folder / prompt_image.filename
    # contents = await prompt_image.read()
    # with open(file_path, "wb") as file:
    #     file.write(contents)
    # image = load_image(str(file_path))
    # 创建BytesIO缓冲区
    image_buffer = BytesIO()
    # 读取文件内容到缓冲区
    contents = await prompt_image.read()
    image_buffer.write(contents)
    image_buffer.seek(0)
    # 使用PIL打开图片
    image = Image.open(image_buffer)
    new_width = 800
    scale_ratio = new_width / image.width
    new_height = int(image.height * scale_ratio)
    # infer_grounding() # 需要微调
    # file_path.unlink()
    return ask(text, image.resize((new_width, new_height), Image.LANCZOS))


def ask(question: str, resized_image):
    processor = AutoProcessor.from_pretrained(model_path)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {"type": "text", "text": question},
            ],
        }
    ]
    text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=[text_prompt], images=[resized_image], padding=True, return_tensors="pt",stream=True)
    inputs = inputs.to('cuda')

    # 推理
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text


# def draw_bbox_qwen2_vl(resized_image, response, norm_bbox: Literal['norm1000', 'none']):
#     matches = re.findall(
#         r'<\|object_ref_start\|>(.*?)<\|object_ref_end\|><\|box_start\|>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|>',
#         response)
#     ref = []
#     bbox = []
#     for match_ in matches:
#         ref.append(match_[0])
#         bbox.append(list(match_[1:]))
#     draw_bbox(resized_image, ref, bbox, norm_bbox=norm_bbox)


# def infer_grounding():
#     output_path = 'test9.jpeg'
#     infer_request = InferRequest(messages=[{'role': 'user', 'content': 'Task: Object Detection'}], images=[resized_image])
#     request_config = RequestConfig(max_tokens=512, temperature=0)
#     adapter_path = safe_snapshot_download(r'D:\gpt-code\checkpoint-123')
#     args = BaseArguments.from_pretrained(adapter_path)
#     engine = PtEngine(model_path, adapters=[adapter_path])
#     resp_list = engine.infer([infer_request], request_config)
#     response = resp_list[0].choices[0].message.content
#     print(f'lora-response: {response}')
#     draw_bbox_qwen2_vl(resized_image, response, norm_bbox=args.norm_bbox)
#     print(f'output_path: {output_path}')
#     resized_image.save(output_path)


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=10095)
