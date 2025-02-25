import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def infer_stream(engine: 'InferEngine', infer_request: 'InferRequest'):
    request_config = RequestConfig(max_tokens=128, temperature=0, stream=True)
    metric = InferStats()
    gen = engine.infer([infer_request], request_config, metrics=[metric])
    query = infer_request.messages[0]['content']
    print(f'query: {query}\nresponse: ', end='')
    for resp_list in gen:
        if resp_list[0] is None:
            continue
        print(resp_list[0].choices[0].delta.content, end='', flush=True)
    print()
    print(f'metric: {metric.compute()}')


if __name__ == '__main__':
    from swift.llm import InferEngine, InferRequest, PtEngine, RequestConfig
    from swift.plugin import InferStats
    from PIL import Image
    model = rf'{os.getcwd()}/model/Qwen/Qwen2___5-VL-3B-Instruct'  # m\模型路径
    engine = PtEngine(model, max_batch_size=64)
    import os
    image = Image.open(f"{os.getcwd()}/input-image/NSHM_PHOTO_2024_7_28_00_11_24.jpg")
    new_width = 800
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
                {"type": "text", "text": "动物是什么"},
            ],
        }
    ]
    infer_stream(engine, InferRequest(messages=messages))
