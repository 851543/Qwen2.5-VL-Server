from modelscope import snapshot_download

import os

now_dir = os.getcwd()

model_dir = snapshot_download("Qwen/Qwen2.5-VL-3B-Instruct", cache_dir=f'{now_dir}\\model')