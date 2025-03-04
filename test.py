import requests
import json
import time

text = "淡水鱼鲫鱼"

data = {
    "model": "deepseek-r1:14b",
    "messages": [
        {
            "role": "system",
            "content": """
            你是一名水产养殖专家。请提供鱼种的具体养殖参数范围。
            请用JSON格式输出，包含以下参数：
            - 水温范围（°C）
            - pH值范围
            - 溶解氧含量范围（mg/L）
            - 氨氮允许浓度范围（mg/L）
            - 亚硝酸盐允许浓度范围（mg/L）
            """
        },
        {
            "role": "user",
            "content": text
        }
    ],
    "stream": False,
    "format": {
        "type": "object",
        "properties": {
            "water_temperature_range": {
                "type": "array",
                "items": {
                    "type": "number"
                },
                "minItems": 2,
                "maxItems": 2
            },
            "ph_range": {
                "type": "array",
                "items": {
                    "type": "number"
                },
                "minItems": 2,
                "maxItems": 2
            },
            "dissolved_oxygen_range": {
                "type": "array",
                "items": {
                    "type": "number"
                },
                "minItems": 2,
                "maxItems": 2
            },
            "ammonia_nitrogen_range": {
                "type": "array",
                "items": {
                    "type": "number"
                },
                "minItems": 2,
                "maxItems": 2
            },
            "nitrite_range": {
                "type": "array",
                "items": {
                    "type": "number"
                },
                "minItems": 2,
                "maxItems": 2
            }
        },
        "required": [
            "water_temperature_range",
            "ph_range",
            "dissolved_oxygen_range",
            "ammonia_nitrogen_range",
            "nitrite_range"
        ]
    }
}

response = requests.post("http://192.168.1.242:11434/api/chat", json=data)

# 解析JSON响应
result = response.json()
print(result)
