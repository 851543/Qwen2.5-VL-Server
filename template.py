from typing import Dict, Any


class DiseaseDiagnosisTemplate:
    """
    疾病诊断模板类，用于定义和验证疾病诊断数据结构。

    模板包含三个主要部分：
    1. 诊断信息（diagnosis）
    2. 快速检查结果（quick_check）
    3. 预防措施和注意事项（suggestions）
    """

    TEMPLATE = {
        "type": "object",
        "properties": {
            "diagnosis": {
                "type": "object",
                "properties": {
                    "confidence": {"type": "integer"},
                    "symptoms": {"type": "string"},
                    "primary_causes": {"type": "string"},
                    "recommended_measures": {"type": "string"}
                },
                "required": ["confidence", "symptoms",
                             "primary_causes", "recommended_measures"]
            },
            "quick_check": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "infection_level": {"type": "integer"},
                    "urgency": {"type": "boolean"}
                },
                "required": ["infection_level",
                             "requires_treatment", "urgency"]
            },
            "suggestions": {
                "type": "object",
                "properties": {
                    "prevention_measures": {"type": "string"},
                    "attention_items": {"type": "string"}
                },
                "required": ["prevention_measures", "attention_items"]
            }
        },
        "required": ["diagnosis", "quick_check",
                     "suggestions"]
    }


def get_template():
    """获取疾病诊断模板"""
    return DiseaseDiagnosisTemplate.TEMPLATE
