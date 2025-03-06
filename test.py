import requests


def get_ai_analysis_params(name, type_flag="plant"):
    """
    获取AI分析参数
    :param name: 作物名称或鱼类名称
    :param type_flag: 类型标志 "plant"(种植) 或 "fish"(养殖)
    :return: AI分析结果
    """

    # 根据类型选择不同的专家角色和参数模板
    if type_flag == "fish":
        system_role = "水产养殖专家"
        params_template = {
            "环境参数": {
                "水温(℃)": ["标准范围", "说明"],
                "pH值": ["标准范围", "说明"],
                "溶解氧(mg/L)": ["标准范围", "说明"],
                "氨氮(mg/L)": ["标准范围", "说明"],
                "亚硝酸盐(mg/L)": ["标准范围", "说明"]
            },
            "建议": [
                {"类型": "水质管理", "内容": "具体建议"},
                {"类型": "投喂管理", "内容": "具体建议"},
                {"类型": "疾病防控", "内容": "具体建议"},
                {"类型": "环境监控", "内容": "具体建议"}
            ],
            "转化率字段": "饲料转化率"
        }
        format_params = {
            "water_temperature": {"type": "array", "items": {"type": "string"}},
            "ph": {"type": "array", "items": {"type": "string"}},
            "dissolved_oxygen": {"type": "array", "items": {"type": "string"}},
            "ammonia_nitrogen": {"type": "array", "items": {"type": "string"}},
            "nitrite": {"type": "array", "items": {"type": "string"}}
        }
    else:
        system_role = "农业种植专家"
        params_template = {
            "环境参数": {
                "温度(°C)": ["标准范围", "说明"],
                "湿度(%)": ["标准范围", "说明"],
                "光照(lux)": ["标准范围", "说明"],
                "风向": ["标准范围", "说明"],
                "风速(m/s)": ["标准范围", "说明"],
                "pH值": ["标准范围", "说明"]
            },
            "建议": [
                {"类型": "水分管理", "内容": "具体建议"},
                {"类型": "肥料管理", "内容": "具体建议"},
                {"类型": "病虫害防治", "内容": "具体建议"},
                {"类型": "环境监控", "内容": "具体建议"}
            ],
            "转化率字段": "肥料转化率"
        }
        format_params = {
            "temperature": {"type": "array", "items": {"type": "string"}},
            "humidity": {"type": "array", "items": {"type": "string"}},
            "light": {"type": "array", "items": {"type": "string"}},
            "wind_direction": {"type": "array", "items": {"type": "string"}},
            "wind_speed": {"type": "array", "items": {"type": "string"}},
            "ph": {"type": "array", "items": {"type": "string"}}
        }

    data = {
        "model": "deepseek-r1:14b",
        "messages": [
            {
                "role": "system",
                "content": f"""
                你是一名{system_role}。请提供{name}的具体{'养殖' if type_flag == 'fish' else '种植'}参数范围。
                中文说明、严格按照格式返回数据类型。
                请按以下格式输出：
                {{
                    "环境参数": {params_template["环境参数"]},
                    "{'养殖' if type_flag == 'fish' else '种植'}建议": {params_template["建议"]},
                    "核心指标": {{
                        "生长速度": "百分比",
                        "抗病能力": "百分比",
                        "{params_template['转化率字段']}": "百分比",
                        "市场认可度": "百分比"
                    }},
                    "综合评估": {{
                        "生长评估": "详细说明",
                        "{'养殖' if type_flag == 'fish' else '种植'}难度": "详细说明",
                        "综合建议": "详细说明",
                        "市场分析": "详细说明"
                    }}
                }}
                """
            },
            {
                "role": "user",
                "content": name
            }
        ],
        "stream": False,
        "format": {
            "type": "object",
            "properties": {
                "environmental_params": {
                    "type": "object",
                    "properties": format_params
                },
                f"{'breeding' if type_flag == 'fish' else 'planting'}_suggestions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string"},
                            "content": {"type": "string"}
                        }
                    }
                },
                "core_indicators": {
                    "type": "object",
                    "properties": {
                        "growth_rate": {"type": "integer"},
                        "disease_resistance": {"type": "integer"},
                        f"{'feed' if type_flag == 'fish' else 'fertilizer'}_conversion": {"type": "integer"},
                        "market_acceptance": {"type": "integer"}
                    }
                },
                "comprehensive_assessment": {
                    "type": "object",
                    "properties": {
                        "growth_assessment": {"type": "string"},
                        f"{'breeding' if type_flag == 'fish' else 'cultivation'}_difficulty": {"type": "string"},
                        "general_recommendations": {"type": "string"},
                        "market_analysis": {"type": "string"}
                    }
                }
            },
            "required": ["environmental_params", f"{'breeding' if type_flag == 'fish' else 'planting'}_suggestions",
                         "core_indicators", "comprehensive_assessment"]
        }
    }

    response = requests.post("http://192.168.1.242:11434/api/chat", json=data)
    return response.json()


# 使用示例
# 种植示例
plant_result = get_ai_analysis_params("小白菜", "plant")
print("种植分析结果:", plant_result["message"]["content"])

# 养殖示例
fish_result = get_ai_analysis_params("淡水鱼草鱼", "fish")
print("养殖分析结果:", fish_result["message"]["content"])