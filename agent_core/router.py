from __future__ import annotations
import json
from typing import Any, Dict, List
from ACE_Agent.tools.llm_client import UniversalLLMClient, LLMSettings

class MasterRouter:
    """智能路由：负责意图识别与初步任务画像。

    识别三种意图：
    - NEW_TASK      : 在当前数据集上执行一次聚类实验
    - FOLLOW_UP     : 解释结果、理论提问、追问已有结果
    - CODE_EXAMPLE  : 用户要代码本身，不是要分析结果

    兜底：router 异常时 fallback 到 FOLLOW_UP（而非 NEW_TASK），
    避免误触发昂贵的新任务流程。
    """

    def analyze_intent(self, prompt: str, history: List[Any], settings: LLMSettings) -> Dict[str, Any]:
        """使用 LLM 识别意图。强制要求返回 JSON。"""
        if not settings.is_configured:
            return {"intent": "NEW_TASK", "reasoning": "LLM 未配置，默认开启新任务"}

        client = UniversalLLMClient(settings)
        history_context = "\n".join([f"{m['role']}: {m['content'][:100]}" for m in history[-3:]])

        system_prompt = (
            "你是一个数据科学专家路由，必须将用户输入归类为以下三种意图之一，并以 JSON 返回。\n\n"
            "## 意图定义与典型例子\n\n"
            "### NEW_TASK（在当前数据集上执行新的聚类分析）\n"
            "触发条件：用户想**运行**聚类实验、获取分析结果。\n"
            "典型例子：\n"
            "  - '帮我分析这个数据集'\n"
            "  - '用 DBSCAN 分析我的数据'\n"
            "  - '换 KMeans 重跑一遍'\n"
            "  - '分析月牙数据'\n"
            "  - '对这份数据做聚类'\n\n"
            "### FOLLOW_UP（解释结果、理论提问、追问已有结果）\n"
            "触发条件：用户在问已有结果的含义、理论知识，或追问上一次分析，不需要重跑算法。\n"
            "典型例子：\n"
            "  - '轮廓系数是什么意思？'\n"
            "  - '为什么 DBSCAN 在月牙数据上更好？'\n"
            "  - '第二名算法和最优算法差距大吗？'\n"
            "  - '请解释这个结果'\n\n"
            "### CODE_EXAMPLE（用户要代码本身，不是要分析结果）\n"
            "触发条件：用户明确索要**代码**，即使提到了算法名称。核心判据是：\n"
            "  用户想拿到一段代码去自己用，而不是让 Agent 运行实验。\n"
            "典型触发词：'给我示例'、'完整可运行示例'、'写一段代码'、'example code'、\n"
            "  'show me the code'、'贴一下代码'、'代码怎么写'、'生成代码'、'帮我写代码'。\n"
            "典型例子：\n"
            "  - '帮我生成 SpectralClustering 在月牙数据上的完整可运行示例（含绘图与指标）'\n"
            "  - '写一段 DBSCAN 聚类的 Python 代码'\n"
            "  - 'show me example code for KMeans'\n"
            "  - '贴一下 AgglomerativeClustering 的代码'\n\n"
            "## 关键区分规则\n"
            "- 提到算法名 + 要**执行/分析/结果** → NEW_TASK\n"
            "- 提到算法名 + 要**代码/示例/怎么写** → CODE_EXAMPLE\n"
            "- 只是提问/追问 → FOLLOW_UP\n"
            "- 不确定时：优先判为 FOLLOW_UP（而非 NEW_TASK），以避免误触发实验。\n\n"
            "输出格式必须为 JSON: {\"intent\": \"NEW_TASK|FOLLOW_UP|CODE_EXAMPLE\", "
            "\"target_dataset\": \"...\", \"reasoning\": \"...\"}"
        )

        user_input = f"对话历史：\n{history_context}\n\n当前输入：{prompt}"
        res = client.chat_completion([{"role": "user", "content": user_input}], system_prompt)

        try:
            import re
            json_match = re.search(r'\{.*\}', res.replace('\n', ''), re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                # 强制转大写，防止大小写不匹配导致的逻辑失效
                raw_intent = str(data.get("intent", "FOLLOW_UP")).upper().strip()
                # 仅接受已知意图；未知值 fallback 到 FOLLOW_UP
                if raw_intent not in {"NEW_TASK", "FOLLOW_UP", "CODE_EXAMPLE"}:
                    raw_intent = "FOLLOW_UP"
                data["intent"] = raw_intent
                return data
            return {"intent": "FOLLOW_UP", "reasoning": "未能提取有效 JSON 意图，兜底为 FOLLOW_UP"}
        except Exception:
            # 兜底：router 异常时 fallback 到 FOLLOW_UP 而非 NEW_TASK，
            # 避免误触发昂贵的新任务流程
            return {
                "intent": "FOLLOW_UP" if ("?" in prompt or "？" in prompt) else "FOLLOW_UP",
                "reasoning": "语义解析异常，安全兜底为 FOLLOW_UP",
            }
