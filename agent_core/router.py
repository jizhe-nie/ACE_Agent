from __future__ import annotations
import json
from typing import Any, Dict, List
from ACE_Agent.tools.llm_client import UniversalLLMClient, LLMSettings

class MasterRouter:
    """智能路由：负责意图识别与初步任务画像"""
    
    def analyze_intent(self, prompt: str, history: List[Any], settings: LLMSettings) -> Dict[str, Any]:
        """使用 LLM 识别意图。强制要求返回 JSON。"""
        if not settings.is_configured:
            return {"intent": "NEW_TASK", "reasoning": "LLM 未配置，默认开启新任务"}

        client = UniversalLLMClient(settings)
        history_context = "\n".join([f"{m['role']}: {m['content'][:100]}" for m in history[-3:]])
        
        system_prompt = (
            "你是一个数据科学专家路由。分析用户输入，并决定下一步动作。\n"
            "意图类型：\n"
            "- NEW_TASK: 用户想对新数据集（或提到的内置数据集）进行聚类分析。\n"
            "- FOLLOW_UP: 用户对刚才的结果提问、要求解释、可视化或微调参数。\n"
            "输出格式必须为 JSON: {\"intent\": \"...\", \"target_dataset\": \"...\", \"reasoning\": \"...\"}"
        )
        
        user_input = f"对话历史：\n{history_context}\n\n当前输入：{prompt}"
        res = client.chat_completion([{"role": "user", "content": user_input}], system_prompt)
        
        try:
            # 强化 JSON 提取：处理可能包含的 Markdown 或前后文字
            import re
            json_match = re.search(r'\{.*\}', res.replace('\n', ''), re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                # 强制转大写，防止大小写不匹配导致的逻辑失效
                data["intent"] = str(data.get("intent", "NEW_TASK")).upper().strip()
                return data
            return {"intent": "NEW_TASK", "reasoning": "未能提取有效 JSON 意图"}
        except:
            return {"intent": "FOLLOW_UP" if "?" in prompt or "？" in prompt else "NEW_TASK", "reasoning": "语义解析异常，进入兜底逻辑"}
