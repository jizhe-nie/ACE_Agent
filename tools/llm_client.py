from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import requests
import json


@dataclass
class LLMSettings:
    provider: str = "DeepSeek"
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    temperature: float = 0.2
    enabled: bool = True

    @property
    def is_configured(self) -> bool:
        return self.enabled and bool(self.base_url.strip()) and bool(self.api_key.strip()) and bool(self.model.strip())


class UniversalLLMClient:
    """通用 LLM 客户端适配器，支持多供应商协议"""
    
    def __init__(self, settings: LLMSettings):
        self.settings = settings

    def chat_completion(self, messages: List[Dict[str, str]], system_prompt: Optional[str] = None) -> str | None:
        if not self.settings.is_configured:
            return "Error: LLM not configured."

        url = f"{self.settings.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
        }
        
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        payload = {
            "model": self.settings.model,
            "temperature": self.settings.temperature,
            "messages": full_messages,
        }

        # 特殊处理：Gemini 的 OpenAI 兼容模式有时需要不同的路径或参数
        # 这里预留扩展空间
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            
            # 统一提取内容
            content = data["choices"][0]["message"]["content"]
            return str(content).strip() if content else None
        except Exception as e:
            return f"Error during LLM call: {str(e)}"

    def summarize_report(self, summary_payload: dict[str, Any]) -> str | None:
        """旧接口兼容逻辑，内部调用新的 chat_completion"""
        system_prompt = (
            "你是一个 ACE 聚类分析助手。你的任务是根据聚类实验结果生成专业的中文总结。\n"
            "关键要求：\n"
            "1. 必须优先响应用户意图 (user_intent)。如果用户要求使用特定算法，你的总结应重点围绕该算法的表现展开。\n"
            "2. 如果用户指定的算法表现不如其他算法（如 KMeans），应客观指出，并简要对比差异，但不能忽略用户的要求。\n"
            "3. 提到优胜算法 (best_algo) 及其评分，并给出后续行动建议。\n"
            "4. 保持专业、简洁、有说服力。不要编造数据。"
        )
        messages = [{"role": "user", "content": json.dumps(summary_payload, ensure_ascii=False)}]
        return self.chat_completion(messages, system_prompt)

# 为了兼容性，保留旧类名但重定向到新逻辑
class OpenAICompatibleClient(UniversalLLMClient):
    pass
