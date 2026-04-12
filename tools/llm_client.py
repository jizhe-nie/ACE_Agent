from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class LLMSettings:
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    temperature: float = 0.2
    enabled: bool = False

    @property
    def is_configured(self) -> bool:
        return self.enabled and bool(self.base_url.strip()) and bool(self.api_key.strip()) and bool(self.model.strip())


class OpenAICompatibleClient:
    def __init__(self, settings: LLMSettings):
        self.settings = settings

    def summarize_report(self, summary_payload: dict[str, Any]) -> str | None:
        if not self.settings.is_configured:
            return None

        url = f"{self.settings.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
        }
        system_prompt = (
            "你是一个 ACE 聚类分析助手。你的任务有两个：\n"
            "1. 如果输入是聚类报告详情，请用简洁的中文总结。提到优胜算法及其原因，并建议用户接下来可以观察什么。\n"
            "2. 如果输入包含用户针对已有报告的提问（follow_up 类型），请结合已有的聚类结果和指标，给出专业且有说服力的回答。不要编造数据。"
        )
        payload = {
            "model": self.settings.model,
            "temperature": self.settings.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": str(summary_payload)},
            ],
        }
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            message = data["choices"][0]["message"]["content"]
            if isinstance(message, list):
                text_parts = [part.get("text", "") for part in message if isinstance(part, dict)]
                return "\n".join(part for part in text_parts if part).strip() or None
            return str(message).strip() or None
        except Exception:
            return None

