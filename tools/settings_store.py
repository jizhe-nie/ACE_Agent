from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

SETTINGS_PATH = Path(__file__).resolve().parents[1] / ".ace_demo_config.json"


class SettingsStore:
    def __init__(self, path: Path = SETTINGS_PATH):
        self.path = path
        self._cache = self._load()

    def _load(self) -> dict[str, Any]:
        settings = {}
        # 1. 从 .env 或环境变量加载默认值
        settings["llm_base_url"] = os.getenv("ACE_LLM_BASE_URL", "")
        settings["llm_api_key"] = os.getenv("ACE_LLM_API_KEY", "")
        settings["llm_model"] = os.getenv("ACE_LLM_MODEL", "gpt-3.5-turbo")
        settings["llm_enabled"] = os.getenv("ACE_LLM_ENABLED", "false").lower() == "true"

        # 2. 从 JSON 文件覆盖（持久化的 UI 设置）
        if self.path.exists():
            try:
                json_data = json.loads(self.path.read_text(encoding="utf-8"))
                settings.update(json_data)
            except (json.JSONDecodeError, OSError):
                pass
        return settings

    def get(self, key: str, default: Any = None) -> Any:
        return self._cache.get(key, default)

    def save(self, payload: dict[str, Any]) -> None:
        self._cache.update(payload)
        self.path.write_text(
            json.dumps(self._cache, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


# 为了兼容旧代码，保留原有的函数
def load_settings() -> dict[str, Any]:
    return SettingsStore()._cache


def save_settings(payload: dict[str, Any]) -> None:
    SettingsStore().save(payload)
