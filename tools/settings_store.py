from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List
from dataclasses import is_dataclass, asdict
import numpy as np
from dotenv import load_dotenv

load_dotenv()

SETTINGS_PATH = Path(__file__).resolve().parents[1] / ".ace_demo_config.json"
SESSIONS_PATH = Path(__file__).resolve().parents[1] / ".ace_sessions.json"

DEFAULT_PROVIDERS = {
    "DeepSeek": {"base_url": "https://api.deepseek.com", "models": ["deepseek-chat", "deepseek-reasoner"], "icon": "🌊"},
    "DashScope": {"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "models": ["qwen-plus", "qwen-max"], "icon": "☁️"},
    "Moonshot": {"base_url": "https://api.moonshot.cn/v1", "models": ["moonshot-v1-8k"], "icon": "🌙"},
    "OpenAI": {"base_url": "https://api.openai.com/v1", "models": ["gpt-4o", "gpt-4-turbo"], "icon": "🤖"},
    "Gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai", "models": ["gemini-1.5-pro"], "icon": "♊"}
}

class ACEJsonEncoder(json.JSONEncoder):
    """自定义 JSON 编码器，处理 ndarray, Path 和 dataclass"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        if is_dataclass(obj):
            return asdict(obj)
        return super().default(obj)

class SettingsStore:
    def __init__(self, path: Path = SETTINGS_PATH):
        self.path = path
        self._cache = self._load()

    def _load(self) -> dict:
        settings = {"active_provider": "DeepSeek", "api_keys": {}, "temperature": 0.2, "llm_enabled": True}
        for p in DEFAULT_PROVIDERS:
            if key := os.getenv(f"ACE_{p.upper()}_API_KEY"): settings["api_keys"][p] = key
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                settings.update({k: v for k, v in data.items() if k in settings or k == "model"})
            except: pass
        return settings

    def get(self, key: str, default: Any = None) -> Any: return self._cache.get(key, default)

    def save(self, payload: dict):
        self._cache.update(payload)
        self.path.write_text(json.dumps(self._cache, ensure_ascii=False, indent=2), encoding="utf-8")

class SessionManager:
    def __init__(self, path: Path = SESSIONS_PATH):
        self.path = path
        self.sessions = self._load()

    def _load(self) -> List[Dict]:
        if self.path.exists():
            try: return json.loads(self.path.read_text(encoding="utf-8"))
            except: return []
        return []

    def save_session(self, session_id: str, messages: List[Dict], metadata: Dict = None):
        session_data = {
            "id": session_id,
            "messages": messages,
            "metadata": metadata or {},
            "updated_at": datetime.now().isoformat()
        }
        
        for i, s in enumerate(self.sessions):
            if s["id"] == session_id:
                session_data["created_at"] = s.get("created_at", session_data["updated_at"])
                self.sessions[i] = session_data
                break
        else:
            session_data["created_at"] = session_data["updated_at"]
            self.sessions.insert(0, session_data)
        
        self.path.write_text(json.dumps(self.sessions, cls=ACEJsonEncoder, ensure_ascii=False, indent=2), encoding="utf-8")

    def delete_session(self, session_id: str):
        self.sessions = [s for s in self.sessions if s["id"] != session_id]
        self.path.write_text(json.dumps(self.sessions, cls=ACEJsonEncoder, ensure_ascii=False, indent=2), encoding="utf-8")

from datetime import datetime


def load_settings() -> dict:
    """Return a flat LLM configuration dict, resolving the active provider.

    Used by demo_runner and benchmark CLI to get the LLM config from
    ``.ace_demo_config.json`` (populated by the Streamlit web UI).

    Returns a dict with keys:
        llm_enabled, llm_provider, llm_base_url, llm_api_key, llm_model
    """
    store = SettingsStore()
    provider = store.get("active_provider", "DeepSeek")
    api_keys = store.get("api_keys", {})
    provider_cfg = DEFAULT_PROVIDERS.get(provider, {})
    return {
        "llm_enabled": store.get("llm_enabled", False),
        "llm_provider": provider,
        "llm_base_url": provider_cfg.get("base_url", ""),
        "llm_api_key": api_keys.get(provider, ""),
        "llm_model": store.get("model", ""),
    }
