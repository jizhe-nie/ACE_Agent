from __future__ import annotations

import contextlib
import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

load_dotenv()

SETTINGS_PATH = Path(__file__).resolve().parents[1] / ".ace_demo_config.json"
SESSIONS_PATH = Path(__file__).resolve().parents[1] / ".ace_sessions.json"

DEFAULT_PROVIDERS = {
    "DeepSeek": {
        "base_url": "https://api.deepseek.com",
        "models": ["deepseek-chat", "deepseek-reasoner"],
        "icon": "🌊",
    },
    "DashScope": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "models": ["qwen-plus", "qwen-max"],
        "icon": "☁️",
    },
    "Moonshot": {"base_url": "https://api.moonshot.cn/v1", "models": ["moonshot-v1-8k"], "icon": "🌙"},
    "OpenAI": {"base_url": "https://api.openai.com/v1", "models": ["gpt-4o", "gpt-4-turbo"], "icon": "🤖"},
    "Gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "models": ["gemini-1.5-pro"],
        "icon": "♊",
    },
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
        # numpy scalar types — some numpy versions don't inherit from Python
        # builtins so the stdlib json encoder rejects them
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


class SettingsStore:
    def __init__(self, path: Path = SETTINGS_PATH):
        self.path = path
        self._cache = self._load()

    def _load(self) -> dict:
        settings = {"active_provider": "DeepSeek", "api_keys": {}, "temperature": 0.2, "llm_enabled": True}
        for p in DEFAULT_PROVIDERS:
            if key := os.getenv(f"ACE_{p.upper()}_API_KEY"):
                settings["api_keys"][p] = key
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                settings.update({k: v for k, v in data.items() if k in settings or k == "model"})
            except Exception:
                pass
        return settings

    def get(self, key: str, default: Any = None) -> Any:
        return self._cache.get(key, default)

    def save(self, payload: dict):
        self._cache.update(payload)
        self.path.write_text(json.dumps(self._cache, ensure_ascii=False, indent=2), encoding="utf-8")


class SessionManager:
    _MAX_FILE_MB: int = 50
    _MAX_SESSIONS: int = 30

    def __init__(self, path: Path = SESSIONS_PATH):
        self.path = path
        self.sessions = self._load()

    def _load(self) -> list[dict]:
        if not self.path.exists():
            return []
        # Refuse to load files that are too large — they cause minutes-long
        # hangs and consume gigabytes of memory during JSON parse.
        file_mb = self.path.stat().st_size / (1024 * 1024)
        if file_mb > self._MAX_FILE_MB:
            _backup = self.path.with_suffix(".json.bak")
            with contextlib.suppress(OSError):
                self.path.rename(_backup)
            return []
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return []

    def _strip_heavy_data(self, data: Any) -> Any:
        """Recursively remove large arrays/lists to keep session files small."""
        if isinstance(data, dict):
            # Target common large keys in our project
            clean_dict = {}
            for k, v in data.items():
                if k in ("X", "y", "labels", "code", "feature_names", "decision_trace", "ranking"):
                    clean_dict[k] = [] # Stripped
                elif k == "thought" and isinstance(v, str) and len(v) > 5000:
                    clean_dict[k] = v[:5000] + "... [TRUNCATED]"
                else:
                    clean_dict[k] = self._strip_heavy_data(v)
            return clean_dict
        if isinstance(data, list):
            # If it's a list of many items and they aren't messages, it might be heavy
            if len(data) > 100:
                # Check if it's a list of dicts (like messages), if so, keep it but strip each
                if len(data) > 0 and isinstance(data[0], dict) and "role" in data[0]:
                    return [self._strip_heavy_data(i) for i in data]
                return [] # Strip other large lists
            return [self._strip_heavy_data(i) for i in data]
        return data

    def save_session(self, session_id: str, messages: list[dict], metadata: dict = None):
        # Strip heavy data from messages before saving
        clean_messages = self._strip_heavy_data(messages)

        session_data = {
            "id": session_id,
            "messages": clean_messages,
            "metadata": metadata or {},
            "updated_at": datetime.now().isoformat(),
        }

        # Check if session data actually changed (ignore updated_at)
        for i, s in enumerate(self.sessions):
            if s["id"] == session_id:
                if (
                    s.get("messages") == clean_messages
                    and s.get("metadata", {}) == (metadata or {})
                ):
                    return  # No changes, skip disk write
                session_data["created_at"] = s.get("created_at", session_data["updated_at"])
                self.sessions[i] = session_data
                break
        else:
            session_data["created_at"] = session_data["updated_at"]
            self.sessions.insert(0, session_data)

        # Cap history to prevent session file inflation
        if len(self.sessions) > self._MAX_SESSIONS:
            self.sessions = self.sessions[: self._MAX_SESSIONS]

        self.path.write_text(
            json.dumps(self.sessions, cls=ACEJsonEncoder, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        # Post-write safety valve: if the file still grew beyond limit,
        # aggressively trim and re-write.
        _written_mb = self.path.stat().st_size / (1024 * 1024)
        if _written_mb > self._MAX_FILE_MB:
            _keep = max(3, self._MAX_SESSIONS // 2)
            self.sessions = self.sessions[:_keep]
            self.path.write_text(
                json.dumps(self.sessions, cls=ACEJsonEncoder, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    def delete_session(self, session_id: str):
        old_len = len(self.sessions)
        self.sessions = [s for s in self.sessions if s["id"] != session_id]
        if len(self.sessions) != old_len:
            self.path.write_text(
                json.dumps(self.sessions, cls=ACEJsonEncoder, ensure_ascii=False, indent=2), encoding="utf-8"
            )


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
