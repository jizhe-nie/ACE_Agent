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
        with contextlib.suppress(OSError):
            self.path.write_text(json.dumps(self._cache, ensure_ascii=False, indent=2), encoding="utf-8")


class SessionManager:
    _MAX_FILE_MB: int = 50
    _MAX_SESSIONS: int = 30

    def __init__(self, path: Path = SESSIONS_PATH):
        self.path = path
        self._full_sessions: dict[str, list[dict]] = {}
        self.sessions = self._load_metadata_only()

    def _load_raw(self) -> list[dict]:
        """Load full sessions array from disk — shared by metadata and full load paths."""
        if not self.path.exists():
            return []
        file_mb = self.path.stat().st_size / (1024 * 1024)
        if file_mb > self._MAX_FILE_MB:
            _backup = self.path.with_suffix(".json.bak")
            with contextlib.suppress(Exception):
                self.path.rename(_backup)
            # OS-level delete if rename failed (e.g., backup already exists on Windows)
            with contextlib.suppress(Exception):
                if self.path.exists() and self.path.stat().st_size > 500 * 1024 * 1024:
                    self.path.unlink()
            return []
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return []

    def _load_metadata_only(self) -> list[dict]:
        """Load only metadata fields (no messages) for the sidebar session list."""
        raw = self._load_raw()
        return [
            {
                "id": s["id"],
                "metadata": s.get("metadata", {}),
                "updated_at": s.get("updated_at", ""),
                "created_at": s.get("created_at", ""),
                "message_count": len(s.get("messages", [])),
            }
            for s in raw
        ]

    def get_full_session(self, session_id: str) -> list[dict] | None:
        """Load full session messages lazily (only on first access)."""
        if session_id in self._full_sessions:
            return self._full_sessions[session_id]
        raw = self._load_raw()
        for s in raw:
            if s["id"] == session_id:
                msgs = s.get("messages", [])
                self._full_sessions[session_id] = msgs
                return msgs
        return None

    def _strip_heavy_data(self, data: Any) -> Any:
        """Recursively remove large arrays/objects to keep session files small."""
        # Terminal: numpy arrays → stripped to empty list
        if isinstance(data, np.ndarray):
            return []

        # Terminal: Path objects → string
        if isinstance(data, Path):
            return str(data)

        # Dataclass: convert to dict, then strip recursively
        if is_dataclass(data) and not isinstance(data, type):
            return self._strip_heavy_data(asdict(data))

        if isinstance(data, dict):
            clean_dict = {}
            for k, v in data.items():
                if k in ("X", "y", "labels", "code", "feature_names", "decision_trace", "ranking"):
                    clean_dict[k] = []
                elif k == "thought" and isinstance(v, str) and len(v) > 5000:
                    clean_dict[k] = v[:5000] + "... [TRUNCATED]"
                elif k == "dataset" and isinstance(v, dict):
                    clean_dict[k] = {
                        "name": v.get("name", ""),
                        "display_name": v.get("display_name", ""),
                        "description": (v.get("description", "") or "")[:500],
                        "shape_family": v.get("shape_family", "unknown"),
                    }
                elif k == "results" and isinstance(v, list):
                    clean_dict[k] = [
                        {
                            "algorithm_name": r.get("algorithm_name", "?") if isinstance(r, dict) else getattr(r, "algorithm_name", "?"),
                            "expert_key": r.get("expert_key", "") if isinstance(r, dict) else getattr(r, "expert_key", ""),
                            "expert_label": r.get("expert_label", "") if isinstance(r, dict) else getattr(r, "expert_label", ""),
                            "metrics": (r.get("metrics", {}) if isinstance(r, dict) else getattr(r, "metrics", {})),
                            "plot_path": str(r.get("plot_path", "")) if isinstance(r, dict) else str(getattr(r, "plot_path", "")),
                        }
                        for r in v
                    ]
                else:
                    clean_dict[k] = self._strip_heavy_data(v)
            return clean_dict
        if isinstance(data, list):
            if len(data) > 100:
                if len(data) > 0 and isinstance(data[0], dict) and "role" in data[0]:
                    return [self._strip_heavy_data(i) for i in data]
                return []
            return [self._strip_heavy_data(i) for i in data]
        return data

    def save_session(self, session_id: str, messages: list[dict], metadata: dict = None):
        # Strip heavy data from messages before saving
        clean_messages = self._strip_heavy_data(messages)
        metadata = metadata or {}

        # Load the raw sessions from disk (full, with messages) so we can
        # update a single entry without losing data from other sessions.
        raw = self._load_raw()

        session_data = {
            "id": session_id,
            "messages": clean_messages,
            "metadata": metadata,
            "updated_at": datetime.now().isoformat(),
        }

        # Find and update in the raw array
        for i, s in enumerate(raw):
            if s["id"] == session_id:
                if (
                    s.get("messages") == clean_messages
                    and s.get("metadata", {}) == metadata
                ):
                    return  # No changes, skip disk write
                session_data["created_at"] = s.get("created_at", session_data["updated_at"])
                raw[i] = session_data
                break
        else:
            session_data["created_at"] = session_data["updated_at"]
            raw.insert(0, session_data)

        # Cap history
        if len(raw) > self._MAX_SESSIONS:
            raw = raw[: self._MAX_SESSIONS]

        # Write full data to disk
        try:
            self.path.write_text(
                json.dumps(raw, cls=ACEJsonEncoder, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:
            return

        # Post-write safety valve
        try:
            _written_mb = self.path.stat().st_size / (1024 * 1024)
        except Exception:
            return
        if _written_mb > self._MAX_FILE_MB:
            _keep = max(3, self._MAX_SESSIONS // 2)
            raw = raw[:_keep]
            with contextlib.suppress(Exception):
                self.path.write_text(
                    json.dumps(raw, cls=ACEJsonEncoder, ensure_ascii=False, indent=2), encoding="utf-8"
                )

        # Update in-memory caches
        if session_id in self._full_sessions:
            self._full_sessions[session_id] = list(clean_messages)
        self.sessions = self._load_metadata_only()

    def delete_session(self, session_id: str):
        old_len = len(self.sessions)
        self.sessions = [s for s in self.sessions if s["id"] != session_id]
        if len(self.sessions) != old_len:
            with contextlib.suppress(Exception):
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
