"""Connection settings entered once in the web UI (OpenAI, Supabase).

Stored in ``<data_dir>/connections.json`` with mode 0600. Secret values are
never sent back to the browser — only a masked hint of their last characters.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict

SECRET_FIELDS = ("openai_api_key", "supabase_key")


@dataclass
class Connections:
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str = ""
    openai_model: str = "gpt-5.4-mini"
    # Language for titles, descriptions and tags. "same" = the transcript's main language.
    ai_language: str = "English"
    supabase_url: str = ""
    supabase_key: str = ""
    supabase_table: str = "transcripts"

    @property
    def ai_ready(self) -> bool:
        return bool(self.openai_api_key and self.openai_base_url and self.openai_model)

    @property
    def supabase_ready(self) -> bool:
        return bool(self.supabase_url and self.supabase_key and self.supabase_table)


def mask(value: str) -> str:
    if not value:
        return ""
    return "•" * 8 + value[-4:] if len(value) > 8 else "•" * len(value)


class ConnectionStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def load(self) -> Connections:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raw = {}
        known = {f.name for f in fields(Connections)}
        conn = Connections(**{k: v for k, v in raw.items() if k in known and isinstance(v, str)})
        # Environment wins for keys provisioned outside the UI (e.g. a secrets file).
        for f in known:
            env = os.getenv("VT_" + f.upper())
            if env:
                setattr(conn, f, env)
        return conn

    def update(self, values: Dict[str, Any]) -> Connections:
        """Merge ``values``; a blank secret means "keep the stored one"."""
        with self._lock:
            current = self.load()
            data = asdict(current)
            for key, value in values.items():
                if key not in data or value is None:
                    continue
                value = str(value).strip()
                if key in SECRET_FIELDS and value == "":
                    continue
                if key == "supabase_url":
                    value = value.rstrip("/")
                data[key] = value
            self._write(data)
            return Connections(**data)

    def clear(self, key: str) -> None:
        with self._lock:
            data = asdict(self.load())
            if key in data:
                data[key] = ""
                self._write(data)

    def public(self) -> Dict[str, Any]:
        """What the settings page may show: secrets masked."""
        data = asdict(self.load())
        for key in SECRET_FIELDS:
            data[key + "_hint"] = mask(data.pop(key))
        return data

    def _write(self, data: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.chmod(tmp, 0o600)
        os.replace(tmp, self.path)
