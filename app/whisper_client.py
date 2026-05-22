from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable, List, Optional

from openai import OpenAI, RateLimitError, APIError, APITimeoutError


class WhisperClient:
    def __init__(self, api_key: str, model: str = "whisper-large-v3-turbo", timeout: int = 600, base_url: str | None = None) -> None:
        self.client = OpenAI(api_key=api_key, timeout=timeout, base_url=base_url)
        self.model = model
        self.timeout = timeout

    def transcribe_file(self, audio_file: str | Path, response_format: str = "text") -> str:
        retries = 3
        backoff = 5
        last_err: Optional[Exception] = None
        for attempt in range(1, retries + 1):
            try:
                with open(audio_file, "rb") as f:
                    resp = self.client.audio.transcriptions.create(
                        model=self.model,
                        file=f,
                        response_format=response_format,
                    )
                # For response_format="text", resp is a string-like object with .text
                if hasattr(resp, "text") and isinstance(resp.text, str):
                    return resp.text
                # Some SDK versions may return a plain string for text mode
                if isinstance(resp, str):
                    return resp
                # Fallback: try to coerce
                return str(resp)
            except (RateLimitError, APIError, APITimeoutError) as e:
                last_err = e
                if attempt < retries:
                    time.sleep(backoff * attempt)
                    continue
                raise
            except Exception as e:  # non-retryable or unexpected
                last_err = e
                raise
        if last_err:
            raise last_err
        return ""

    def transcribe_many(self, audio_files: Iterable[str | Path]) -> str:
        parts: List[str] = []
        for idx, p in enumerate(audio_files, start=1):
            text = self.transcribe_file(p)
            header = f"\n\n--- Segment {idx} ---\n\n"
            parts.append(header + text.strip())
        return "".join(parts).strip() or ""
