from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable, List

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

# Only transient failures are worth retrying; 4xx errors (bad key, bad model,
# file too large) would just re-upload the audio three times to fail the same way.
_RETRYABLE_ERRORS = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)


class WhisperClient:
    def __init__(self, api_key: str, model: str = "whisper-large-v3", timeout: int = 600, base_url: str | None = None) -> None:
        self.client = OpenAI(api_key=api_key, timeout=timeout, base_url=base_url)
        self.model = model
        self.timeout = timeout

    def transcribe_file(self, audio_file: str | Path, response_format: str = "text") -> str:
        retries = 3
        backoff = 5
        for attempt in range(1, retries + 1):
            try:
                with open(audio_file, "rb") as f:
                    # Use the translations endpoint so the output is ALWAYS English,
                    # regardless of the spoken language (e.g. Russian / Romanian).
                    # temperature=0 minimises Whisper's repetition/hallucination loops.
                    # NOTE: Groq supports translations only with whisper-large-v3.
                    resp = self.client.audio.translations.create(
                        model=self.model,
                        file=f,
                        response_format=response_format,
                        temperature=0,
                    )
                # For response_format="text", resp is a string-like object with .text
                if hasattr(resp, "text") and isinstance(resp.text, str):
                    return resp.text
                # Some SDK versions may return a plain string for text mode
                if isinstance(resp, str):
                    return resp
                # Fallback: try to coerce
                return str(resp)
            except _RETRYABLE_ERRORS:
                if attempt < retries:
                    time.sleep(backoff * attempt)
                    continue
                raise
        raise RuntimeError("unreachable: retry loop exited without returning or raising")

    def transcribe_many(self, audio_files: Iterable[str | Path]) -> str:
        parts: List[str] = []
        for idx, p in enumerate(audio_files, start=1):
            text = self.transcribe_file(p)
            header = f"\n\n--- Segment {idx} ---\n\n"
            parts.append(header + text.strip())
        return "".join(parts).strip() or ""
