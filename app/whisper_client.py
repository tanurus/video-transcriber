from __future__ import annotations

import time
from pathlib import Path

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
    def __init__(self, api_key: str, model: str = "whisper-large-v3-turbo", timeout: int = 600, base_url: str | None = None, language: str = "en") -> None:
        self.client = OpenAI(api_key=api_key, timeout=timeout, base_url=base_url)
        self.model = model
        self.timeout = timeout
        self.language = language

    def transcribe_file(self, audio_file: str | Path, response_format: str = "text") -> str:
        retries = 3
        backoff = 5
        for attempt in range(1, retries + 1):
            try:
                with open(audio_file, "rb") as f:
                    resp = self.client.audio.transcriptions.create(
                        model=self.model,
                        file=f,
                        response_format=response_format,
                        language=self.language,
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
