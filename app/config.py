from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv


GROQ_BASE_URL = "https://api.groq.com/openai/v1"


@dataclass
class Config:
    openai_api_key: str
    model: str = "whisper-large-v3"
    timeout: int = 600  # seconds
    chunk_target_mb: int = 24
    audio_bitrate: str = "96k"  # ffmpeg format, e.g., "64k", "96k", "128k"
    base_url: Optional[str] = None  # None = OpenAI default
    language: str = "en"  # forced transcription language (Whisper does not auto-detect here)

    @staticmethod
    def load(env_path: Optional[str] = None) -> "Config":
        load_dotenv(dotenv_path=env_path, override=False)

        groq_key = os.getenv("GROQ_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")

        if groq_key:
            api_key = groq_key
            base_url = GROQ_BASE_URL
            # whisper-large-v3 = highest quality AND the only Groq model that
            # supports the translations (-> English) endpoint used by WhisperClient.
            default_model = "whisper-large-v3"
        elif openai_key:
            api_key = openai_key
            base_url = None
            default_model = "gpt-4o-transcribe"
        else:
            raise RuntimeError(
                "No API key found. Set GROQ_API_KEY or OPENAI_API_KEY in your environment or .env file."
            )

        # "or default" (not getenv's default) so blank placeholders in .env
        # ("OPENAI_TIMEOUT=") behave like unset variables instead of crashing int().
        model = os.getenv("TRANSCRIBE_MODEL") or default_model
        timeout = int(os.getenv("OPENAI_TIMEOUT") or "600")
        chunk_target_mb = int(os.getenv("CHUNK_TARGET_MB") or "24")
        audio_bitrate = os.getenv("AUDIO_BITRATE") or "96k"
        language = os.getenv("TRANSCRIBE_LANGUAGE") or "en"

        return Config(
            openai_api_key=api_key,
            model=model,
            timeout=timeout,
            chunk_target_mb=chunk_target_mb,
            audio_bitrate=audio_bitrate,
            base_url=base_url,
            language=language,
        )
