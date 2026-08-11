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
    language: Optional[str] = None  # None = Whisper auto-detects the spoken language

    # Silence-aware chunking. Short chunks let Whisper re-detect the language on
    # each utterance, which is what keeps multilingual meetings from being locked
    # into (and garbled by) whichever language the first 30s happened to be.
    chunk_target_sec: int = 45  # aim to cut around here, snapped to a pause
    chunk_max_sec: int = 90  # force a cut by here even without a pause
    silence_noise_db: str = "-30dB"  # ffmpeg silencedetect noise floor
    silence_min_sec: float = 0.5  # min pause length to count as a cut point

    # Transcribe chunks concurrently. Kept modest so we stay under Groq's rate
    # limits; each chunk is tiny (~45-90s) so a small pool already saturates.
    max_concurrency: int = 4

    # Confidence thresholds for dropping hallucinated/no-speech segments. These
    # are Whisper's own internal defaults, surfaced here so they can be tuned.
    no_speech_threshold: float = 0.6
    logprob_threshold: float = -1.0
    compression_ratio_threshold: float = 2.4

    @staticmethod
    def load(env_path: Optional[str] = None) -> "Config":
        load_dotenv(dotenv_path=env_path, override=False)

        groq_key = os.getenv("GROQ_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")

        if groq_key:
            api_key = groq_key
            base_url = GROQ_BASE_URL
            # whisper-large-v3 = Groq's highest-accuracy Whisper model
            # (large-v3-turbo is faster but slightly less accurate).
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
        language = os.getenv("TRANSCRIBE_LANGUAGE") or None

        chunk_target_sec = int(os.getenv("CHUNK_TARGET_SEC") or "45")
        chunk_max_sec = int(os.getenv("CHUNK_MAX_SEC") or "90")
        silence_noise_db = os.getenv("SILENCE_NOISE_DB") or "-30dB"
        silence_min_sec = float(os.getenv("SILENCE_MIN_SEC") or "0.5")
        max_concurrency = max(1, int(os.getenv("MAX_CONCURRENCY") or "4"))
        no_speech_threshold = float(os.getenv("NO_SPEECH_THRESHOLD") or "0.6")
        logprob_threshold = float(os.getenv("LOGPROB_THRESHOLD") or "-1.0")
        compression_ratio_threshold = float(os.getenv("COMPRESSION_RATIO_THRESHOLD") or "2.4")

        return Config(
            openai_api_key=api_key,
            model=model,
            timeout=timeout,
            chunk_target_mb=chunk_target_mb,
            audio_bitrate=audio_bitrate,
            base_url=base_url,
            language=language,
            chunk_target_sec=chunk_target_sec,
            chunk_max_sec=chunk_max_sec,
            silence_noise_db=silence_noise_db,
            silence_min_sec=silence_min_sec,
            max_concurrency=max_concurrency,
            no_speech_threshold=no_speech_threshold,
            logprob_threshold=logprob_threshold,
            compression_ratio_threshold=compression_ratio_threshold,
        )
