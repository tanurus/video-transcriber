from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import List, Optional

from dotenv import load_dotenv


GROQ_BASE_URL = "https://api.groq.com/openai/v1"
# The bundled GPU server (gpu_server/) listens here by default.
DEFAULT_LOCAL_URL = "http://127.0.0.1:18921/v1"
PROVIDERS = ("auto", "local", "groq", "openai")


def health_url(base_url: str) -> str:
    """``http://host:port/v1`` -> ``http://host:port/health``."""
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    return root + "/health"


def local_health_error(base_url: str, timeout: float = 2.0) -> Optional[str]:
    """Return None if the local Whisper server is ready, else a short reason."""
    try:
        with urllib.request.urlopen(health_url(base_url), timeout=timeout) as resp:
            body = json.loads(resp.read() or b"{}")
            if body.get("status") == "ok":
                return None
            return f"status {body.get('status')!r}"
    except urllib.error.HTTPError as e:
        # 503 carries a JSON body such as {"status": "no-gpu"}.
        try:
            return f"status {json.loads(e.read() or b'{}').get('status')!r}"
        except Exception:  # noqa: BLE001
            return f"HTTP {e.code}"
    except Exception as e:  # noqa: BLE001 - refused, timeout, DNS: all mean "not ready"
        return str(getattr(e, "reason", e))


@dataclass
class Config:
    openai_api_key: str
    model: str = "whisper-large-v3"
    timeout: int = 600  # seconds
    chunk_target_mb: int = 24
    audio_bitrate: str = "96k"  # ffmpeg format, e.g., "64k", "96k", "128k"
    base_url: Optional[str] = None  # None = OpenAI default
    language: Optional[str] = None  # None = Whisper auto-detects the spoken language

    # When set, each chunk is transcribed once per candidate language and the
    # highest-confidence decode wins. This pins Whisper to a known set (e.g.
    # ro/ru/en) so it can never drift into Polish/Ukrainian on ambiguous audio.
    candidate_languages: Optional[List[str]] = None

    # Silence-aware chunking. Short chunks let Whisper re-detect the language on
    # each utterance, which is what keeps multilingual meetings from being locked
    # into (and garbled by) whichever language the first 30s happened to be.
    chunk_target_sec: int = 25  # aim to cut around here, snapped to a pause
    chunk_max_sec: int = 40  # force a cut by here even without a pause
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

    # Which backend was chosen: "local" (bundled GPU server), "groq" or "openai".
    provider: str = "openai"

    @staticmethod
    def load(env_path: Optional[str] = None) -> "Config":
        load_dotenv(dotenv_path=env_path, override=False)

        provider, api_key, base_url, default_model = _choose_provider()
        # "or default" (not getenv's default) so blank placeholders in .env
        # ("OPENAI_TIMEOUT=") behave like unset variables instead of crashing int().
        model = os.getenv("TRANSCRIBE_MODEL") or default_model
        timeout = int(os.getenv("OPENAI_TIMEOUT") or "600")
        chunk_target_mb = int(os.getenv("CHUNK_TARGET_MB") or "24")
        audio_bitrate = os.getenv("AUDIO_BITRATE") or "96k"
        language = os.getenv("TRANSCRIBE_LANGUAGE") or None

        raw_candidates = os.getenv("CANDIDATE_LANGUAGES") or ""
        candidate_languages = [c.strip() for c in raw_candidates.split(",") if c.strip()] or None

        chunk_target_sec = int(os.getenv("CHUNK_TARGET_SEC") or "25")
        chunk_max_sec = int(os.getenv("CHUNK_MAX_SEC") or "40")
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
            candidate_languages=candidate_languages,
            chunk_target_sec=chunk_target_sec,
            chunk_max_sec=chunk_max_sec,
            silence_noise_db=silence_noise_db,
            silence_min_sec=silence_min_sec,
            max_concurrency=max_concurrency,
            no_speech_threshold=no_speech_threshold,
            logprob_threshold=logprob_threshold,
            compression_ratio_threshold=compression_ratio_threshold,
            provider=provider,
        )


def _no_key_error() -> RuntimeError:
    return RuntimeError(
        "No transcription backend configured. Set LOCAL_WHISPER_URL (local GPU server), "
        "GROQ_API_KEY or OPENAI_API_KEY in your environment or .env file."
    )


def _choose_provider() -> tuple:
    """Return (provider, api_key, base_url, default_model) from the environment.

    ``auto`` (the default) keeps the original behaviour when no local server is
    configured: Groq if its key is set, else OpenAI. With LOCAL_WHISPER_URL set it
    prefers that server whenever its /health answers, falling back to a cloud key.
    """
    provider = (os.getenv("TRANSCRIBE_PROVIDER") or "auto").strip().lower()
    if provider not in PROVIDERS:
        raise RuntimeError(
            f"Unknown TRANSCRIBE_PROVIDER '{provider}'. Use one of: {', '.join(PROVIDERS)}."
        )
    groq_key = os.getenv("GROQ_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    local_url = os.getenv("LOCAL_WHISPER_URL") or None

    def local() -> tuple:
        url = local_url or DEFAULT_LOCAL_URL
        # The OpenAI SDK refuses an empty key; the local server ignores it unless
        # it was started with WHISPER_API_KEY.
        key = os.getenv("LOCAL_WHISPER_API_KEY") or "local"
        # whisper-large-v3 = the most accurate open Whisper model; the id contains
        # "whisper" and the server maps it onto faster-whisper's "large-v3".
        return ("local", key, url, "whisper-large-v3")

    def groq() -> tuple:
        if not groq_key:
            raise RuntimeError("TRANSCRIBE_PROVIDER=groq but GROQ_API_KEY is not set.")
        # whisper-large-v3 = Groq's highest-accuracy Whisper model
        # (large-v3-turbo is faster but slightly less accurate).
        return ("groq", groq_key, GROQ_BASE_URL, "whisper-large-v3")

    def openai() -> tuple:
        if not openai_key:
            raise RuntimeError("TRANSCRIBE_PROVIDER=openai but OPENAI_API_KEY is not set.")
        return ("openai", openai_key, None, "gpt-4o-transcribe")

    def unreachable(reason: str) -> RuntimeError:
        url = local_url or DEFAULT_LOCAL_URL
        return RuntimeError(
            f"Local Whisper server at {url} is not responding ({reason}). "
            "Check the whisper-gpu service, or set a cloud API key as a fallback."
        )

    if provider == "local":
        err = local_health_error(local_url or DEFAULT_LOCAL_URL)
        if err:
            raise unreachable(err)
        return local()
    if provider == "groq":
        return groq()
    if provider == "openai":
        return openai()

    # auto
    if local_url:
        err = local_health_error(local_url)
        if err is None:
            return local()
        if groq_key:
            return groq()
        if openai_key:
            return openai()
        raise unreachable(err)
    if groq_key:
        return groq()
    if openai_key:
        return openai()
    raise _no_key_error()
