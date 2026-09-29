from __future__ import annotations

import gc
import threading
import time
from typing import Any, Callable, Dict, Iterable, List, Optional

# Whisper's own fallback schedule: decode at 0, and only if that decode fails the
# compression-ratio / log-prob checks retry at progressively higher temperatures.
FALLBACK_TEMPERATURES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


class UnknownModelError(ValueError):
    pass


# name -> (type, min, max). Anything else in the form is ignored; values are clamped.
DECODE_FIELDS: Dict[str, tuple] = {
    "beam_size": (int, 1, 10),
    "best_of": (int, 1, 10),
    "patience": (float, 0.5, 3.0),
    "repetition_penalty": (float, 1.0, 2.0),
    "no_repeat_ngram_size": (int, 0, 10),
    "no_speech_threshold": (float, 0.0, 1.0),
    "log_prob_threshold": (float, -5.0, 0.0),
    "compression_ratio_threshold": (float, 1.0, 5.0),
    "hallucination_silence_threshold": (float, 0.0, 10.0),
    "vad_threshold": (float, 0.1, 0.9),
    "vad_min_silence_ms": (int, 100, 5000),
    "vad_speech_pad_ms": (int, 0, 2000),
    "vad_min_speech_ms": (int, 0, 2000),
    "condition_on_previous_text": (bool, None, None),
    "temperature_fallback": (bool, None, None),
    "vad_filter": (bool, None, None),
}


def parse_decode_options(form: Any) -> Dict[str, Any]:
    """Validated faster-whisper options from request form fields (strings)."""
    out: Dict[str, Any] = {}
    for key, (typ, lo, hi) in DECODE_FIELDS.items():
        raw = form.get(key)
        if raw is None or raw == "":
            continue
        if typ is bool:
            out[key] = str(raw).strip().lower() in ("1", "true", "on", "yes")
            continue
        try:
            value = typ(float(raw))
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be a number")
        out[key] = max(lo, min(hi, value))
    hot = (form.get("hotwords") or "").strip()
    if hot:
        out["hotwords"] = hot[:1000]
    if out.get("hallucination_silence_threshold") == 0:
        out.pop("hallucination_silence_threshold")
    return out


def normalize_model(requested: str, default: str, allowed: Iterable[str]) -> str:
    """Map an OpenAI-style model id onto a faster-whisper model name.

    Clients send ``whisper-large-v3`` (the id Groq uses) or OpenAI's generic
    ``whisper-1``; faster-whisper wants ``large-v3``. Anything outside the
    allowlist is rejected so a typo can never start a multi-GB download.
    """
    allowed = list(allowed)
    name = (requested or "").strip().lower()
    if name in ("", "whisper-1"):
        return default
    if name.startswith("whisper-"):
        name = name[len("whisper-"):]
    if name not in allowed:
        raise UnknownModelError(
            f"Model '{requested}' is not available here. Choose one of: "
            + ", ".join(f"whisper-{m}" for m in allowed)
        )
    return name


def _default_loader(device: str, compute_type: str) -> Callable[[str], Any]:
    def load(name: str) -> Any:
        # Imported lazily so the server module (and its tests) work without CUDA.
        from faster_whisper import WhisperModel

        return WhisperModel(name, device=device, compute_type=compute_type)

    return load


class ModelManager:
    """Owns the one Whisper model on the GPU.

    One decode runs at a time (a single 16 GB card gains nothing from parallel
    decodes of the same model, and two would double VRAM). The model loads on
    first use and unloads after ``idle_unload_sec`` so the card is free for other
    work between meetings.
    """

    def __init__(
        self,
        default_model: str = "large-v3",
        allowed_models: Optional[List[str]] = None,
        device: str = "cuda",
        compute_type: str = "float16",
        idle_unload_sec: int = 900,
        beam_size: int = 5,
        condition_on_previous_text: bool = False,
        loader: Optional[Callable[[str], Any]] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.default_model = default_model
        self.allowed_models = list(allowed_models or [default_model])
        self.idle_unload_sec = idle_unload_sec
        self.beam_size = beam_size
        self.condition_on_previous_text = condition_on_previous_text
        self._loader = loader or _default_loader(device, compute_type)
        self._clock = clock
        self._lock = threading.Lock()
        self._model: Any = None
        self.loaded_model: Optional[str] = None
        self._last_used = clock()
        self.busy = False

    def resolve(self, requested: str) -> str:
        return normalize_model(requested, self.default_model, self.allowed_models)

    def _ensure_loaded(self, name: str) -> Any:
        if self.loaded_model != name:
            self._unload_locked()
            self._model = self._loader(name)
            self.loaded_model = name
        return self._model

    def _unload_locked(self) -> None:
        if self._model is not None:
            self._model = None
            self.loaded_model = None
            # ctranslate2 releases VRAM when the model object is collected.
            gc.collect()

    def transcribe(
        self,
        audio_path: str,
        model: str,
        language: Optional[str] = None,
        temperature: float = 0.0,
        prompt: Optional[str] = None,
        decode: Optional[Dict[str, Any]] = None,
    ) -> dict:
        """``decode`` carries validated faster-whisper options (see parse_decode_options)."""
        name = self.resolve(model)
        decode = dict(decode or {})
        fallback = decode.pop("temperature_fallback", True)
        if temperature == 0:
            temperatures = FALLBACK_TEMPERATURES if fallback else [0.0]
        else:
            temperatures = [temperature]
        kwargs: Dict[str, Any] = {
            "beam_size": self.beam_size,
            "condition_on_previous_text": self.condition_on_previous_text,
            "vad_filter": False,
        }
        vad = {k: decode.pop(k) for k in list(decode) if k.startswith("vad_") and k != "vad_filter"}
        kwargs.update(decode)
        if kwargs.get("vad_filter"):
            kwargs["vad_parameters"] = {
                "threshold": vad.get("vad_threshold", 0.5),
                "min_silence_duration_ms": vad.get("vad_min_silence_ms", 1000),
                "speech_pad_ms": vad.get("vad_speech_pad_ms", 300),
                "min_speech_duration_ms": vad.get("vad_min_speech_ms", 250),
            }
        if kwargs.get("hallucination_silence_threshold"):
            kwargs["word_timestamps"] = True  # faster-whisper needs them for this check
        with self._lock:
            self.busy = True
            try:
                whisper = self._ensure_loaded(name)
                # Defaults: chunks are already cut at pauses by the app, and
                # conditioning on the previous window mostly feeds repetition loops.
                segments, info = whisper.transcribe(
                    audio_path,
                    language=language or None,
                    temperature=temperatures,
                    initial_prompt=prompt or None,
                    **kwargs,
                )
                # The segment generator does the actual decoding — drain it inside the lock.
                segs = [
                    {
                        "id": getattr(s, "id", i),
                        "seek": getattr(s, "seek", 0),
                        "start": float(s.start),
                        "end": float(s.end),
                        "text": s.text,
                        "tokens": list(getattr(s, "tokens", []) or []),
                        "temperature": float(getattr(s, "temperature", 0.0) or 0.0),
                        "avg_logprob": float(s.avg_logprob),
                        "compression_ratio": float(s.compression_ratio),
                        "no_speech_prob": float(s.no_speech_prob),
                    }
                    for i, s in enumerate(segments)
                ]
            finally:
                self.busy = False
                self._last_used = self._clock()
        return {
            "task": "transcribe",
            "language": info.language,
            "duration": float(info.duration),
            "text": " ".join(s["text"].strip() for s in segs if s["text"].strip()),
            "segments": segs,
        }

    def maybe_unload(self) -> bool:
        """Unload the model if it has sat idle past the threshold. Returns True if it did."""
        if self.idle_unload_sec <= 0:
            return False
        # Never wait on a running decode; the next sweep will catch it.
        if not self._lock.acquire(blocking=False):
            return False
        try:
            if self._model is None:
                return False
            if self._clock() - self._last_used < self.idle_unload_sec:
                return False
            self._unload_locked()
            return True
        finally:
            self._lock.release()
