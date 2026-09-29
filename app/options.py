"""The transcription settings catalogue — one definition drives the web form,
the upload API, validation, presets and the pipeline.

Every job stores the full resolved options (``resolve()`` output) so it can be
regenerated later with a tweaked copy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

GROUPS: List[Tuple[str, str]] = [
    ("model", "Model & language"),
    ("context", "Vocabulary & context"),
    ("audio", "Audio preparation"),
    ("segmentation", "Segmentation"),
    ("decoding", "Decoding (local GPU)"),
    ("filters", "Hallucination filters"),
    ("after", "After transcription"),
]


@dataclass(frozen=True)
class Option:
    key: str
    label: str
    group: str
    type: str  # choice | int | float | bool | text | textarea
    default: Any
    help: str = ""
    choices: Sequence[Tuple[str, str]] = ()
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    advanced: bool = True  # hidden behind "Advanced settings" in the form
    local_only: bool = False  # only the local GPU server honours it
    show_if: Optional[Tuple[str, Tuple[str, ...]]] = None  # (key, values) that make it relevant


CATALOG: List[Option] = [
    # --- model & language ------------------------------------------------------
    Option("model", "Model", "model", "choice", "whisper-large-v3",
           "large-v3 is the most accurate open Whisper model. Turbo is quicker and slightly less accurate.",
           choices=[("whisper-large-v3", "Whisper large-v3 — best accuracy"),
                    ("whisper-large-v3-turbo", "Whisper large-v3-turbo — faster")],
           advanced=False),
    Option("language_mode", "Language", "model", "choice", "auto",
           "Auto detects the language of every chunk, which is right for mixed RO/RU/EN meetings. "
           "Fix it only when the whole recording is one language.",
           choices=[("auto", "Auto-detect per chunk"), ("fixed", "One fixed language"),
                    ("candidates", "Pick the best of a few languages per chunk")],
           advanced=False),
    Option("language", "Fixed language (ISO code)", "model", "text", "",
           "e.g. ru, ro, en, uk", show_if=("language_mode", ("fixed",)), advanced=False),
    Option("candidate_languages", "Candidate languages", "model", "text", "ro,ru,en",
           "Each chunk is decoded once per language and the most confident decode wins. "
           "Costs one decode per language.",
           show_if=("language_mode", ("candidates",)), advanced=False),
    # --- vocabulary & context ------------------------------------------------------
    Option("profile_id", "Transcription profile", "context", "choice", "",
           "A saved set of prompt, hotwords and languages for a recurring meeting (Settings → Profiles).",
           advanced=False),
    Option("prompt", "Initial prompt", "context", "textarea", "",
           "Spelling and style hints: names, brands, jargon. Whisper reads up to ~224 tokens. "
           "Steers, never enforces — an overloaded prompt can cause repetition loops."),
    Option("hotwords", "Hotwords", "context", "text", "",
           "Comma-separated terms to boost (local GPU only). Use for names Whisper keeps mangling.",
           local_only=True),
    # --- audio preparation ------------------------------------------------------------
    Option("normalize", "Normalize loudness", "audio", "bool", True,
           "One gentle linear gain to −16 LUFS (EBU R128), capped at ±20 dB, no compression. "
           "Helps quiet and far-away recordings; harmless on good ones."),
    Option("denoise", "Noise reduction", "audio", "choice", "off",
           "Tested 2026-09-30 on noisy meeting audio: none of these beat the untouched audio, "
           "and heavy filters made it worse. Whisper was trained on noisy audio. Leave off unless "
           "a recording has steady hiss or hum.",
           choices=[("off", "Off (recommended)"), ("light", "Light — steady hiss / hum"),
                    ("strong", "Strong — very noisy, may damage speech"),
                    ("rnnoise", "RNNoise — neural, voice-focused")]),
    Option("highpass", "Cut rumble below 80 Hz", "audio", "bool", False,
           "Removes low-frequency rumble (air conditioning, handling noise). Speech is unaffected."),
    # --- segmentation ----------------------------------------------------------------------
    Option("segmentation", "Segmentation", "segmentation", "choice", "chunks",
           "Smart chunks cut at pauses so each chunk holds one language — the most accurate for mixed-language "
           "meetings. Whisper native sends the whole file with Whisper's own speech detection: only for "
           "single-language recordings (it garbled 40–80% of words on mixed RO/RU/EN audio in tests).",
           choices=[("chunks", "Smart chunks (recommended)"),
                    ("native", "Whisper native — whole file, single language")]),
    Option("chunk_target_sec", "Chunk target (s)", "segmentation", "int", 8,
           "Aim to cut around here, snapped to the nearest pause. Shorter = cleaner language switches.",
           min=4, max=120, step=1, show_if=("segmentation", ("chunks",))),
    Option("chunk_max_sec", "Chunk maximum (s)", "segmentation", "int", 15,
           "Force a cut by here even without a pause.", min=6, max=300, step=1,
           show_if=("segmentation", ("chunks",))),
    Option("silence_noise_db", "Silence level (dB)", "segmentation", "int", -30,
           "What counts as a pause. Raise towards −25 for noisy rooms.", min=-60, max=-10, step=1,
           show_if=("segmentation", ("chunks",))),
    Option("silence_min_sec", "Minimum pause (s)", "segmentation", "float", 0.5,
           "Shortest gap that counts as a cut point.", min=0.1, max=5, step=0.1,
           show_if=("segmentation", ("chunks",))),
    Option("max_concurrency", "Parallel chunks", "segmentation", "int", 2,
           "Chunks in flight at once. The GPU decodes one at a time; 2 keeps it busy.",
           min=1, max=8, step=1, show_if=("segmentation", ("chunks",))),
    Option("vad_threshold", "Speech detection threshold", "segmentation", "float", 0.5,
           "Silero VAD sensitivity. 0.5 is calibrated; 0.35–0.4 keeps quiet or distant speakers "
           "(chest-worn mics).", min=0.1, max=0.9, step=0.05, local_only=True,
           show_if=("segmentation", ("native",))),
    Option("vad_min_silence_ms", "Minimum silence (ms)", "segmentation", "int", 1000,
           "Pauses shorter than this stay inside speech.", min=100, max=5000, step=50, local_only=True,
           show_if=("segmentation", ("native",))),
    Option("vad_speech_pad_ms", "Speech padding (ms)", "segmentation", "int", 300,
           "Kept either side of speech so word edges are not clipped.", min=0, max=2000, step=50,
           local_only=True, show_if=("segmentation", ("native",))),
    Option("vad_min_speech_ms", "Minimum speech (ms)", "segmentation", "int", 250,
           "Shorter blips are ignored.", min=0, max=2000, step=50, local_only=True,
           show_if=("segmentation", ("native",))),
    # --- decoding -------------------------------------------------------------------------------
    Option("beam_size", "Beam size", "decoding", "int", 5,
           "How many hypotheses Whisper keeps while decoding. 5 is Whisper's accuracy default; 1 is greedy.",
           min=1, max=10, step=1, local_only=True),
    Option("best_of", "Best of", "decoding", "int", 5,
           "Candidates sampled when a decode falls back to a higher temperature.",
           min=1, max=10, step=1, local_only=True),
    Option("patience", "Beam patience", "decoding", "float", 1.0,
           "Above 1 explores more before stopping. Slower, occasionally better.",
           min=0.5, max=3, step=0.1, local_only=True),
    Option("temperature_fallback", "Temperature fallback", "decoding", "bool", True,
           "Retry a failed decode at rising temperatures (Whisper's standard recovery)."),
    Option("condition_on_previous_text", "Condition on previous text", "decoding", "bool", False,
           "Feeds each window the previous text. Better flow on long single-language audio, but the main "
           "cause of repetition loops. Off for chunks.", local_only=True),
    Option("repetition_penalty", "Repetition penalty", "decoding", "float", 1.0,
           "Above 1 discourages repeated tokens.", min=1.0, max=2.0, step=0.05, local_only=True),
    Option("no_repeat_ngram_size", "No-repeat n-gram size", "decoding", "int", 0,
           "Blocks any n-gram from repeating (0 = off). 3–5 kills loops but can hurt legit repetition.",
           min=0, max=10, step=1, local_only=True),
    # --- hallucination filters -------------------------------------------------------------------
    Option("no_speech_threshold", "No-speech threshold", "filters", "float", 0.6,
           "Drop a segment when Whisper thinks it is silence above this probability (and confidence is low).",
           min=0.0, max=1.0, step=0.05),
    Option("logprob_threshold", "Log-probability threshold", "filters", "float", -1.0,
           "Segments below this average confidence count as failed decodes.", min=-5.0, max=0.0, step=0.1),
    Option("compression_ratio_threshold", "Compression-ratio threshold", "filters", "float", 2.4,
           "Above this the text is a repetition loop and is dropped.", min=1.0, max=5.0, step=0.1),
    Option("hallucination_silence_threshold", "Skip silence hallucinations (s)", "filters", "float", 0.0,
           "When > 0, skips silent stretches longer than this if a hallucination is detected there "
           "(uses word timestamps; slower). 0 = off.", min=0.0, max=10.0, step=0.5, local_only=True),
    # --- after transcription ----------------------------------------------------------------------
    Option("ai_finish", "AI finishing", "after", "bool", True,
           "Clean the text (fillers, false starts, ASR artefacts), then write a title, description and tags. "
           "Needs an OpenAI key in Settings. The raw transcript is always kept.", advanced=False),
    Option("sync_supabase", "Send to Supabase", "after", "bool", True,
           "Upsert the transcript and its metadata into your Supabase table. Needs Settings → Supabase.",
           advanced=False),
]

BY_KEY: Dict[str, Option] = {o.key: o for o in CATALOG}

PRESETS: Dict[str, Dict[str, Any]] = {
    "accuracy": {},  # the catalogue defaults ARE maximum accuracy
    "balanced": {"chunk_target_sec": 12, "chunk_max_sec": 25, "beam_size": 5, "best_of": 5},
    "fast": {"model": "whisper-large-v3-turbo", "chunk_target_sec": 20, "chunk_max_sec": 35,
             "beam_size": 1, "best_of": 1},
}
PRESET_LABELS = [("accuracy", "Maximum accuracy"), ("balanced", "Balanced"), ("fast", "Fast")]


def defaults() -> Dict[str, Any]:
    return {o.key: o.default for o in CATALOG}


def _coerce(opt: Option, value: Any) -> Any:
    if opt.type == "bool":
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("1", "true", "on", "yes")
    if opt.type in ("int", "float"):
        try:
            num = float(value)
        except (TypeError, ValueError):
            return opt.default
        if opt.min is not None:
            num = max(opt.min, num)
        if opt.max is not None:
            num = min(opt.max, num)
        return int(round(num)) if opt.type == "int" else round(num, 4)
    text = "" if value is None else str(value)
    if opt.type == "choice" and opt.choices:
        return text if text in {c[0] for c in opt.choices} else opt.default
    return text.strip() if opt.type == "text" else text


def resolve(
    raw: Optional[Dict[str, Any]] = None,
    base: Optional[Dict[str, Any]] = None,
    preset: Optional[str] = None,
) -> Dict[str, Any]:
    """Full, validated options: catalogue defaults < base < preset < raw.

    Unknown keys are dropped, numbers are clamped to their ranges and bad values
    fall back to the default, so anything from a form or API is safe to run.
    """
    out = defaults()
    for layer in (base or {}, PRESETS.get(preset or "", {}), raw or {}):
        for key, value in layer.items():
            if key in BY_KEY:
                out[key] = _coerce(BY_KEY[key], value)
    if out["chunk_max_sec"] < out["chunk_target_sec"]:
        out["chunk_max_sec"] = out["chunk_target_sec"]
    out["preset"] = preset if preset in PRESETS else (base or {}).get("preset", "custom")
    return out


def from_form(form: Any, prefix: str = "opt_") -> Dict[str, Any]:
    """Pick catalogue keys out of a form/MultiDict. Unchecked checkboxes are False."""
    raw: Dict[str, Any] = {}
    present = set(form.keys())
    for o in CATALOG:
        name = prefix + o.key
        if o.type == "bool":
            # A checkbox is absent when unticked; a hidden "<name>__present" marks the field was shown.
            if name in present or f"{name}__present" in present:
                raw[o.key] = name in present and form.get(name) not in ("0", "false", "off")
        elif name in present:
            raw[o.key] = form.get(name)
    return raw


def candidate_list(opts: Dict[str, Any]) -> Optional[List[str]]:
    if opts.get("language_mode") != "candidates":
        return None
    langs = [c.strip().lower() for c in str(opts.get("candidate_languages", "")).split(",") if c.strip()]
    return langs or None


def fixed_language(opts: Dict[str, Any]) -> Optional[str]:
    if opts.get("language_mode") != "fixed":
        return None
    return (opts.get("language") or "").strip().lower() or None


def decode_extra(opts: Dict[str, Any]) -> Dict[str, Any]:
    """Form fields for the local GPU server (sent via the OpenAI SDK's extra_body)."""
    extra: Dict[str, Any] = {
        "beam_size": opts["beam_size"],
        "best_of": opts["best_of"],
        "patience": opts["patience"],
        "temperature_fallback": opts["temperature_fallback"],
        "condition_on_previous_text": opts["condition_on_previous_text"],
        "repetition_penalty": opts["repetition_penalty"],
        "no_repeat_ngram_size": opts["no_repeat_ngram_size"],
        "no_speech_threshold": opts["no_speech_threshold"],
        "log_prob_threshold": opts["logprob_threshold"],
        "compression_ratio_threshold": opts["compression_ratio_threshold"],
    }
    if opts.get("hotwords"):
        extra["hotwords"] = opts["hotwords"]
    if opts.get("hallucination_silence_threshold"):
        extra["hallucination_silence_threshold"] = opts["hallucination_silence_threshold"]
    if opts.get("segmentation") == "native":
        extra["vad_filter"] = True
        extra["vad_threshold"] = opts["vad_threshold"]
        extra["vad_min_silence_ms"] = opts["vad_min_silence_ms"]
        extra["vad_speech_pad_ms"] = opts["vad_speech_pad_ms"]
        extra["vad_min_speech_ms"] = opts["vad_min_speech_ms"]
    return extra


def apply_profile(opts: Dict[str, Any], profile: Any) -> Dict[str, Any]:
    """Fill prompt / hotwords / candidate languages from a profile where the job left them empty."""
    if profile is None:
        return opts
    out = dict(opts)
    if not out.get("prompt") and getattr(profile, "prompt", ""):
        out["prompt"] = profile.prompt
    if not out.get("hotwords") and getattr(profile, "hotwords", ""):
        out["hotwords"] = profile.hotwords
    langs = getattr(profile, "languages", "") or ""
    if langs and out.get("language_mode") == "auto":
        out["language_mode"] = "candidates"
        out["candidate_languages"] = langs
    out["profile_name"] = getattr(profile, "name", "")
    return out


def summary(opts: Dict[str, Any]) -> str:
    """One human line describing the important choices, for logs and the library."""
    model = "large-v3-turbo" if "turbo" in str(opts.get("model")) else "large-v3"
    parts = [model]
    mode = opts.get("language_mode")
    if mode == "fixed" and opts.get("language"):
        parts.append(f"lang {opts['language']}")
    elif mode == "candidates":
        parts.append(f"race {opts.get('candidate_languages')}")
    if opts.get("segmentation") == "native":
        parts.append("native VAD")
    else:
        parts.append(f"chunks {opts.get('chunk_target_sec')}/{opts.get('chunk_max_sec')}s")
    parts.append(f"beam {opts.get('beam_size')}")
    if opts.get("normalize"):
        parts.append("normalized")
    if opts.get("denoise") and opts.get("denoise") != "off":
        parts.append(f"denoise {opts['denoise']}")
    if opts.get("profile_name"):
        parts.append(f"profile {opts['profile_name']}")
    elif opts.get("prompt") or opts.get("hotwords"):
        parts.append("custom vocabulary")
    return ", ".join(parts)
