from types import SimpleNamespace

from werkzeug.datastructures import MultiDict

from app import options as O


def test_defaults_are_maximum_accuracy():
    d = O.resolve()
    assert d["model"] == "whisper-large-v3"
    assert d["beam_size"] == 5 and d["best_of"] == 5
    assert d["segmentation"] == "chunks"
    assert (d["chunk_target_sec"], d["chunk_max_sec"]) == (8, 15)
    # Measured 2026-09-30: no denoiser beat untouched audio; normalization never hurt.
    assert d["denoise"] == "off"
    assert d["normalize"] is True
    assert d["preset"] == "custom"


def test_fast_preset_switches_to_turbo_and_greedy():
    d = O.resolve(preset="fast")
    assert d["model"] == "whisper-large-v3-turbo"
    assert d["beam_size"] == 1
    assert d["preset"] == "fast"


def test_raw_values_override_preset_and_are_coerced():
    d = O.resolve({"beam_size": "3", "normalize": "off", "denoise": "light"}, preset="fast")
    assert d["beam_size"] == 3
    assert d["normalize"] is False
    assert d["denoise"] == "light"
    assert d["model"] == "whisper-large-v3-turbo"


def test_numbers_are_clamped_and_garbage_falls_back():
    d = O.resolve({"beam_size": "99", "chunk_target_sec": "-5", "patience": "abc"})
    assert d["beam_size"] == 10
    assert d["chunk_target_sec"] == 4
    assert d["patience"] == 1.0


def test_unknown_choice_and_unknown_keys_are_dropped():
    d = O.resolve({"model": "whisper-tiny", "rm -rf": "/"})
    assert d["model"] == "whisper-large-v3"
    assert "rm -rf" not in d


def test_chunk_max_never_below_target():
    d = O.resolve({"chunk_target_sec": 30, "chunk_max_sec": 10})
    assert d["chunk_max_sec"] == 30


def test_base_layer_lets_regenerate_start_from_a_previous_job():
    previous = O.resolve({"beam_size": 3, "prompt": "Kayak"})
    d = O.resolve({"model": "whisper-large-v3-turbo"}, base=previous)
    assert d["beam_size"] == 3 and d["prompt"] == "Kayak"
    assert d["model"] == "whisper-large-v3-turbo"


def test_from_form_handles_unticked_checkboxes():
    form = MultiDict([("opt_beam_size", "4"), ("opt_normalize__present", "1"),
                      ("opt_ai_finish__present", "1"), ("opt_ai_finish", "on")])
    raw = O.from_form(form)
    assert raw == {"beam_size": "4", "normalize": False, "ai_finish": True}


def test_language_helpers():
    assert O.candidate_list(O.resolve({"language_mode": "candidates", "candidate_languages": "RO, ru"})) == ["ro", "ru"]
    assert O.candidate_list(O.resolve()) is None
    assert O.fixed_language(O.resolve({"language_mode": "fixed", "language": "UK"})) == "uk"
    assert O.fixed_language(O.resolve({"language": "uk"})) is None  # auto mode ignores it


def test_decode_extra_carries_native_vad_only_in_native_mode():
    chunks = O.decode_extra(O.resolve({"hotwords": "AranGrant, Kayak"}))
    assert chunks["beam_size"] == 5 and chunks["hotwords"] == "AranGrant, Kayak"
    assert "vad_filter" not in chunks
    native = O.decode_extra(O.resolve({"segmentation": "native", "vad_threshold": 0.35}))
    assert native["vad_filter"] is True and native["vad_threshold"] == 0.35


def test_profile_fills_only_empty_fields():
    prof = SimpleNamespace(name="Economic Potential", prompt="Daily sync", hotwords="Kayak", languages="ru,ro,en")
    d = O.apply_profile(O.resolve({"hotwords": "Mine"}), prof)
    assert d["prompt"] == "Daily sync"
    assert d["hotwords"] == "Mine"  # the job's own value wins
    assert d["language_mode"] == "candidates" and d["candidate_languages"] == "ru,ro,en"
    assert d["profile_name"] == "Economic Potential"


def test_summary_is_readable():
    s = O.summary(O.resolve({"denoise": "light"}, preset="fast"))
    assert "large-v3-turbo" in s and "beam 1" in s and "denoise light" in s
