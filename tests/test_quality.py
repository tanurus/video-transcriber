from app.quality import check_chunk, repeated_ngram


def seg(start=0.0, end=5.0, lp=-0.3, ns=0.02, cr=1.2):
    return {"start": start, "end": end, "avg_logprob": lp, "no_speech_prob": ns, "compression_ratio": cr}


def test_repeated_phrase_from_andrews_sample():
    assert repeated_ngram("mes padar bukinguri nesu pasak kad nesu pasak nesu pasak nesu pasak nesu pasak") == \
        {"phrase": "nesu pasak", "repeats": 4}
    assert repeated_ngram("Dar pe la muncii se vând. Dar pe la muncii se vând. Dar pe la muncii se vând. Noi") == \
        {"phrase": "dar pe la muncii se vând", "repeats": 3}


def test_natural_repetition_is_not_a_loop():
    assert repeated_ngram("no, no, no, I disagree") is None
    assert repeated_ngram("Apoi am pornit autobiderea, acolo am făcut un upgrade la screping") is None


def test_wrong_language_fails():
    r = check_chunk("Taigi kurias pasak", [seg()], "lt")
    assert r["passed"] is False and r["problems"] == ["wrong_language:lt"]


def test_clean_mixed_chunk_passes():
    r = check_chunk("И спойник типа, it's not about the money", [seg()], "ru")
    assert r["passed"] and not r["borderline"]


def test_compression_ratio_and_empty_and_borderline():
    assert "loop:compression_ratio" in check_chunk("x y z", [seg(cr=3.1)], "ro")["problems"]
    assert check_chunk("", [], "ro", dropped=[("Thanks for watching", "no speech")])["problems"] == ["empty_after_filter"]
    shaky = check_chunk("Bună ziua tuturor", [seg(lp=-0.95)], "ro")
    assert shaky["passed"] and shaky["borderline"]


def test_allowed_set_is_configurable():
    assert check_chunk("Добрий день", [seg()], "uk", allowed_languages=["ro", "ru", "en", "uk"])["passed"]


def test_low_coverage_flags_a_dropped_sentence():
    # Live 2026-10-19: a RU+EN chunk of 11.6 s came back as one 4 s Russian segment.
    r = check_chunk("Стоимость лида снизилась", [seg(0.0, 4.0)], "ru", chunk_duration=11.6)
    assert "low_coverage" in r["problems"]
    assert r["signals"]["speech_coverage"] == 0.345 and r["signals"]["largest_gap"] == {"start": 4.0, "end": 11.6}


def test_normal_pauses_do_not_trigger_low_coverage():
    r = check_chunk("Bună ziua. Cum merge", [seg(0.2, 4.0), seg(4.6, 8.4)], "ro", chunk_duration=9.0)
    assert r["passed"] and r["signals"]["speech_coverage"] > 0.8
