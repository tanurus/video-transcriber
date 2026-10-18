"""Chunk-level sessions (/api/v2) on real ffmpeg audio with a scripted Whisper."""
import io
import json
import shutil
import subprocess

import pytest

from app.whisper_client import ChunkResult

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")

LITHUANIAN = "Taigi kurias pasak ir koks klausimas ir kaip atsakyti"
LOOP = "nesu pasak nesu pasak nesu pasak nesu pasak"


class ScriptedClient:
    """Auto-detect drifts to Lithuanian on chunk 1; a ro/ru/en race fixes it."""

    def __init__(self, opts):
        self.opts = opts
        self.model = opts["model"]

    def transcribe_detailed(self, path, log):
        name = path.name
        chunk = int(name.split("-")[0])
        # Each chunk holds one 6 s tone; a healthy decode covers it.
        seg = {"start": 0.5, "end": 5.9, "avg_logprob": -0.25, "no_speech_prob": 0.01, "compression_ratio": 1.3}
        if self.opts["language_mode"] == "candidates":
            return ChunkResult(f"Bună ziua, chunk {chunk}.", [dict(seg, text="Bună ziua")], "ro",
                               race_scores={"ro": -0.2, "ru": -0.9, "en": -1.1})
        if chunk == 1:
            return ChunkResult(LITHUANIAN, [dict(seg, text=LITHUANIAN)], "lt", language_probability=0.41)
        if chunk == 2:
            return ChunkResult(LOOP, [dict(seg, text=LOOP)], "ro")
        return ChunkResult(f"Hello from chunk {chunk}.", [dict(seg, text="Hello")], "en")


@pytest.fixture
def api(client, settings, tmp_path):
    client.post("/settings/tokens", data={"name": "n8n"})
    with client.session_transaction() as s:
        token = s["new_token"]["token"]
    client.app.config["SESSIONS"].client_factory = lambda cfg, opts, timeout=None: ScriptedClient(opts)
    wav = tmp_path / "meeting.wav"
    # 4 tones separated by 1.5 s of silence -> 4 chunks at 5/8 s.
    graph = ";".join(f"sine=f={300 + 100 * i}:d=6" + (",apad=pad_dur=1.5" if i < 3 else "") + f"[a{i}]"
                     for i in range(4))
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i",
                    graph + ";" + "".join(f"[a{i}]" for i in range(4)) + "concat=n=4:v=0:a=1",
                    "-ar", "16000", "-ac", "1", str(wav)], check=True)
    client.h = {"Authorization": f"Bearer {token}"}
    client.wav = wav
    return client


def upload_validated(api):
    resp = api.post("/api/v1/jobs", headers=api.h, content_type="multipart/form-data",
                    data={"file": (io.BytesIO(api.wav.read_bytes()), "2026-10-19 10-00-00.wav"),
                          "pipeline": "validated"})
    assert resp.status_code == 201
    return resp.get_json()["job"]["id"]


def open_session(api, job_id, **extra):
    r = api.post("/api/v2/sessions", headers=api.h, json={"job_id": job_id, "chunk_target_sec": 5,
                                                          "chunk_max_sec": 8, **extra})
    assert r.status_code == 201, r.get_json()
    return r.get_json()


def tx(api, sid, cid, **params):
    r = api.post(f"/api/v2/sessions/{sid}/chunks/{cid}/transcribe", headers=api.h, json=params)
    assert r.status_code == 200, r.get_json()
    return r.get_json()


def test_requires_a_token(api):
    assert api.post("/api/v2/sessions", json={"job_id": "x"}).status_code == 401


def test_validated_upload_waits_for_the_pipeline(api):
    job_id = upload_validated(api)
    assert api.storage.get_job(job_id).status == "received"
    assert api.jobs.submitted == []  # not transcribed by the standard pipeline


def test_session_plan_covers_the_recording(api):
    s = open_session(api, upload_validated(api))
    chunks = s["chunks"]
    assert s["chunk_count"] == 4 and s["allowed_languages"] == ["ro", "ru", "en"]
    assert chunks[0]["start"] == 0.0 and abs(chunks[-1]["end"] - s["duration_sec"]) < 0.01
    assert all(abs(a["end"] - b["start"]) < 1e-6 for a, b in zip(chunks, chunks[1:]))
    assert api.storage.get_job(s["job_id"]).audio_path  # kept lossless audio created


def test_retry_ladder_fixes_the_lithuanian_drift(api):
    s = open_session(api, upload_validated(api))
    sid, c1 = s["session_id"], s["chunks"][1]
    first = tx(api, sid, 1, label="default")
    assert first["passed"] is False and first["checks"]["problems"] == ["wrong_language:lt"]
    assert first["checks"]["signals"]["language_probability"] == 0.41
    assert first["segments"][0]["start"] == round(c1["start"] + 0.5, 2)  # on the recording's timeline
    second = tx(api, sid, 1, candidate_languages=["ro", "ru", "en"], label="race")
    assert second["passed"] and second["language"] == "ro"
    assert second["checks"]["signals"]["race_scores"]["ro"] == -0.2
    assert second["params"]["candidate_languages"] == ["ro", "ru", "en"]
    loop = tx(api, sid, 2)
    assert "loop:repeated_phrase" in loop["checks"]["problems"]
    state = api.get(f"/api/v2/sessions/{sid}", headers=api.h).get_json()
    assert state["chunks"][1]["best_attempt_id"] == second["attempt_id"]
    assert state["chunks"][1]["attempts"] == 2


def test_external_validation_is_stored_and_ranked(api):
    s = open_session(api, upload_validated(api))
    sid = s["session_id"]
    a = tx(api, sid, 0)
    b = tx(api, sid, 0, beam_size=3)
    api.post(f"/api/v2/sessions/{sid}/attempts/{a['attempt_id']}/validation", headers=api.h,
             json={"passed": False, "jev": {"language": "other", "p": 0.9}})
    v = api.post(f"/api/v2/sessions/{sid}/attempts/{b['attempt_id']}/validation", headers=api.h,
                 json={"passed": True}).get_json()
    assert v["external"] == {"passed": True}
    state = api.get(f"/api/v2/sessions/{sid}?attempts=1", headers=api.h).get_json()
    assert state["chunks"][0]["best_attempt_id"] == b["attempt_id"]
    assert len(state["chunks"][0]["attempt_list"]) == 2


def test_recut_moves_neighbours_and_marks_old_attempts_stale(api):
    s = open_session(api, upload_validated(api))
    sid, c = s["session_id"], s["chunks"]
    old = tx(api, sid, 1)
    new_start = c[1]["start"] + 1.0
    r = api.post(f"/api/v2/sessions/{sid}/chunks/1/recut", headers=api.h,
                 json={"start": new_start, "end": c[1]["end"]}).get_json()
    assert r["chunks"][0]["end"] == round(new_start, 3) and r["chunks"][1]["start"] == round(new_start, 3)
    stale = api.get(f"/api/v2/sessions/{sid}/chunks/1/attempts", headers=api.h).get_json()["attempts"][0]
    assert stale["stale"] is True
    bad = api.post(f"/api/v2/sessions/{sid}/assemble", headers=api.h, json={"choices": {"1": old["attempt_id"]}})
    assert bad.status_code == 400 and "re-cut" in bad.get_json()["error"]
    assert api.post(f"/api/v2/sessions/{sid}/chunks/1/recut", headers=api.h,
                    json={"start": 0.1, "end": 0.5}).status_code == 400


def test_split_adds_a_new_stable_id(api):
    s = open_session(api, upload_validated(api))
    sid, c3 = s["session_id"], s["chunks"][3]
    r = api.post(f"/api/v2/sessions/{sid}/chunks/3/split", headers=api.h, json={}).get_json()
    assert r["new_chunk_id"] == 4 and r["chunk_count"] == 5
    assert [ch["chunk_id"] for ch in r["chunks"]] == [0, 1, 2, 3, 4]
    assert r["chunks"][3]["end"] == r["chunks"][4]["start"] and r["chunks"][4]["end"] == c3["end"]


def test_assemble_fills_the_waiting_job_and_reports_review_items(api, settings):
    job_id = upload_validated(api)
    s = open_session(api, job_id)
    sid = s["session_id"]
    for cid in (0, 3):
        tx(api, sid, cid)
    tx(api, sid, 1)
    fixed = tx(api, sid, 1, candidate_languages="ro,ru,en")
    tx(api, sid, 2)  # loop, never fixed -> needs review
    out = api.post(f"/api/v2/sessions/{sid}/assemble", headers=api.h,
                   json={"choices": {"1": fixed["attempt_id"]}, "report": {"workflow": "transcribe__validated"}}).get_json()
    assert out["job_id"] == job_id and out["needs_review"] == [2] and out["chunks"] == 4
    job = api.storage.get_job(job_id)
    assert job.status == "done" and job.settings["pipeline"] == "validated" and "2 need" not in (job.error or "")
    text = (settings.data_dir / job.transcript_path).read_text()
    assert "Bună ziua, chunk 1." in text and "Taigi" not in text
    assert text.count("\n\n") >= 2  # paragraph breaks where the language changes
    report = json.loads((settings.data_dir / "reports" / f"{job_id}.json").read_text())
    assert report["needs_review"] == [2] and report["caller"] == {"workflow": "transcribe__validated"}
    assert set(job.languages.split(",")) == {"en", "ro"}
    assert api.post(f"/api/v2/sessions/{sid}/chunks/0/transcribe", headers=api.h, json={}).status_code == 409
    assert api.post(f"/api/v2/sessions/{sid}/assemble", headers=api.h, json={}).status_code == 409


def test_assemble_on_a_finished_job_creates_a_new_version(api):
    job_id = upload_validated(api)
    for _ in range(2):
        s = open_session(api, job_id)
        for ch in s["chunks"]:
            tx(api, s["session_id"], ch["chunk_id"], candidate_languages=["ro", "ru", "en"])
        out = api.post(f"/api/v2/sessions/{s['session_id']}/assemble", headers=api.h, json={}).get_json()
    assert out["job_id"] != job_id
    v2 = api.storage.get_job(out["job_id"])
    assert v2.parent_id == job_id and v2.version == 2 and out["needs_review"] == []


def test_missing_attempts_and_expired_audio(api):
    s = open_session(api, upload_validated(api))
    sid = s["session_id"]
    r = api.post(f"/api/v2/sessions/{sid}/assemble", headers=api.h, json={"fallback": "best"})
    assert r.status_code == 400 and "0, 1, 2, 3" in r.get_json()["error"]
    shutil.rmtree(api.jobs.data_dir / "sessions" / sid)
    assert api.post(f"/api/v2/sessions/{sid}/chunks/0/transcribe", headers=api.h, json={}).status_code == 410
    assert api.post("/api/v2/sessions", headers=api.h, json={"job_id": "nope"}).status_code == 404


class TranslatingClient(ScriptedClient):
    """Chunk 0 decodes as Romanian; its second segment's own audio is Russian."""

    def transcribe_detailed(self, path, log):
        if "-" in path.stem and path.parent.name == "probe":
            idx = int(path.stem.rsplit("-", 1)[1])
            return ChunkResult("Хорошо, тогда" if idx == 1 else "Noi le punem", [], "ru" if idx == 1 else "ro")
        seg = {"avg_logprob": -0.2, "no_speech_prob": 0.01, "compression_ratio": 1.2}
        return ChunkResult("Noi le punem egal. Bine, atunci Andrei pregătește",
                           [dict(seg, start=0.0, end=2.5, text="Noi le punem egal."),
                            dict(seg, start=3.0, end=5.5, text="Bine, atunci Andrei pregătește")], "ro")


def test_probe_finds_a_translated_sentence_and_suggests_the_split(api):
    api.app.config["SESSIONS"].client_factory = lambda cfg, opts, timeout=None: TranslatingClient(opts)
    s = open_session(api, upload_validated(api))
    sid = s["session_id"]
    a = tx(api, sid, 0)
    assert a["passed"]  # text checks alone cannot see a translation
    p = api.post(f"/api/v2/sessions/{sid}/chunks/0/probe", headers=api.h, json={}).get_json()
    assert p["mismatch"] is True and p["attempt_language"] == "ro"
    assert [x.get("detected_language") for x in p["segments"]] == ["ro", "ru"]
    assert p["suggested_split"] == 3.0
    r = api.post(f"/api/v2/sessions/{sid}/chunks/0/split", headers=api.h, json={"at": p["suggested_split"]}).get_json()
    assert r["chunks"][0]["end"] == 3.0 and r["chunks"][1]["start"] == 3.0


class DroppingClient(ScriptedClient):
    def transcribe_detailed(self, path, log):
        seg = {"avg_logprob": -0.2, "no_speech_prob": 0.01, "compression_ratio": 1.2}
        return ChunkResult("Стоимость лида снизилась", [dict(seg, start=0.2, end=2.0, text="Стоимость лида")], "ru")


def test_low_coverage_suggests_a_split_at_the_gap(api):
    api.app.config["SESSIONS"].client_factory = lambda cfg, opts, timeout=None: DroppingClient(opts)
    s = open_session(api, upload_validated(api))
    c0 = s["chunks"][0]
    a = tx(api, s["session_id"], 0)
    assert "low_coverage" in a["checks"]["problems"]
    assert a["checks"]["suggested_split"] == round(c0["start"] + 2.15, 2)
