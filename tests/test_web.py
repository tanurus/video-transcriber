import io
import json

import pytest


def test_run_stores_text_segments_and_metadata(client, done_job, settings):
    job = done_job
    assert job.status == "done"
    assert (settings.data_dir / job.transcript_path).read_text() == "the transcript"
    assert json.loads((settings.data_dir / job.segments_path).read_text())[0]["text"] == "the transcript"
    assert job.languages == "en" and job.model == "whisper-large-v3" and job.audio_path
    lines, _ = client.jobs.get_logs("j1")
    assert "Done." in lines and "transcribing" in lines
    assert any("AI finishing skipped" in l for l in lines)  # no key yet: says so, never fails


def test_logs_survive_a_restart_on_disk(client, done_job):
    client.jobs._logs.clear()  # as after a process restart
    lines, nxt = client.jobs.get_logs("j1")
    assert "Done." in lines and nxt == len(lines)


def test_run_error_is_recorded_never_raised(make_app, settings):
    def boom(*a, **k):
        raise RuntimeError("ffmpeg exploded")

    c = make_app(transcribe_fn=boom)
    c.storage.create_job("j2", "bad.mp4", "2026-09-30T10:00:00+00:00")
    (settings.uploads_dir / "j2").mkdir()
    (settings.uploads_dir / "j2" / "bad.mp4").write_bytes(b"x")
    c.jobs._run("j2")
    job = c.storage.get_job("j2")
    assert job.status == "error" and "ffmpeg exploded" in job.error


def test_run_without_any_audio_fails_clearly(client):
    client.storage.create_job("j3", "gone.mp4", "2026-09-30T10:00:00+00:00")
    client.jobs._run("j3")
    assert "both gone" in client.storage.get_job("j3").error


def test_waits_for_a_down_gpu_then_fails_with_its_message(make_app, settings):
    c = make_app()

    def down():
        raise RuntimeError("Local Whisper server at x is not responding (refused)")

    c.jobs.cfg_loader = down
    c.storage.create_job("j4", "a.mp4", "2026-09-30T10:00:00+00:00")
    (settings.uploads_dir / "j4").mkdir()
    (settings.uploads_dir / "j4" / "a.mp4").write_bytes(b"x")
    c.jobs._run("j4")  # backend_wait=0 -> no waiting in tests
    assert "not responding" in c.storage.get_job("j4").error


# --- pages -------------------------------------------------------------------------------

def test_pages_render(client, done_job):
    for url in ("/", "/library", "/settings", f"/job/{done_job.id}"):
        resp = client.get(url)
        assert resp.status_code == 200, url
    assert b"Meeting.mp4" in client.get("/library").data
    assert b"the transcript" in client.get("/job/j1").data


def test_history_redirects_to_library(client):
    assert client.get("/history").status_code == 302


def test_job_page_404_and_error_message(client):
    assert client.get("/job/missing").status_code == 404
    client.storage.create_job("e", "E.mp4", "2026-09-30T10:00:00+00:00")
    client.storage.update_status("e", "error", error="ffmpeg exploded: no audio stream")
    assert b"ffmpeg exploded: no audio stream" in client.get("/job/e").data


def test_job_api_streams_lines_and_clamps_since(client, done_job):
    body = client.get("/api/job/j1?since=-5").get_json()
    assert body["status"] == "done" and body["download_ready"] is True
    assert "Done." in body["lines"] and body["next_index"] == len(body["lines"])


# --- downloads -----------------------------------------------------------------------------

def test_download_kinds(client, done_job):
    resp = client.get("/download/j1?kind=raw")
    assert resp.data == b"the transcript"
    assert "2026-09-30" in resp.headers["Content-Disposition"] and "Meeting" in resp.headers["Content-Disposition"]
    assert b"00:00:00,000 --> 00:00:02,000" in client.get("/download/j1?kind=srt").data
    record = json.loads(client.get("/download/j1?kind=json").data)
    assert record["id"] == "j1" and record["transcript_raw"] == "the transcript"
    assert client.get("/download/j1?kind=clean").status_code == 404  # no AI pass yet


def test_download_uses_title_once_known(client, done_job):
    client.storage.update_job("j1", title="Kayak review")
    assert "Kayak review.txt" in client.get("/download/j1?kind=raw").headers["Content-Disposition"]


def test_download_rejects_path_escape(client, done_job):
    client.storage.update_job("j1", transcript_path="../../etc/passwd")
    assert client.get("/download/j1?kind=raw").status_code == 404


def test_zip_download(client, done_job):
    import zipfile
    resp = client.get("/download.zip?ids=j1,missing&kind=raw")
    names = zipfile.ZipFile(io.BytesIO(resp.data)).namelist()
    assert len(names) == 1 and names[0].endswith(".txt")


# --- plain form upload (no JavaScript) -----------------------------------------------------------

def test_form_upload_multiple_files(client, settings):
    data = {"video": [(io.BytesIO(b"one"), "A.mp4"), (io.BytesIO(b"two"), "B.m4a"), (io.BytesIO(b"x"), "evil.exe")],
            "preset": "fast", "opt_normalize__present": "1"}
    resp = client.post("/upload", data=data, content_type="multipart/form-data")
    assert resp.status_code == 302
    jobs = client.storage.list_jobs()
    assert sorted(j.original_name for j in jobs) == ["A.mp4", "B.m4a"]
    assert len(client.jobs.submitted) == 2
    opts = jobs[0].settings
    assert opts["model"] == "whisper-large-v3-turbo" and opts["normalize"] is False


def test_form_upload_non_ascii_name_keeps_extension(client, settings):
    client.post("/upload", data={"video": (io.BytesIO(b"v"), "Видео.mp4")}, content_type="multipart/form-data")
    job = client.storage.list_jobs()[0]
    assert job.original_name == "Видео.mp4"
    saved = list((settings.uploads_dir / job.id).iterdir())
    assert saved[0].suffix == ".mp4"


def test_form_upload_without_files_flashes(client):
    assert client.post("/upload", data={}).status_code == 302
    assert client.storage.list_jobs() == []


# --- chunked, resumable upload -------------------------------------------------------------------------

def _start(client, name="Call 2026-09-29 10-15-00.m4a", size=10, lm=1759140000000):
    return client.post("/api/uploads", json={"filename": name, "size": size, "last_modified": lm})


def test_chunked_upload_with_retry_and_resume(client):
    up = _start(client).get_json()
    uid = up["upload_id"]
    assert client.put(f"/api/uploads/{uid}?offset=0", data=b"01234").get_json()["received"] == 5
    # the same chunk retried after a lost response: accepted, nothing duplicated
    assert client.put(f"/api/uploads/{uid}?offset=0", data=b"01234").get_json()["received"] == 5
    # a gap is refused with 409 and the client resyncs from GET
    assert client.put(f"/api/uploads/{uid}?offset=7", data=b"xyz").status_code == 409
    assert client.get(f"/api/uploads/{uid}").get_json()["received"] == 5
    assert client.post(f"/api/uploads/{uid}/complete", json={}).status_code == 409  # incomplete
    client.put(f"/api/uploads/{uid}?offset=5", data=b"56789")
    done = client.post(f"/api/uploads/{uid}/complete", json={"preset": "accuracy",
                                                             "options": {"beam_size": 3}},
                       headers={"User-Agent": "Mozilla/5.0 (Linux; Android 14; SM-S918B) Chrome/129"}).get_json()
    job = client.storage.get_job(done["job"]["id"])
    assert done["duplicate"] is False and client.jobs.submitted == [job.id]
    assert job.recorded_at == "2026-09-29T10:15:00"  # from the file name, not the phone's mtime
    assert job.source == "web" and job.source_detail.startswith("Android (SM-S918B)")
    assert job.file_size == 10 and len(job.file_sha256) == 64
    assert job.settings["beam_size"] == 3


def test_same_recording_from_two_devices_is_not_transcribed_twice(client):
    ids = []
    for _ in range(2):
        uid = _start(client, name="memo.m4a", size=3).get_json()["upload_id"]
        client.put(f"/api/uploads/{uid}?offset=0", data=b"abc")
        ids.append(client.post(f"/api/uploads/{uid}/complete", json={}).get_json())
    assert ids[1]["duplicate"] is True and ids[1]["job"]["id"] == ids[0]["job"]["id"]
    assert len(client.storage.list_jobs()) == 1 and len(client.jobs.submitted) == 1


def test_upload_rejections(client):
    assert _start(client, name="virus.exe").status_code == 400
    assert _start(client, size=0).status_code == 400
    assert _start(client, size=10**12).status_code == 413
    assert client.put("/api/uploads/nothex?offset=0", data=b"x").status_code == 404
    uid = _start(client, size=3).get_json()["upload_id"]
    assert client.put(f"/api/uploads/{uid}?offset=0", data=b"abcdef").status_code == 400  # over declared size


# --- editing, bulk actions ---------------------------------------------------------------------------------

def test_edit_title_description_tags(client, done_job):
    body = client.post("/api/job/j1/meta", json={"title": " Weekly sync ", "tags": "Kayak, Budget"}).get_json()
    assert body["title"] == "Weekly sync" and body["tags"] == ["kayak", "budget"]


def test_bulk_sync_and_ai_need_connections(client, done_job):
    assert "Supabase" in client.post("/api/jobs/bulk", json={"action": "sync", "ids": ["j1"]}).get_json()["error"]
    assert "OpenAI" in client.post("/api/jobs/bulk", json={"action": "ai", "ids": ["j1"]}).get_json()["error"]
    client.conns.update({"supabase_url": "https://x.supabase.co", "supabase_key": "k"})
    assert client.post("/api/jobs/bulk", json={"action": "sync", "ids": ["j1"]}).get_json()["count"] == 1
    assert client.storage.get_job("j1").sync_status == "pending"


def test_bulk_regenerate_creates_versions_from_kept_audio(client, done_job, settings):
    import shutil
    shutil.rmtree(settings.uploads_dir / "j1")  # video purged by retention
    r = client.post("/api/jobs/bulk", json={"action": "regenerate", "ids": ["j1"], "preset": "fast",
                                            "options": {"denoise": "light"}, "keep_base": True}).get_json()
    new = client.storage.get_job(r["created"][0])
    assert new.parent_id == "j1" and new.version == 2 and new.file_sha256 == "abc"
    assert new.settings["model"] == "whisper-large-v3-turbo" and new.settings["denoise"] == "light"
    client.jobs._run(new.id)
    assert client.storage.get_job(new.id).status == "done"
    assert [j.id for j in client.storage.versions_of(new.id)] == ["j1", new.id]
    # the library hides superseded versions by default
    assert b"v2" in client.get("/library").data


def test_bulk_delete_removes_files(client, done_job, settings):
    audio = done_job.audio_path
    client.post("/api/jobs/bulk", json={"action": "delete", "ids": ["j1"]})
    assert client.storage.get_job("j1") is None
    assert not (settings.data_dir / done_job.transcript_path).exists()
    import os
    assert not os.path.exists(audio)


# --- settings --------------------------------------------------------------------------------------

def test_connections_are_masked_and_blank_keeps_secret(client, settings):
    client.post("/settings/connections", data={"openai_api_key": "sk-secret-1234", "openai_model": "m"})
    client.post("/settings/connections", data={"openai_api_key": "", "openai_model": "m2"})
    conns = client.conns.load()
    assert conns.openai_api_key == "sk-secret-1234" and conns.openai_model == "m2"
    page = client.get("/settings").data
    assert b"sk-secret-1234" not in page and b"1234" in page
    import os, stat
    assert stat.S_IMODE(os.stat(settings.data_dir / "connections.json").st_mode) == 0o600


def test_defaults_apply_to_new_uploads_on_every_device(client):
    client.post("/api/settings/defaults", json={"preset": "fast", "options": {"hotwords": "Kayak"}})
    uid = _start(client, name="x.mp4", size=1).get_json()["upload_id"]
    client.put(f"/api/uploads/{uid}?offset=0", data=b"z")
    job = client.post(f"/api/uploads/{uid}/complete", json={}).get_json()["job"]
    opts = client.storage.get_job(job["id"]).settings
    assert opts["model"] == "whisper-large-v3-turbo" and opts["hotwords"] == "Kayak"


def test_profile_is_applied_at_upload(client):
    client.post("/settings/profiles", data={"name": "Econ", "prompt": "Daily sync", "hotwords": "AranGrant",
                                            "languages": "ru,ro,en"})
    pid = client.storage.list_profiles()[0].id
    uid = _start(client, name="y.mp4", size=1).get_json()["upload_id"]
    client.put(f"/api/uploads/{uid}?offset=0", data=b"q")
    job = client.post(f"/api/uploads/{uid}/complete", json={"options": {"profile_id": str(pid)}}).get_json()["job"]
    opts = client.storage.get_job(job["id"]).settings
    assert opts["prompt"] == "Daily sync" and opts["language_mode"] == "candidates"
    assert "profile Econ" in opts["_summary"]


# --- token API ---------------------------------------------------------------------------------------

def test_api_token_flow(client):
    assert client.post("/api/v1/jobs").status_code == 401
    client.post("/settings/tokens", data={"name": "S23 phone"})
    with client.session_transaction() as s:
        token = s["new_token"]["token"]
    assert b"S23 phone" in client.get("/settings").data  # shows once...
    with client.session_transaction() as s:
        assert "new_token" not in s  # ...then it is gone
    h = {"Authorization": f"Bearer {token}"}
    assert client.post("/api/v1/jobs", headers={"Authorization": "Bearer nope"}).status_code == 401
    resp = client.post("/api/v1/jobs", headers=h, content_type="multipart/form-data",
                       data={"file": (io.BytesIO(b"rec"), "VID_20260930_091500.mp4"), "source": "phone",
                             "options": json.dumps({"beam_size": 2})})
    assert resp.status_code == 201
    job = client.storage.get_job(resp.get_json()["job"]["id"])
    assert job.source == "phone" and job.source_detail.startswith("S23 phone")
    assert job.recorded_at == "2026-09-30T09:15:00" and job.settings["beam_size"] == 2
    again = client.post("/api/v1/jobs", headers=h, content_type="multipart/form-data",
                        data={"file": (io.BytesIO(b"rec"), "copy.mp4")})
    assert again.status_code == 200 and again.get_json()["duplicate"] is True
    assert client.get(f"/api/v1/jobs/{job.id}", headers=h).get_json()["status"] == "queued"
    assert client.post("/api/v1/jobs", headers=h, data={"options": "[]", "file": (io.BytesIO(b"z"), "a.mp4")},
                       content_type="multipart/form-data").status_code == 400
