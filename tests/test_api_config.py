"""Token-authenticated configuration API (/api/v1 settings, profiles, actions)."""
import hashlib

import pytest


@pytest.fixture
def auth(client):
    token = "vt_test-token"
    client.storage.add_token("mcp", hashlib.sha256(token.encode()).hexdigest(), token[:10])
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.parametrize("method,path", [
    ("get", "/api/v1/settings/catalog"), ("get", "/api/v1/settings/defaults"),
    ("put", "/api/v1/settings/defaults"), ("get", "/api/v1/profiles"), ("post", "/api/v1/profiles"),
    ("delete", "/api/v1/profiles/1"), ("get", "/api/v1/connections"), ("get", "/api/v1/health"),
    ("post", "/api/v1/jobs/actions"), ("get", "/api/v1/jobs/x/segments"),
])
def test_every_config_endpoint_needs_a_token(client, method, path):
    assert getattr(client, method)(path).status_code == 401


def test_catalog_describes_options_presets_and_profiles(client, auth):
    client.storage.save_profile("Sales calls", languages="ro,ru")
    data = client.get("/api/v1/settings/catalog", headers=auth).get_json()
    keys = {o["key"] for o in data["options"]}
    assert {"model", "language_mode", "beam_size", "chunk_target_sec"} <= keys
    model = next(o for o in data["options"] if o["key"] == "model")
    assert {"value": "whisper-large-v3", "label": model["choices"][0]["label"]} == model["choices"][0]
    profile = next(o for o in data["options"] if o["key"] == "profile_id")
    assert [c["label"] for c in profile["choices"]] == ["None", "Sales calls"]
    assert [p["key"] for p in data["presets"]] == ["accuracy", "balanced", "fast"]


def test_defaults_merge_validate_and_apply_to_uploads(client, auth):
    r = client.put("/api/v1/settings/defaults", headers=auth,
                   json={"options": {"language_mode": "fixed", "language": "ro", "beam_size": 999}})
    assert r.status_code == 200
    opts = r.get_json()["options"]
    assert opts["language"] == "ro" and opts["beam_size"] <= 10  # clamped to the catalogue range
    # merge: a second change keeps the first
    client.put("/api/v1/settings/defaults", headers=auth, json={"options": {"model": "whisper-large-v3-turbo"}})
    got = client.get("/api/v1/settings/defaults", headers=auth).get_json()["options"]
    assert got["language"] == "ro" and got["model"] == "whisper-large-v3-turbo"
    # merge=false starts again from the catalogue defaults
    client.put("/api/v1/settings/defaults", headers=auth, json={"preset": "fast", "merge": False})
    got = client.get("/api/v1/settings/defaults", headers=auth).get_json()["options"]
    assert got["language_mode"] == "auto" and got["preset"] == "fast"


def test_defaults_reject_unknown_keys_and_presets(client, auth):
    r = client.put("/api/v1/settings/defaults", headers=auth, json={"options": {"modle": "x"}})
    assert r.status_code == 400 and "modle" in r.get_json()["error"]
    assert client.put("/api/v1/settings/defaults", headers=auth, json={"preset": "turbo"}).status_code == 400
    assert client.put("/api/v1/settings/defaults", headers=auth, data="nope").status_code == 400


def test_profiles_create_update_partially_and_delete(client, auth):
    r = client.post("/api/v1/profiles", headers=auth,
                    json={"name": "Kayak QBR", "hotwords": "Kayak, BWK", "languages": "en"})
    assert r.status_code == 201
    pid = r.get_json()["profile"]["id"]
    r = client.post("/api/v1/profiles", headers=auth, json={"id": pid, "prompt": "Meta-search partner call."})
    p = r.get_json()["profile"]
    assert r.status_code == 200 and p["hotwords"] == "Kayak, BWK" and p["prompt"] == "Meta-search partner call."
    assert client.post("/api/v1/profiles", headers=auth, json={"name": ""}).status_code == 400
    assert client.post("/api/v1/profiles", headers=auth, json={"id": 999, "name": "x"}).status_code == 404
    assert len(client.get("/api/v1/profiles", headers=auth).get_json()["profiles"]) == 1
    assert client.delete(f"/api/v1/profiles/{pid}", headers=auth).status_code == 200
    assert client.delete(f"/api/v1/profiles/{pid}", headers=auth).status_code == 404


def test_connections_never_expose_secrets(client, auth):
    client.conns.update({"openai_api_key": "sk-secret-1234567890", "supabase_key": "sb-secret-abcdefgh"})
    body = client.get("/api/v1/connections", headers=auth).get_data(as_text=True)
    assert "secret" not in body and "7890" not in body and "efgh" not in body
    assert '"ai_ready": true' in body.replace(":true", ": true")


def test_job_list_filters_segments_versions_and_actions(client, auth, done_job):
    jobs = client.get("/api/v1/jobs?q=Meeting&status=done", headers=auth).get_json()["jobs"]
    assert [j["id"] for j in jobs] == ["j1"]
    assert client.get("/api/v1/jobs?q=nothing-like-this", headers=auth).get_json()["jobs"] == []
    one = client.get("/api/v1/jobs/j1", headers=auth).get_json()
    assert "settings" in one and not any(k.startswith("_") for k in one["settings"])
    segs = client.get("/api/v1/jobs/j1/segments", headers=auth).get_json()["segments"]
    assert segs and segs[0]["text"] == "the transcript"
    r = client.post("/api/v1/jobs/actions", headers=auth,
                    json={"action": "regenerate", "ids": ["j1"], "preset": "fast"})
    created = r.get_json()["created"]
    assert len(created) == 1 and created[0] in client.jobs.submitted
    versions = client.get("/api/v1/jobs/j1/versions", headers=auth).get_json()["versions"]
    assert [v["version"] for v in versions] == [1, 2] and versions[1]["settings"]["preset"] == "fast"
    # AI and Supabase need their connections; delete is not offered to tokens.
    assert client.post("/api/v1/jobs/actions", headers=auth, json={"action": "ai", "ids": ["j1"]}).status_code == 400
    r = client.post("/api/v1/jobs/actions", headers=auth, json={"action": "delete", "ids": ["j1"]})
    assert r.status_code == 400 and client.storage.get_job("j1") is not None


def test_health_over_token(client, auth):
    data = client.get("/api/v1/health", headers=auth).get_json()
    assert {"gpu", "counts", "ai_ready", "supabase_ready"} <= set(data)


def test_defaults_round_trip_keeps_the_preset_label(client, auth):
    client.put("/api/v1/settings/defaults", headers=auth, json={"preset": "balanced", "merge": False})
    got = client.get("/api/v1/settings/defaults", headers=auth).get_json()["options"]
    back = client.put("/api/v1/settings/defaults", headers=auth, json={"options": got, "merge": False}).get_json()
    assert back["options"]["preset"] == "balanced"
    got["beam_size"] = 2  # no longer the preset's values -> honestly "custom"
    back = client.put("/api/v1/settings/defaults", headers=auth, json={"options": got, "merge": False}).get_json()
    assert back["options"]["preset"] == "custom"
