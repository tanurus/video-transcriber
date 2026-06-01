def test_wsgi_exposes_app(settings, monkeypatch):
    # Point DATA_DIR at a temp dir so create_app() doesn't touch /data.
    monkeypatch.setenv("DATA_DIR", str(settings.data_dir))
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    import importlib
    import web.wsgi as wsgi
    importlib.reload(wsgi)
    assert wsgi.app is not None
    client = wsgi.app.test_client()
    assert client.get("/healthz").status_code == 200
