import importlib
import sys


def test_app_main_imports_without_tkinter(monkeypatch):
    """The transcription core must import on a headless server with no Tk libraries.

    Regression test for the Docker slim image, which lacks libtk: the web worker
    imports `transcribe_video` from `app.main`, so `app.main` must not require tkinter
    at import time.
    """
    # Simulate tkinter being unavailable (as in the python:slim Docker image).
    monkeypatch.setitem(sys.modules, "tkinter", None)
    monkeypatch.setitem(sys.modules, "_tkinter", None)
    sys.modules.pop("app.main", None)
    try:
        m = importlib.import_module("app.main")
        assert hasattr(m, "transcribe_video")
        assert m.tk is None  # guard took the headless branch
    finally:
        # Drop the headless-imported module so later tests re-import the real one.
        sys.modules.pop("app.main", None)
