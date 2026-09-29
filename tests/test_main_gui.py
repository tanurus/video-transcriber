"""Tests for the platform open/reveal helpers used by the desktop GUI.

These are deliberately Tk-free so they run headless (no display in CI).
"""
import sys
from pathlib import Path

import pytest

import app.main as m
from app.config import Config


@pytest.mark.parametrize("language,candidates", [
    ("de", None), (None, ["de", "fr"]), ("de", ["ro", "ru", "en"]),
])
def test_language_dropdown_preserves_configured_languages(language, candidates):
    cfg = Config(openai_api_key="test", language=language, candidate_languages=candidates)
    options = m._language_options(cfg)
    selected = m._default_language_label(cfg)
    assert options[selected] == (language, candidates)
    assert options["Auto-detect"] == (None, None)


def _expect(p: str) -> str:
    # Path str form is OS-dependent; compare against the same normalization
    # the helper uses so these tests pass on any runner.
    return str(Path(p))


def test_open_path_windows_uses_startfile(monkeypatch):
    calls = []
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(m.os, "startfile", lambda p: calls.append(p), raising=False)
    m.open_path("C:/notes/lecture.txt")
    assert calls == [_expect("C:/notes/lecture.txt")]


def test_open_path_macos_uses_open(monkeypatch):
    calls = []
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(m.subprocess, "run", lambda *a, **k: calls.append(a[0]))
    m.open_path("/notes/lecture.txt")
    assert calls == [["open", _expect("/notes/lecture.txt")]]


def test_open_path_linux_uses_xdg_open(monkeypatch):
    calls = []
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(m.subprocess, "run", lambda *a, **k: calls.append(a[0]))
    m.open_path("/notes/lecture.txt")
    assert calls == [["xdg-open", _expect("/notes/lecture.txt")]]


def test_reveal_path_windows_selects_in_explorer(monkeypatch):
    calls = []
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(m.subprocess, "run", lambda *a, **k: calls.append(a[0]))
    m.reveal_path("C:/notes/lecture.txt")
    assert calls == [["explorer", "/select,", _expect("C:/notes/lecture.txt")]]


def test_reveal_path_macos_uses_open_R(monkeypatch):
    calls = []
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(m.subprocess, "run", lambda *a, **k: calls.append(a[0]))
    m.reveal_path("/notes/lecture.txt")
    assert calls == [["open", "-R", _expect("/notes/lecture.txt")]]


def test_reveal_path_linux_opens_parent_dir(monkeypatch):
    calls = []
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(m.subprocess, "run", lambda *a, **k: calls.append(a[0]))
    m.reveal_path("/notes/lecture.txt")
    # Falls back to opening the containing folder.
    assert calls == [["xdg-open", _expect("/notes")]]
