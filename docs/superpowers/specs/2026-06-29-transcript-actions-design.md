# Transcript actions in the desktop GUI

**Date:** 2026-06-29
**Scope:** `app/main.py` — `TranscriptionApp` (desktop Tkinter GUI) only.

## Problem

After a queued video is transcribed, the GUI only writes a `.txt` next to the
video and logs the path. To read or use the transcript the user has to leave the
app, find the folder, and open the file by hand. There is no in-app way to copy
the text or open it.

## Goal

For a completed queue item, let the user, from the app:

1. Copy the whole transcript text to the clipboard.
2. Open the transcript `.txt` in the OS default editor.
3. Reveal the transcript in the OS file manager (selected/highlighted).

## Non-goals

- No changes to the CLI, the web app, or `transcribe_video`.
- No editing of the transcript inside the app.
- No preview pane.

## Design

### Interaction
- Queue `Treeview` `selectmode` changes from `"none"` to `"browse"` so one row
  can be selected.
- A new button row sits directly under the queue: **Copy transcript**,
  **Open file**, **Open folder**.
- Buttons are enabled only when the selected row's status is `Done` **and** its
  saved `.txt` still exists on disk. A `<<TreeviewSelect>>` handler refreshes the
  enabled/disabled state; otherwise the buttons are disabled.

### State
- New `self._output_paths: dict[str, Path]` mapping a tree row iid to its saved
  transcript path. Populated in `_run_queue` at the point status becomes `Done`.
  Written from the worker thread, read on the UI thread; a plain dict assignment
  is sufficient (single writer, atomic in CPython, read only after status flips).

### Actions
- **Copy transcript** — read the `.txt` as UTF-8, push onto the Tk clipboard
  (`clipboard_clear` / `clipboard_append` / `update`), then flash a status line
  such as `Copied lecture.txt`.
- **Open file** — `open_path(txt)`.
- **Open folder** — `reveal_path(txt)`.

### Cross-platform helpers (module-level, no Tk dependency → unit-testable)
- `open_path(path)`:
  - Windows: `os.startfile(path)`
  - macOS: `subprocess.run(["open", path])`
  - Linux/other: `subprocess.run(["xdg-open", path])`
- `reveal_path(path)`:
  - Windows: `subprocess.run(["explorer", "/select,", path])`
  - macOS: `subprocess.run(["open", "-R", path])`
  - Linux/other: open the parent directory via `open_path(path.parent)`

Platform is selected via `sys.platform`. Helpers raise on failure; callers wrap
them and surface errors through `messagebox.showerror` plus a status line so the
UI thread never crashes.

## Testing

- Unit tests for `open_path` and `reveal_path` that monkeypatch `os.startfile`
  and `subprocess.run`, asserting the correct command per platform (`win32`,
  `darwin`, `linux`). These run headless — no Tk, no display.
- GUI wiring (button creation, selection handler) stays thin and is not
  unit-tested; there is no display in CI.

## Risks

- Clipboard contents on Tk are owned by the app window; copying then closing the
  app immediately can drop the clipboard on some platforms. Acceptable for this
  desktop use; `update()` flushes it to the OS while the app is open.
