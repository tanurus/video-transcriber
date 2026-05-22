# Multi-Select Video Queue — Design Spec

**Date:** 2026-05-22

## Context

The current GUI supports transcribing a single video at a time. Users want to select up to ~10 videos at once and have them processed sequentially without manual re-triggering between files.

## Goal

Replace the single-file picker with a multi-select queue UI. Videos are processed one at a time in the order they were added. Each video shows its own status. A shared log shows combined output.

## UI Layout

```
[ Add Files...  ]  [ Start Transcription ]   Status: Idle

 #  | File                              | Status
----|-----------------------------------|----------
 1  | Airline Revenue Review.mp4        | Pending
 2  | Q1 Planning Meeting.mkv           | In Progress
 3  | Sales Call Recording.mp4          | Done
 4  | Product Demo.mp4                  | Failed

--- Status log ---
[ scrollable log output ]
```

## Components

### File picker
- Replace `ttk.Entry` + single `Browse` button with an "Add Files…" button
- Uses `filedialog.askopenfilenames` (native multi-select, same filetypes filter as before)
- Each selected file is appended as a new row with status `Pending`
- Re-clicking "Add Files…" while the queue is idle appends more files

### Queue list (`ttk.Treeview`)
- Columns: `#` (index), `File` (basename), `Status`
- Status values: `Pending` / `In Progress` / `Done` / `Failed`
- Read-only (no removal needed)

### Processing
- Single background thread iterates queue rows sequentially
- For each row: set status → `In Progress`, call `transcribe_video()`, set → `Done` or `Failed`
- Log prints `--- <filename> ---` header before each video's output
- Start button disabled for the duration; re-enabled when all rows finish

### Log
- Existing `ScrolledText` widget retained unchanged
- Per-video headers (`--- filename ---`) visually separate output sections

## Files Changed

- `app/main.py` — all changes are in `TranscriptionApp`; `transcribe_video()` and CLI path are untouched

## Out of Scope

- Removing items from the queue
- Parallel processing
- Pausing / cancelling mid-queue
