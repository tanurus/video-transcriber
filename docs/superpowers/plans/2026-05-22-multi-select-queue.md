# Multi-Select Video Queue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single-file picker in `TranscriptionApp` with a multi-select queue that shows per-video status and processes files sequentially.

**Architecture:** All changes are confined to `TranscriptionApp` in `app/main.py`. The queue state lives in a `ttk.Treeview` (rows = videos, columns = #/File/Status) plus a `dict[str, Path]` mapping tree item IDs to full paths. A single background thread walks pending rows sequentially, updating each row's status via `root.after`. `transcribe_video()` and the CLI path are untouched.

**Tech Stack:** Python 3.9+, tkinter/ttk (stdlib), existing `transcribe_video()` function

---

### Task 1: Replace `__init__` and `_build_ui` with queue UI

**Files:**
- Modify: `app/main.py` — `TranscriptionApp.__init__` and `TranscriptionApp._build_ui`

- [ ] **Step 1: Replace `__init__`**

Replace the entire `__init__` method with:

```python
def __init__(self, root: tk.Tk) -> None:
    self.root = root
    self.root.title("Video Transcription")
    self.root.geometry("780x580")

    self.status_var = tk.StringVar(value="Idle")
    self._queue_paths: dict[str, Path] = {}  # treeview iid -> full path
    self._log_queue: queue.Queue[str] = queue.Queue()
    self._worker: threading.Thread | None = None

    self._build_ui()
    self.root.after(150, self._drain_logs)
```

- [ ] **Step 2: Replace `_build_ui`**

Replace the entire `_build_ui` method with:

```python
def _build_ui(self) -> None:
    main = ttk.Frame(self.root, padding=12)
    main.pack(fill="both", expand=True)

    header = ttk.Label(main, text="Video to Transcript", font=("Segoe UI", 14, "bold"))
    header.pack(anchor="w")

    subtitle = ttk.Label(
        main,
        text="Add video files to the queue, then click Start. "
        "Transcripts are saved next to each video.",
        wraplength=720,
        justify="left",
    )
    subtitle.pack(anchor="w", pady=(4, 10))

    controls = ttk.Frame(main)
    controls.pack(fill="x", pady=(0, 8))
    ttk.Button(controls, text="Add Files...", command=self._add_files).pack(side="left")
    self.start_btn = ttk.Button(controls, text="Start Transcription", command=self._start_transcription)
    self.start_btn.pack(side="left", padx=(8, 0))
    ttk.Label(controls, textvariable=self.status_var, foreground="#2d6a4f").pack(side="left", padx=(12, 0))

    queue_label = ttk.Label(main, text="Queue", font=("Segoe UI", 10, "bold"))
    queue_label.pack(anchor="w")

    tree_frame = ttk.Frame(main)
    tree_frame.pack(fill="x", pady=(4, 8))

    self.queue_tree = ttk.Treeview(
        tree_frame,
        columns=("num", "file", "status"),
        show="headings",
        height=8,
        selectmode="none",
    )
    self.queue_tree.heading("num", text="#")
    self.queue_tree.heading("file", text="File")
    self.queue_tree.heading("status", text="Status")
    self.queue_tree.column("num", width=40, anchor="center", stretch=False)
    self.queue_tree.column("file", width=560)
    self.queue_tree.column("status", width=110, anchor="center", stretch=False)

    vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.queue_tree.yview)
    self.queue_tree.configure(yscrollcommand=vsb.set)
    self.queue_tree.pack(side="left", fill="x", expand=True)
    vsb.pack(side="left", fill="y")

    log_label = ttk.Label(main, text="Status log", font=("Segoe UI", 10, "bold"))
    log_label.pack(anchor="w")
    self.log_box = scrolledtext.ScrolledText(main, height=12, wrap="word", state="disabled")
    self.log_box.pack(fill="both", expand=True, pady=(4, 0))
```

- [ ] **Step 3: Add `_add_files` method (replaces `_choose_file`)**

Remove the old `_choose_file` method and add:

```python
def _add_files(self) -> None:
    paths = filedialog.askopenfilenames(
        title="Select video files",
        filetypes=[("Video files", "*.mp4 *.mkv *.mov *.avi *.webm"), ("All files", "*.*")],
    )
    for path in paths:
        idx = len(self.queue_tree.get_children()) + 1
        iid = self.queue_tree.insert("", "end", values=(idx, Path(path).name, "Pending"))
        self._queue_paths[iid] = Path(path)
```

- [ ] **Step 4: Launch the app and verify queue UI appears**

```powershell
$ffmpegPath = "C:\Users\andrew.t\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin"
$env:PATH = "$ffmpegPath;$env:PATH"
python -m app.main --gui
```

Expected: Window opens showing "Add Files..." and "Start Transcription" buttons above an empty 8-row Treeview with columns `#`, `File`, `Status`. Clicking "Add Files..." opens a native file picker that allows selecting multiple videos. Selected files appear as rows with status `Pending`.

- [ ] **Step 5: Commit**

```bash
git add app/main.py
git commit -m "feat: replace single-file picker with multi-select queue Treeview"
```

---

### Task 2: Replace processing logic with sequential queue runner

**Files:**
- Modify: `app/main.py` — replace `_start_transcription`, `_run_transcription`, `_notify_error`; add `_run_queue`, `_set_item_status`

- [ ] **Step 1: Replace `_start_transcription`**

Remove the old `_start_transcription` and replace with:

```python
def _start_transcription(self) -> None:
    if self._worker and self._worker.is_alive():
        return

    pending = [
        iid for iid in self.queue_tree.get_children()
        if self.queue_tree.set(iid, "status") == "Pending"
    ]
    if not pending:
        if not self.queue_tree.get_children():
            messagebox.showerror("Empty queue", "Add at least one video file first.")
        else:
            messagebox.showinfo("Nothing to do", "All queued files have already been processed.")
        return

    self._clear_log()
    self._set_status("Working...")
    self.start_btn.state(["disabled"])

    self._worker = threading.Thread(target=self._run_queue, args=(pending,), daemon=True)
    self._worker.start()
```

- [ ] **Step 2: Remove `_run_transcription` and `_notify_error`; add `_run_queue` and `_set_item_status`**

Delete the old `_run_transcription` and `_notify_error` methods. Add:

```python
def _run_queue(self, item_ids: list[str]) -> None:
    try:
        cfg = Config.load()
    except Exception as e:
        self._log(f"Config error: {e}")
        self._set_status("Error")
        self._enable_start()
        self.root.after(0, lambda: messagebox.showerror("Config error", str(e)))
        return

    for iid in item_ids:
        video_path = self._queue_paths[iid]
        self._set_item_status(iid, "In Progress")
        self._log(f"\n--- {video_path.name} ---")
        try:
            out_path = transcribe_video(video_path, cfg, logger=self._log)
            self._set_item_status(iid, "Done")
            self._log(f"Saved to {out_path}")
        except Exception as e:
            self._set_item_status(iid, "Failed")
            self._log(f"Error: {e}")

    self._set_status("Completed")
    self._enable_start()

def _set_item_status(self, iid: str, status: str) -> None:
    self.root.after(0, lambda i=iid, s=status: self.queue_tree.set(i, "status", s))
```

- [ ] **Step 3: Launch the app and verify full queue flow**

```powershell
$ffmpegPath = "C:\Users\andrew.t\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin"
$env:PATH = "$ffmpegPath;$env:PATH"
python -m app.main --gui
```

1. Click "Add Files..." and select 2+ video files → rows appear as `Pending`
2. Click "Start Transcription"
3. First row status changes to `In Progress`, log shows `--- filename ---` header
4. On completion, row changes to `Done` (or `Failed` on error)
5. Next row starts automatically
6. When all done: status bar shows `Completed`, Start button re-enables
7. Clicking "Add Files..." again adds new `Pending` rows; clicking Start processes only those new ones

- [ ] **Step 4: Commit**

```bash
git add app/main.py
git commit -m "feat: sequential queue processing with per-video status updates"
```
