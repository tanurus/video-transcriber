from __future__ import annotations

import argparse
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Optional, Sequence
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, scrolledtext, ttk
except ImportError:  # headless environments (e.g. the Docker server image) have no Tk
    tk = None  # type: ignore[assignment]
    filedialog = messagebox = scrolledtext = ttk = None  # type: ignore[assignment]

from tqdm import tqdm

from .config import Config
from .audio import extract_audio, chunk_audio_by_silence, chunk_audio_by_size
from .whisper_client import SegmentFilter, WhisperClient


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract audio from video and transcribe via OpenAI")
    parser.add_argument(
        "video",
        nargs="?",
        default=None,
        help="Path to the video file (omit only with --gui)",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch the graphical interface for selecting a video and viewing progress",
    )
    return parser.parse_args(argv)


def transcribe_video(
    video_path: Path,
    cfg: Config,
    logger: Callable[[str], None] | None = None,
    use_tqdm: bool = False,
    out_path: Path | None = None,
) -> Path:
    """Transcribe a video, reporting progress via the provided logger."""
    log = logger or (lambda msg: print(msg))
    video_path = Path(video_path).expanduser()

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found at {video_path}")

    temp_dir = Path(tempfile.mkdtemp(prefix="avtx_work_"))
    log(f"Working directory: {temp_dir}")

    chunk_dirs: list[Path] = []
    try:
        log("Extracting audio with ffmpeg...")
        use_video_directly = False
        try:
            audio_path = extract_audio(video_path, out_dir=temp_dir, bitrate=cfg.audio_bitrate)
        except Exception as e:
            log(f"Audio extraction failed: {e}")
            log("Will attempt to upload the original video directly to the transcription API.")
            use_video_directly = True

        audio_files = []
        if not use_video_directly:
            audio_files = [audio_path]
            log("Splitting audio into silence-aware segments (better per-segment language detection)...")
            try:
                audio_files = chunk_audio_by_silence(
                    audio_path,
                    target_sec=cfg.chunk_target_sec,
                    max_sec=cfg.chunk_max_sec,
                    noise_db=cfg.silence_noise_db,
                    min_silence_sec=cfg.silence_min_sec,
                )
            except Exception as e:
                # Fall back to the old size-based split so we never regress to no
                # chunking at all on a large file.
                log(f"Silence-based chunking failed ({e}); falling back to size-based chunking...")
                try:
                    size_mb = audio_path.stat().st_size / (1024 * 1024)
                except Exception:
                    size_mb = cfg.chunk_target_mb + 1  # force chunk if unknown
                if size_mb > cfg.chunk_target_mb:
                    try:
                        audio_files = chunk_audio_by_size(
                            audio_path, target_mb=cfg.chunk_target_mb, bitrate=cfg.audio_bitrate
                        )
                    except Exception as e2:
                        log(f"Chunking failed (continuing with single file): {e2}")
                        audio_files = [audio_path]
                else:
                    audio_files = [audio_path]

            # The chunkers write splits into their own temp dir; track it for
            # cleanup. Only when a split actually happened (len > 1) is there a
            # separate dir — a single unsplit file still lives in temp_dir.
            if len(audio_files) > 1 and audio_files[0].parent != temp_dir:
                chunk_dirs.append(audio_files[0].parent)
            if len(audio_files) > 1:
                log(f"Split into {len(audio_files)} segments.")
            else:
                log("Audio short enough to transcribe as a single segment.")
        else:
            try:
                size_mb = video_path.stat().st_size / (1024 * 1024)
            except Exception:
                size_mb = 0
            if size_mb > 0:
                log(f"Video size: {size_mb:.1f} MB")
            log("Note: Without ffmpeg, large files cannot be chunked; upload may fail if too large.")
            audio_files = [video_path]

        client = WhisperClient(
            api_key=cfg.openai_api_key,
            model=cfg.model,
            timeout=cfg.timeout,
            base_url=cfg.base_url,
            language=cfg.language,
            candidate_languages=cfg.candidate_languages,
            segment_filter=SegmentFilter(
                no_speech_threshold=cfg.no_speech_threshold,
                logprob_threshold=cfg.logprob_threshold,
                compression_ratio_threshold=cfg.compression_ratio_threshold,
            ),
        )

        total_segments = len(audio_files)
        workers = max(1, min(cfg.max_concurrency, total_segments))
        log(f"Transcribing {total_segments} segment(s), up to {workers} in parallel...")
        # Results are keyed by index so the transcript stays in audio order even
        # though segments finish out of order.
        results: list[str] = [""] * total_segments
        completed = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            future_to_idx = {
                ex.submit(client.transcribe_file, f, log): i for i, f in enumerate(audio_files)
            }
            completion = as_completed(future_to_idx)
            if use_tqdm:
                completion = tqdm(completion, total=total_segments, desc="Segments")
            for fut in completion:
                idx = future_to_idx[fut]
                results[idx] = fut.result().strip()  # propagate a segment error to the caller
                completed += 1
                log(f"Segment {completed}/{total_segments} done.")

        transcript = "\n\n".join(part for part in results if part).strip()

        out_txt = Path(out_path) if out_path else video_path.with_suffix(".txt")
        out_txt.parent.mkdir(parents=True, exist_ok=True)
        # Write via temp file + atomic rename so a crash mid-write can never
        # leave a truncated transcript at the expected output path.
        tmp_txt = out_txt.with_name(out_txt.name + ".tmp")
        tmp_txt.write_text(transcript, encoding="utf-8")
        os.replace(tmp_txt, out_txt)
        log(f"Transcript saved to {out_txt}")
        return out_txt
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        for d in chunk_dirs:
            shutil.rmtree(d, ignore_errors=True)


def open_path(path: Path | str) -> None:
    """Open a file or folder in the OS default handler."""
    path = Path(path)
    if sys.platform == "win32":
        os.startfile(str(path))  # type: ignore[attr-defined]  # Windows-only
    elif sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=False)
    else:
        subprocess.run(["xdg-open", str(path)], check=False)


def reveal_path(path: Path | str) -> None:
    """Reveal a file in the OS file manager, selecting it when supported."""
    path = Path(path)
    if sys.platform == "win32":
        # explorer returns a non-zero exit code even on success, so don't check.
        subprocess.run(["explorer", "/select,", str(path)], check=False)
    elif sys.platform == "darwin":
        subprocess.run(["open", "-R", str(path)], check=False)
    else:
        # No portable "select file" on Linux; open the containing folder.
        open_path(path.parent)


def run_cli(args: argparse.Namespace) -> int:
    if not args.video:
        print("Error: provide a path to a video file, or use --gui.", file=sys.stderr)
        return 2

    video_path = Path(args.video)
    if not video_path.is_absolute():
        video_path = Path.cwd() / video_path

    if not video_path.exists():
        print(f"Error: Video not found at {video_path}", file=sys.stderr)
        return 2

    try:
        cfg = Config.load()
    except Exception as e:
        print(f"Config error: {e}", file=sys.stderr)
        return 2

    try:
        transcribe_video(video_path, cfg, logger=print, use_tqdm=True)
    except Exception as e:
        print(f"Transcription failed: {e}", file=sys.stderr)
        return 4

    return 0


# Language dropdown label -> (forced language, candidate languages for the race).
LANG_OPTIONS: "dict[str, tuple[Optional[str], Optional[list[str]]]]" = {
    "Auto-detect": (None, None),
    "Auto (RO/RU/EN)": (None, ["ro", "ru", "en"]),
    "Romanian": ("ro", None),
    "Russian": ("ru", None),
    "English": ("en", None),
}


def _default_language_label(cfg: Config) -> str:
    """Pick the dropdown label that reflects the loaded config."""
    for label, (lang, cands) in LANG_OPTIONS.items():
        if (lang, cands) == (cfg.language, cfg.candidate_languages):
            return label
    return "Configured (.env)"


def _language_options(cfg: Config) -> dict:
    """Keep custom configured languages selectable without altering presets."""
    options = dict(LANG_OPTIONS)
    options[_default_language_label(cfg)] = (cfg.language, cfg.candidate_languages)
    return options


def _parse_db(value: str) -> int:
    """Extract the integer dB from a string like '-30dB' (default -30)."""
    m = re.search(r"-?\d+", value or "")
    return int(m.group()) if m else -30


class TranscriptionApp:
    """Lightweight Tkinter UI for selecting a video and viewing transcription logs."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Video Transcription")
        self.root.geometry("780x580")

        self.status_var = tk.StringVar(value="Idle")
        self._queue_paths: dict[str, Path] = {}  # treeview iid -> source video path
        self._output_paths: dict[str, Path] = {}  # treeview iid -> saved transcript .txt
        self._next_idx: int = 1
        self._log_queue: queue.Queue[str] = queue.Queue()
        self._worker: threading.Thread | None = None

        # Experiment controls. Defaults reflect .env / config defaults; loading
        # without a key still yields the dataclass defaults for the UI.
        try:
            defaults = Config.load()
        except Exception:
            defaults = Config(openai_api_key="")
        self.language_options = _language_options(defaults)
        self.language_var = tk.StringVar(value=_default_language_label(defaults))
        self.chunk_target_var = tk.IntVar(value=defaults.chunk_target_sec)
        self.chunk_max_var = tk.IntVar(value=defaults.chunk_max_sec)
        self.silence_db_var = tk.IntVar(value=_parse_db(defaults.silence_noise_db))
        self.max_parallel_var = tk.IntVar(value=defaults.max_concurrency)

        self._build_ui()
        self.root.after(150, self._drain_logs)

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
        ttk.Button(controls, text="Remove selected", command=self._remove_selected).pack(side="left", padx=(8, 0))
        self.start_btn = ttk.Button(controls, text="Start Transcription", command=self._start_transcription)
        self.start_btn.pack(side="left", padx=(8, 0))

        ttk.Label(controls, text="Language:").pack(side="left", padx=(12, 4))
        self.language_combo = ttk.Combobox(
            controls,
            textvariable=self.language_var,
            state="readonly",
            width=16,
            values=list(self.language_options.keys()),
        )
        self.language_combo.pack(side="left")
        ttk.Button(controls, text="Settings...", command=self._open_settings).pack(side="left", padx=(8, 0))

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
            selectmode="browse",
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
        self.queue_tree.bind("<<TreeviewSelect>>", lambda _e: self._refresh_actions())

        actions = ttk.Frame(main)
        actions.pack(fill="x", pady=(0, 8))
        self.copy_btn = ttk.Button(actions, text="Copy transcript", command=self._copy_transcript)
        self.copy_btn.pack(side="left")
        self.open_btn = ttk.Button(actions, text="Open file", command=self._open_transcript)
        self.open_btn.pack(side="left", padx=(8, 0))
        self.folder_btn = ttk.Button(actions, text="Open folder", command=self._reveal_transcript)
        self.folder_btn.pack(side="left", padx=(8, 0))
        ttk.Label(
            actions,
            text="Select a completed file to copy or open its transcript.",
            foreground="#6c757d",
        ).pack(side="left", padx=(12, 0))
        self._refresh_actions()

        log_label = ttk.Label(main, text="Status log", font=("Segoe UI", 10, "bold"))
        log_label.pack(anchor="w")
        self.log_box = scrolledtext.ScrolledText(main, height=12, wrap="word", state="disabled")
        self.log_box.pack(fill="both", expand=True, pady=(4, 0))

    def _open_settings(self) -> None:
        """Modal dialog to tweak chunking/parallelism; applies on the next run."""
        dlg = tk.Toplevel(self.root)
        dlg.title("Settings")
        dlg.transient(self.root)
        dlg.resizable(False, False)
        frm = ttk.Frame(dlg, padding=12)
        frm.pack(fill="both", expand=True)

        # Snapshot so Cancel (or closing the window) restores prior values.
        snapshot = (
            self.chunk_target_var.get(),
            self.chunk_max_var.get(),
            self.silence_db_var.get(),
            self.max_parallel_var.get(),
        )

        rows = [
            ("Chunk target (seconds)", self.chunk_target_var),
            ("Chunk max (seconds)", self.chunk_max_var),
            ("Silence sensitivity (dB, negative)", self.silence_db_var),
            ("Max parallel segments", self.max_parallel_var),
        ]
        for i, (label, var) in enumerate(rows):
            ttk.Label(frm, text=label).grid(row=i, column=0, sticky="w", pady=4, padx=(0, 8))
            ttk.Entry(frm, textvariable=var, width=8).grid(row=i, column=1, sticky="e", pady=4)

        ttk.Label(
            frm,
            text="Shorter chunks = more frequent language detection.\nBelow ~20s can hurt accuracy.",
            foreground="#6c757d",
        ).grid(row=len(rows), column=0, columnspan=2, sticky="w", pady=(6, 10))

        def on_ok() -> None:
            try:
                t = self.chunk_target_var.get()
                mx = self.chunk_max_var.get()
                db = self.silence_db_var.get()
                par = self.max_parallel_var.get()
            except Exception:
                messagebox.showerror("Invalid settings", "All fields must be whole numbers.", parent=dlg)
                return
            if t <= 0 or mx <= 0 or par < 1:
                messagebox.showerror("Invalid settings", "Seconds and parallelism must be positive.", parent=dlg)
                return
            if mx < t:
                messagebox.showerror("Invalid settings", "Chunk max must be >= chunk target.", parent=dlg)
                return
            if db >= 0:
                messagebox.showerror("Invalid settings", "Silence dB should be negative (e.g. -30).", parent=dlg)
                return
            dlg.destroy()

        def on_cancel() -> None:
            self.chunk_target_var.set(snapshot[0])
            self.chunk_max_var.set(snapshot[1])
            self.silence_db_var.set(snapshot[2])
            self.max_parallel_var.set(snapshot[3])
            dlg.destroy()

        btns = ttk.Frame(frm)
        btns.grid(row=len(rows) + 1, column=0, columnspan=2, sticky="e")
        ttk.Button(btns, text="Cancel", command=on_cancel).pack(side="right", padx=(8, 0))
        ttk.Button(btns, text="OK", command=on_ok).pack(side="right")

        dlg.protocol("WM_DELETE_WINDOW", on_cancel)
        dlg.grab_set()

    def _remove_selected(self) -> None:
        """Drop the selected queue row(s). Blocked while a run is in progress."""
        if self._worker and self._worker.is_alive():
            messagebox.showinfo("Busy", "Can't remove files while transcription is running.")
            return
        selection = self.queue_tree.selection()
        if not selection:
            return
        for iid in selection:
            self.queue_tree.delete(iid)
            self._queue_paths.pop(iid, None)
            self._output_paths.pop(iid, None)
        self._refresh_actions()

    def _add_files(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Select video files",
            filetypes=[
                ("Video files", ("*.mp4", "*.mkv", "*.mov", "*.avi", "*.webm")),
                ("All files", "*.*"),
            ],
        )
        existing = set(self._queue_paths.values())
        for path in paths:
            p = Path(path)
            if p in existing:
                continue
            idx = self._next_idx
            self._next_idx += 1
            iid = self.queue_tree.insert("", "end", values=(idx, p.name, "Pending"))
            self._queue_paths[iid] = p

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

        # Read the Tk variables here on the main thread; the worker thread must
        # not touch Tk. Fall back gracefully if a field holds invalid text.
        try:
            lang, candidates = self.language_options[self.language_var.get()]
            overrides = {
                "language": lang,
                "candidate_languages": candidates,
                "chunk_target_sec": self.chunk_target_var.get(),
                "chunk_max_sec": self.chunk_max_var.get(),
                "silence_noise_db": f"{self.silence_db_var.get()}dB",
                "max_concurrency": self.max_parallel_var.get(),
                "label": self.language_var.get(),
            }
        except Exception:
            messagebox.showerror("Invalid settings", "Please fix the values under Settings... first.")
            return

        self._clear_log()
        self._set_status("Working...")
        self.start_btn.state(["disabled"])

        self._worker = threading.Thread(
            target=self._run_queue, args=(pending, overrides), daemon=True
        )
        self._worker.start()

    def _run_queue(self, item_ids: list[str], overrides: dict | None = None) -> None:
        overrides = overrides or {}
        try:
            cfg = Config.load()
        except Exception as e:
            self._log(f"Config error: {e}")
            self._set_status("Error")
            self._enable_start()
            self.root.after(0, lambda: messagebox.showerror("Config error", str(e)))
            return

        # Apply the UI overrides for this run (per-run, not persisted).
        for key in (
            "language",
            "candidate_languages",
            "chunk_target_sec",
            "chunk_max_sec",
            "silence_noise_db",
            "max_concurrency",
        ):
            if key in overrides:
                setattr(cfg, key, overrides[key])
        self._log(
            f"Settings: language={overrides.get('label', '?')}, "
            f"chunks {cfg.chunk_target_sec}-{cfg.chunk_max_sec}s, "
            f"silence {cfg.silence_noise_db}, parallel {cfg.max_concurrency}"
        )

        try:
            for iid in item_ids:
                video_path = self._queue_paths[iid]
                self._set_item_status(iid, "In Progress")
                self._log(f"\n--- {video_path.name} ---")
                try:
                    out_path = transcribe_video(video_path, cfg, logger=self._log)
                    self._output_paths[iid] = out_path
                    self._set_item_status(iid, "Done")
                    self._log(f"Saved to {out_path}")
                except Exception as e:
                    self._set_item_status(iid, "Failed")
                    self._log(f"Error: {e}")
        finally:
            self._set_status("Completed")
            self._enable_start()
            self.root.after(0, self._refresh_actions)

    def _set_item_status(self, iid: str, status: str) -> None:
        self.root.after(0, lambda i=iid, s=status: self.queue_tree.set(i, "status", s))

    def _enable_start(self) -> None:
        self.root.after(0, lambda: self.start_btn.state(["!disabled"]))

    def _selected_transcript(self) -> Path | None:
        """Return the saved .txt for the selected row if it is Done and on disk."""
        selection = self.queue_tree.selection()
        if not selection:
            return None
        iid = selection[0]
        if self.queue_tree.set(iid, "status") != "Done":
            return None
        out_path = self._output_paths.get(iid)
        if out_path is None or not out_path.exists():
            return None
        return out_path

    def _refresh_actions(self) -> None:
        """Enable the transcript buttons only when a usable .txt is selected."""
        state = "!disabled" if self._selected_transcript() else "disabled"
        for btn in (self.copy_btn, self.open_btn, self.folder_btn):
            btn.state([state])

    def _copy_transcript(self) -> None:
        out_path = self._selected_transcript()
        if out_path is None:
            return
        try:
            text = out_path.read_text(encoding="utf-8")
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.root.update()  # flush to the OS clipboard while the app is open
        except Exception as e:
            messagebox.showerror("Copy failed", str(e))
            return
        self._set_status(f"Copied {out_path.name}")

    def _open_transcript(self) -> None:
        out_path = self._selected_transcript()
        if out_path is None:
            return
        try:
            open_path(out_path)
        except Exception as e:
            messagebox.showerror("Open failed", str(e))

    def _reveal_transcript(self) -> None:
        out_path = self._selected_transcript()
        if out_path is None:
            return
        try:
            reveal_path(out_path)
        except Exception as e:
            messagebox.showerror("Open folder failed", str(e))

    def _set_status(self, text: str) -> None:
        self.root.after(0, lambda: self.status_var.set(text))

    def _clear_log(self) -> None:
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", tk.END)
        self.log_box.configure(state="disabled")

    def _log(self, message: str) -> None:
        # Push log messages onto a queue to keep UI thread responsive
        self._log_queue.put(message)

    def _drain_logs(self) -> None:
        try:
            try:
                while True:
                    msg = self._log_queue.get_nowait()
                    self.log_box.configure(state="normal")
                    self.log_box.insert(tk.END, msg + "\n")
                    self.log_box.see(tk.END)
                    self.log_box.configure(state="disabled")
            except queue.Empty:
                pass
            except Exception:
                return
            try:
                self.root.after(150, self._drain_logs)
            except Exception:
                pass
        except Exception:
            pass


def run_gui() -> int:
    if tk is None:
        print(
            "The graphical interface requires tkinter, which is not available in this "
            "environment. Use the CLI or the web app instead.",
            file=sys.stderr,
        )
        return 2
    root = tk.Tk()
    TranscriptionApp(root)
    root.mainloop()
    return 0


def main() -> int:
    args = parse_args()
    if args.gui:
        return run_gui()
    return run_cli(args)


if __name__ == "__main__":
    raise SystemExit(main())
