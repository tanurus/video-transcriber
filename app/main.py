from __future__ import annotations

import argparse
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
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
from .audio import extract_audio, chunk_audio_by_size
from .whisper_client import WhisperClient


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
            try:
                size_mb = audio_path.stat().st_size / (1024 * 1024)
            except Exception:
                size_mb = cfg.chunk_target_mb + 1  # force chunk if unknown

            audio_files = [audio_path]
            if size_mb > cfg.chunk_target_mb:
                log(f"Audio is {size_mb:.1f} MB; chunking into ~{cfg.chunk_target_mb} MB segments...")
                try:
                    audio_files = chunk_audio_by_size(audio_path, target_mb=cfg.chunk_target_mb, bitrate=cfg.audio_bitrate)
                    if audio_files and audio_files[0].parent != temp_dir:
                        chunk_dirs.append(audio_files[0].parent)
                except Exception as e:
                    log(f"Chunking failed (continuing with single file): {e}")
                    audio_files = [audio_path]
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
            language=getattr(cfg, "language", "en"),
        )

        log("Transcribing...")
        transcript_parts = []
        iterator = enumerate(audio_files, start=1)
        total_segments = len(audio_files)
        if use_tqdm:
            iterator = enumerate(tqdm(audio_files, desc="Segments"), start=1)
        for idx, f in iterator:
            log(f"Transcribing segment {idx}/{total_segments}: {f.name}")
            text = client.transcribe_file(f)
            transcript_parts.append(text.strip())

        transcript = "\n\n".join(transcript_parts).strip()

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

        self._clear_log()
        self._set_status("Working...")
        self.start_btn.state(["disabled"])

        self._worker = threading.Thread(target=self._run_queue, args=(pending,), daemon=True)
        self._worker.start()

    def _run_queue(self, item_ids: list[str]) -> None:
        try:
            cfg = Config.load()
        except Exception as e:
            self._log(f"Config error: {e}")
            self._set_status("Error")
            self._enable_start()
            self.root.after(0, lambda: messagebox.showerror("Config error", str(e)))
            return

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
