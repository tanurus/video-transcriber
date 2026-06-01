# Audio/Video to Transcript CLI

A small Python CLI that extracts audio from a video file and sends it to the OpenAI Transcriptions API (Whisper-compatible) to get a transcript, saving a `.txt` next to the original video.

## Features
- Extract audio from most video formats via ffmpeg
- Automatically chunk large audio into smaller parts when needed
- Calls OpenAI Transcriptions API (default model: `gpt-4o-transcribe`)
- Saves transcript in the same folder as the input video

## Requirements
- Python 3.9+
- ffmpeg available on PATH (https://ffmpeg.org/download.html)
- An OpenAI API key with access to transcriptions

## Setup (Windows PowerShell)
```powershell
# 1) Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) Install dependencies
pip install -r requirements.txt

# 3) Set your OpenAI API key (temporarily for this session)
$env:OPENAI_API_KEY = "sk-..."

# Optional: Or create a .env file (copied from .env.example) to load automatically
Copy-Item .env.example .env
# then edit .env to add your key

# 4) Ensure ffmpeg is installed and on PATH
# Option A: If you use winget (Windows 11+)
winget install --id=Gyan.FFmpeg  --accept-package-agreements --accept-source-agreements
# Option B: Using Chocolatey (if installed)
choco install ffmpeg -y
# Option C: Manual — download a static build from https://www.gyan.dev/ffmpeg/builds/ and add its bin folder to PATH

# 5) Run the CLI against your file (default file is the one the request mentioned)
python -m app.main "with Vlass (updated priorities for Priceline and Arangrant) 2025-09-26 14-06-17.mkv"
```

If you omit the argument, it tries to use the default filename above in the current directory. The transcript is written as a `.txt` file with the same base name in the same folder.

## Configuration
You can customize behavior via environment variables (in your shell or `.env`):

- `OPENAI_API_KEY` (required): Your API key.
- `OPENAI_TRANSCRIBE_MODEL` (optional): Defaults to `gpt-4o-transcribe`. You may set to `whisper-1` if your account still supports it.
- `OPENAI_TIMEOUT` (optional): Request timeout in seconds, default 600.
- `CHUNK_TARGET_MB` (optional): Target max size per audio chunk before uploading, default 24 (MB).
- `AUDIO_BITRATE` (optional): MP3 bitrate used for export (e.g., `64k`, `96k`, `128k`). Default `96k`.

## Web app on a VPS (browser access over Tailscale)

You can run this as a browser app on a Linux VPS and reach it from any device on
your Tailscale tailnet — upload a video, watch live progress, and download the
transcript, with no local app running.

See [docs/deploy.md](docs/deploy.md) for full instructions. In short:

```bash
git clone https://github.com/tanurus/video-transcriber.git && cd video-transcriber
echo 'GROQ_API_KEY=gsk_...' > .env
chmod +x deploy.sh && ./deploy.sh
sudo tailscale serve --bg --https=443 http://127.0.0.1:8000
```

Then open `https://transcribe.<your-tailnet>.ts.net`.

## Notes
- ffmpeg is required for extracting audio and chunking. If not found, install it with winget/choco or manually and reopen your terminal so PATH updates.
- The app uses temporary directories for intermediate audio/chunks. They are cleaned up automatically.
- Be mindful of API usage costs.

## Troubleshooting
- PowerShell quoting: if a command errors with "string is missing the terminator", retype the quotes and ensure they are straight quotes `"` and not curly ones. You can also avoid quotes by renaming the file to not include spaces, or by using the short path syntax.
- ffmpeg not found: after installing ffmpeg, close and reopen your terminal window so PATH changes take effect.
