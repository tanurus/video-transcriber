# Deploying the Web Transcriber to a VPS

The app runs as a Docker container, reachable only over your Tailscale tailnet.

## Prerequisites (on the VPS)
- Docker + the `docker compose` plugin
- Tailscale installed and logged in (`tailscale status` shows the machine)

## One-time setup
1. Clone the repo and enter it:
   ```bash
   git clone https://github.com/tanurus/video-transcriber.git
   cd video-transcriber
   ```
2. Create `.env` with your Groq key (never committed):
   ```bash
   echo 'GROQ_API_KEY=gsk_your_key_here' > .env
   ```
3. Make the deploy script executable:
   ```bash
   chmod +x deploy.sh
   ```
4. Deploy:
   ```bash
   ./deploy.sh
   ```
5. Expose it on the tailnet (run once):
   ```bash
   sudo tailscale serve --bg --https=443 http://127.0.0.1:8000
   ```
6. (Optional) Rename the machine for a short domain:
   ```bash
   sudo tailscale set --hostname=transcribe
   ```
7. Open `https://transcribe.<your-tailnet>.ts.net` from any device on your tailnet.

## Redeploying after changes
```bash
./deploy.sh
```

## Where data lives
- `./data/app.db` — job history (SQLite)
- `./data/uploads/<job-id>/` — uploaded videos (auto-deleted after `RETAIN_VIDEO_DAYS`, default 30)
- `./data/transcripts/<job-id>.txt` — transcripts (kept forever)

## Configuration (set in `docker-compose.yml` or `.env`)
- `RETAIN_VIDEO_DAYS` (default 30)
- `MAX_CONTENT_MB` (default 2048)
- `ALLOWED_EXT` (default `mp4,mkv,mov,avi,webm,m4a,mp3,wav`)

## Notes
- The container binds only to `127.0.0.1:8000`; nothing is exposed to the public internet.
- Do **not** enable `tailscale funnel` unless you also add authentication — the app has none by design (the private tailnet is the access boundary).
