#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f .env ]; then
  echo "ERROR: .env not found in $(pwd)." >&2
  echo "Create it with your key, e.g.:  echo 'GROQ_API_KEY=gsk_...' > .env" >&2
  exit 1
fi

echo "==> Pulling latest code..."
git pull --ff-only

echo "==> Building and starting the container..."
docker compose up -d --build

echo "==> Container status:"
docker compose ps

cat <<'EOF'

==> Done.

First-time only — expose the app on your tailnet (run once on the host):
  sudo tailscale serve --bg --https=443 http://127.0.0.1:8000

Optional — give it the short domain by renaming this machine to "transcribe":
  sudo tailscale set --hostname=transcribe

Then open:  https://transcribe.<your-tailnet>.ts.net

To view logs:        docker compose logs -f
To redeploy later:   ./deploy.sh
EOF
