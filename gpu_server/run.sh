#!/usr/bin/env bash
# Start the GPU Whisper server with gunicorn.
#
# Usage: gpu_server/run.sh            (from the repo root, venv at $WHISPER_VENV)
# Env:   WHISPER_VENV  path to the venv holding gpu_server/requirements.txt (default: ./.venv-gpu)
#        WHISPER_BIND  space-separated host:port list (default: 127.0.0.1:18921)
set -euo pipefail
cd "$(dirname "$0")/.."

VENV="${WHISPER_VENV:-$PWD/.venv-gpu}"
PY="$VENV/bin/python"

# ctranslate2 dlopens libcublas.so.12 / libcudnn.so.9 by name; the pip wheels ship
# them inside site-packages/nvidia/*/lib, which the loader does not search on its own.
NV_LIBS=$("$PY" - <<'PY'
import glob, os, site
dirs = []
for sp in site.getsitepackages():
    dirs += sorted(glob.glob(os.path.join(sp, "nvidia", "*", "lib")))
print(":".join(dirs))
PY
)
export LD_LIBRARY_PATH="${NV_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

BINDS=()
for b in ${WHISPER_BIND:-127.0.0.1:18921}; do BINDS+=(--bind "$b"); done

# One worker: the model lives in-process and one GPU runs one decode at a time.
# Threads let /health answer while a decode holds the lock.
exec "$VENV/bin/gunicorn" --workers 1 --threads 4 --timeout 1800 "${BINDS[@]}" gpu_server.wsgi:app
