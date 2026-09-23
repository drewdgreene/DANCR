#!/usr/bin/env bash
# Build a one-folder DANCR app for Linux or macOS with PyInstaller.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
if ! "$PY" -c "import PyInstaller, PIL" 2>/dev/null; then "$PY" -m pip install --quiet -e ".[build]"; fi
if [ ! -f packaging/icon.icns ] && [ "$(uname)" = "Darwin" ]; then
  "$PY" packaging/make_icons.py
fi
"$PY" -m PyInstaller --noconfirm --clean packaging/dancr.spec
if [ "$(uname)" = "Darwin" ]; then
  (cd dist && zip -qr "DANCR-macos.zip" DANCR.app)
  echo "built dist/DANCR.app (dancr-cli inside Contents/MacOS) and dist/DANCR-macos.zip"
else
  (cd dist && tar -czf "DANCR-linux.tar.gz" DANCR)
  echo "built dist/DANCR/ (DANCR = window, dancr-cli = command line and MCP) and dist/DANCR-linux.tar.gz"
fi
