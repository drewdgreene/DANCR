#!/usr/bin/env bash
# Build a one-folder DANCR app for Linux or macOS with PyInstaller.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
if ! "$PY" -c "import PyInstaller, PIL" 2>/dev/null; then
  echo "PyInstaller and Pillow are missing from $PY. Install the build tools first: pip install -e ".[build]"" >&2
  exit 1
fi
if [ ! -f packaging/icon.icns ] && [ "$(uname)" = "Darwin" ]; then
  "$PY" packaging/make_icons.py
fi
"$PY" -m PyInstaller --noconfirm --clean packaging/dancr.spec
if [ "$(uname)" = "Darwin" ]; then
  # ditto keeps the symlinks inside the bundle (zip -r follows them, which breaks the app's signature)
  (cd dist && rm -f DANCR-macos.zip && ditto -c -k --keepParent DANCR.app DANCR-macos.zip)
  echo "built dist/DANCR.app (dancr-cli inside Contents/MacOS) and dist/DANCR-macos.zip"
else
  (cd dist && tar -czf "DANCR-linux.tar.gz" DANCR)
  echo "built dist/DANCR/ (DANCR = window, dancr-cli = command line and MCP) and dist/DANCR-linux.tar.gz"
fi
