#!/usr/bin/env bash
# Build the DANCR Windows installer from a frozen app folder.
# NSIS cross-compiles, so this runs on Linux or macOS as well as Windows.
#
#   packaging/build-installer.sh                      # uses dist/DANCR
#   APPDIR=dist/DANCR OUT=dist/Setup.exe packaging/build-installer.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python3}
VERSION=$("$PY" -c "import re,pathlib;print(re.search(r'__version__ = \"([^\"]+)\"', pathlib.Path('dancr/__init__.py').read_text()).group(1))")
APPDIR=${APPDIR:-dist/DANCR}
OUT=${OUT:-dist/DANCR-Setup-$VERSION.exe}
ICON=${ICON:-packaging/icon.ico}

[ -d "$APPDIR" ] || { echo "no frozen app at $APPDIR (run PyInstaller first)"; exit 1; }
[ -f "$ICON" ] || "$PY" packaging/make_icons.py
ICON=$(realpath "$ICON")
APPDIR=$(realpath "$APPDIR")
OUT=$(realpath -m "$OUT")

makensis -DAPPDIR="$APPDIR" -DVERSION="$VERSION" -DOUTFILE="$OUT" -DICON="$ICON" packaging/windows/installer.nsi
echo "built $OUT"
