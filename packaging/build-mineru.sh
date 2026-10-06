#!/usr/bin/env bash
# Bundle MinerU beside a DANCR build, so documents (PDF/Office/EPUB/HTML) work out of the box.
#
#   packaging/build-mineru.sh [DIST_DIR]      (default: dist/DANCR)
#   MINERU_TIER=basic|standard                (default: basic — table + OCR models, ~350 MB)
#
# Creates a RELOCATABLE Python 3.12 virtual environment with MinerU and the requested models at
# <DIST_DIR>/.dancr-mineru. The packaged DANCR finds it automatically (next to the DANCR/dancr-cli
# binaries); a source checkout finds .dancr-mineru in the repo root the same way. MinerU is Apache-2.0.
set -euo pipefail
cd "$(dirname "$0")/.."
DEST=${1:-dist/DANCR}
TIER=${MINERU_TIER:-basic}
ENV="$DEST/.dancr-mineru"

command -v uv >/dev/null 2>&1 || { echo "build-mineru needs uv (https://docs.astral.sh/uv)" >&2; exit 1; }
[ -d "$DEST" ] || { echo "build DANCR first: $DEST not found (run packaging/build.sh)" >&2; exit 1; }

echo "Creating a relocatable MinerU environment at $ENV (Python 3.12)…"
rm -rf "$ENV"
uv venv --relocatable --python 3.12 "$ENV"
uv pip install --python "$ENV/bin/python" "mineru>=4.0,<5"

echo "Downloading MinerU '$TIER' models into the bundle…"
MINERU_HOME="$PWD/$ENV" "$ENV/bin/mineru-models-download" --tier "$TIER" \
  || echo "Model download skipped or failed; MinerU will fetch models on first use (needs the network)."

"$ENV/bin/mineru-kit" --version
du -sh "$ENV"
echo "Bundled MinerU in $ENV — the app in $DEST now reads documents with no separate install."
