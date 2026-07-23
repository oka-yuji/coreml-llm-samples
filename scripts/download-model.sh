#!/usr/bin/env bash
set -euo pipefail

REPO="${CORELLM_HF_REPO:-okayuji/gemma-4-12b-it-coreml-128k}"
DEFAULT_DEST="$(cd "$(dirname "$0")/.." && pwd)/models/gemma-4-12b-it-coreml-128k"
DEST="${1:-$DEFAULT_DEST}"

echo "Model repo:   $REPO"
echo "Destination:  $DEST"
echo "Size:         ~11 GB — make sure you have the disk space and a good connection."
echo

if ! command -v hf >/dev/null 2>&1; then
  echo "error: the 'hf' CLI was not found." >&2
  echo "Install it with:  pip install -U huggingface_hub" >&2
  exit 1
fi

mkdir -p "$DEST"

hf download "$REPO" \
  --repo-type model \
  --local-dir "$DEST"

echo
echo "Done. Run the chat CLI with:"
echo "  swift run -c release corellm-chat --model \"$DEST\" --stats"
