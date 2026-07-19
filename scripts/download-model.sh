#!/usr/bin/env bash
#
# download-model.sh — fetch the Gemma 4 12B Core ML 128K Context Ladder bundle.
#
# Downloads the model bundle (~11 GB) into ./models/. Contents:
#   - int4 transformer chunks + int8 lm_head as .mlpackage (compiled to .mlmodelc once on the
#     first load and cached inside the bundle directory; not shipped pre-compiled)
#   - the two MTP drafters (drafter_ring.mlmodelc / drafter_ring32k.mlmodelc)
#   - the fp16 embedding sidecar (embed_fp16.bin)
#   - the tokenizer (tokenizer.json / tokenizer_config.json)
#
# Usage:
#   ./scripts/download-model.sh                 # → ./models/gemma-4-12b-it-coreml-128k
#   ./scripts/download-model.sh /path/to/dest   # custom destination
#
# The Hugging Face repo is private (Gemma Terms of Use). Authenticate first:
#   hf auth login
#
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

# --local-dir places real files (no cache symlinks) so the bundle is self-contained and portable.
hf download "$REPO" \
  --repo-type model \
  --local-dir "$DEST"

echo
echo "Done. Run the chat CLI with:"
echo "  swift run -c release corellm-chat --model \"$DEST\" --stats"
