#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
PYTHON='/mnt/HDD/AliWorks/HCI Tagging Database/HCI/bin/python3'
if [[ ! -x "$PYTHON" ]]; then
  printf 'BLOCKED: exact HCI venv required: %s\n' "$PYTHON" >&2
  exit 2
fi
cd "$ROOT"
"$PYTHON" scripts/release_preflight.py --tier source
"$PYTHON" -m unittest discover -s tests -p test_release_preflight.py -v
"$PYTHON" scripts/release_preflight.py --tier replay
