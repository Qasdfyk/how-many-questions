#!/usr/bin/env bash
# Download Nutrition5k subset (metadata, RGB splits, overhead RGB). Resumable.
# Usage: scripts/01_download.sh [--dry-run] [--config configs/default.yaml]
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run python -m pbad.cli download "$@"
