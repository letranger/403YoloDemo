#!/usr/bin/env bash

set -euo pipefail

PYTHON=${PYTHON:-python3}
ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)

SOURCE=${1:-0}
TEXT=${2:-"請發問"}

exec "$PYTHON" "$ROOT_DIR/src/yolo_pose_handraise.py" \
  --source "$SOURCE" \
  --device cpu \
  --speak-text "$TEXT" \
  --show


