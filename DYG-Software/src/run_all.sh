#!/usr/bin/env bash
set -euo pipefail

# Simple runner that executes scripts in order: 0_prepare.sh, 1_annotate.py, 2_extract_ts.py
# Usage: ./run_all.sh --input-dir <videos_dir> --project <project_name>

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

usage() {
  cat <<EOF
Usage: $0 --input-dir <videos_dir> --project <project_name>

This will run the prepare script, annotate videos with YOLO, then extract timeseries.
EOF
}

INPUT_DIR=""
PROJECT=""

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --input-dir)
      INPUT_DIR="$2"; shift 2;;
    --project)
      PROJECT="$2"; shift 2;;
    -h|--help)
      usage; exit 0;;
    *)
      log "Unknown arg: $1"; usage; exit 1;;
  esac
done

if [[ -z "$INPUT_DIR" || -z "$PROJECT" ]]; then
  usage; exit 1
fi

log "Starting full run: prepare -> annotate -> extract"

# 0: prepare (download model)
if [[ -x "${ROOT_DIR}/0_prepare.sh" ]]; then
  log "Running prepare script"
  bash "${ROOT_DIR}/0_prepare.sh"
else
  log "Prepare script not found or not executable: ${ROOT_DIR}/0_prepare.sh"
fi

# 1: annotate
log "Running annotate"
python "${ROOT_DIR}/1_annotate.py" --video_folder "${INPUT_DIR}" --project "${PROJECT}"

# 2: extract
log "Running extract"
python "${ROOT_DIR}/2_extract_ts.py" --path "${PROJECT}"

# 3: add frame numbers to annotated videos
log "Adding frame numbers to annotated videos"
PROJECT=${PROJECT%/}
bash "${ROOT_DIR}/add_frame_numbers.sh" --input-dir "${PROJECT}"

log "Full run complete"
