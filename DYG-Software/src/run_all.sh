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
Usage: $0 [--input-dir <videos_dir>] [--project <project_path>]

This will run the prepare script, normalize video orientation, annotate videos with YOLO, then extract timeseries.
If no arguments are provided, defaults will be used:
  input-dir -> ../tests (relative to this script)
  project   -> ../annotated_videos (relative to this script)
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

# Default paths (relative to script directory)
DEFAULT_INPUT="${ROOT_DIR}/../input_videos"
DEFAULT_PROJECT="${ROOT_DIR}/../default_project"

if [[ -z "$INPUT_DIR" ]]; then
  INPUT_DIR="$DEFAULT_INPUT"
  log "No input-dir provided, using default: $INPUT_DIR"
fi
if [[ -z "$PROJECT" ]]; then
  PROJECT="$DEFAULT_PROJECT"
  log "No project provided, using default: $PROJECT"
fi

# Normalize PROJECT (remove trailing slash)
PROJECT=${PROJECT%/}

log "Starting full run: prepare -> normalize -> annotate -> extract"

# 0: prepare (download model)
if [[ -f "${ROOT_DIR}/0_prepare.sh" ]]; then
  log "Running prepare script"
  bash "${ROOT_DIR}/0_prepare.sh" || { log "Prepare script failed"; exit 1; }
else
  log "Prepare script not found: ${ROOT_DIR}/0_prepare.sh"
  exit 1
fi

# NEW: normalize video orientation
log "Normalizing video orientation in ${INPUT_DIR}"
mkdir -p "${INPUT_DIR}/normalized"
for f in "${INPUT_DIR}"/*.mov; do
  [ -e "$f" ] || continue  # skip if no .mov files
  rotation=$(ffprobe -v error -select_streams v:0 -show_entries stream_tags=rotate -of default=nw=1:nk=1 "$f" || echo 0)
  out="${INPUT_DIR}/normalized/$(basename "$f")"
  if [[ "$rotation" != "0" && -n "$rotation" ]]; then
    log "Fixing rotation ($rotation°) for $f"
    case "$rotation" in
      90)  ffmpeg -y -i "$f" -vf "transpose=1" -c:a copy "$out" ;;
      180) ffmpeg -y -i "$f" -vf "transpose=2,transpose=2" -c:a copy "$out" ;;
      270) ffmpeg -y -i "$f" -vf "transpose=2" -c:a copy "$out" ;;
      *)   cp "$f" "$out" ;;
    esac
  else
    log "No rotation needed for $f"
    cp "$f" "$out"
  fi
done

# 1: annotate (use normalized folder as input)
log "Running annotate"
python "${ROOT_DIR}/1_annotate.py" --video_folder "${INPUT_DIR}/normalized" --project "${PROJECT}" || { log "Annotate step failed"; exit 1; }

# 2: extract
log "Running extract"
python "${ROOT_DIR}/2_extract_ts.py" --path "${PROJECT}" || { log "Extract step failed"; exit 1; }

# 3: add frame numbers to annotated videos
log "Adding frame numbers to annotated videos"
if [[ -f "${ROOT_DIR}/add_frame_numbers.sh" ]]; then
  bash "${ROOT_DIR}/add_frame_numbers.sh" --input-dir "${PROJECT}" || { log "Adding frame numbers failed"; exit 1; }
else
  log "add_frame_numbers.sh not found: ${ROOT_DIR}/add_frame_numbers.sh"
  exit 1
fi

log "Full run complete"
