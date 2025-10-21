#!/usr/bin/env bash
set -euo pipefail

# add_frame_numbers.sh
# For each subfolder in INPUT_DIR, find the .avi file named like the folder
# and create an mp4 copy with frame numbers overlaid.

usage() {
  cat <<EOF
Usage: $0 --input-dir <dir> [--font <fontfile>]

Example:
  $0 --input-dir /path/to/annotated_videos

If --font is not provided the script will try common system fonts.
EOF
}

INPUT_DIR=""
FONT=""

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --input-dir) INPUT_DIR="$2"; shift 2;;
    --font) FONT="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

if [[ -z "$INPUT_DIR" ]]; then
  usage; exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg not found on PATH. Please install ffmpeg and try again." >&2
  exit 2
fi

# Default font detection
if [[ -z "$FONT" ]]; then
  for f in "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" \
           "/usr/share/fonts/truetype/freefont/FreeSans.ttf" \
           "/Library/Fonts/Arial.ttf" \
           "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"; do
    if [[ -f "$f" ]]; then
      FONT="$f"
      break
    fi
  done
fi

if [[ -z "$FONT" ]]; then
  echo "Warning: no font file found; drawtext may fail or use default font." >&2
fi

INPUT_DIR="$(realpath "$INPUT_DIR")"
echo "Processing folders under: $INPUT_DIR"

# Loop over each folder in INPUT_DIR
for folder in $INPUT_DIR/*/; do
  [[ -d "$folder" ]] || continue  # skip if not a directory

  folder_name=$(basename "$folder")
  video="$folder/${folder_name}.avi"

  if [[ ! -f "$video" ]]; then
    echo "No .avi file named $folder_name.avi in folder $folder, skipping."
    continue
  fi

  out="$folder/${folder_name}-with_frames.mp4"
  if [[ -f "$out" ]]; then
    echo "Output already exists, skipping: $out"
    continue
  fi

  # Build drawtext filter
  if [[ -n "$FONT" ]]; then
    drawtxt="drawtext=fontfile='$FONT': text='%{n}': x=(w-tw)/2: y=h-(2*lh): \
fontcolor=white: fontsize=32: box=1: boxcolor=0x00000099"
  else
    drawtxt="drawtext=text='%{n}': x=(w-tw)/2: y=h-(2*lh): \
fontcolor=white: fontsize=32: box=1: boxcolor=0x00000099"
  fi

  echo "Creating: $out"
  ffmpeg -hide_banner -loglevel error -y -i "$video" -vf "$drawtxt" -c:a copy -- "$out" \
    && echo "Saved $out" || echo "Failed for $video"

done

echo "All done."
