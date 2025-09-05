#!/usr/bin/env bash
set -euo pipefail

# add_frame_numbers.sh
# Recursively find .mp4 files under a directory and create a copy with frame numbers
# overlaid using ffmpeg drawtext. Output files get suffix "-with_frames.mp4".

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
    --input-dir)
      INPUT_DIR="$2"; shift 2;;
    --font)
      FONT="$2"; shift 2;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

if [[ -z "$INPUT_DIR" ]]; then
  usage; exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg not found on PATH. Please install ffmpeg and try again." >&2
  exit 2
fi

# Choose a sensible default font if not provided
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


# Normalize INPUT_DIR to absolute path
INPUT_DIR="$(realpath "$INPUT_DIR")"


echo "Processing video files under: $INPUT_DIR"

# Find video files and process them
find "$INPUT_DIR" -type f \( -iname '*.mp4' -o -iname '*.mov' -o -iname '*.avi' -o -iname '*.mkv' \) -print0 |
while IFS= read -r -d '' video; do
  echo "Found: $video"

  # Skip if already processed
  if [[ "$video" == *"-with_frames.mp4" ]]; then
    echo "Skipping already processed file: $video"
    continue
  fi

  dir=$(dirname "$video")
  base=$(basename "$video")
  name_noext="${base%.*}"
  out="$dir/${name_noext}-with_frames.mp4"

  # Skip if output already exists
  if [[ -f "$out" ]]; then
    echo "Output already exists, skipping: $out"
    continue
  fi

  # Build drawtext filter (quote $FONT in case of spaces)
  if [[ -n "$FONT" ]]; then
    drawtxt="drawtext=fontfile='$FONT': text='%{n}': x=(w-tw)/2: y=h-(2*lh): \
fontcolor=white: fontsize=32: box=1: boxcolor=0x00000099"
  else
    drawtxt="drawtext=text='%{n}': x=(w-tw)/2: y=h-(2*lh): \
fontcolor=white: fontsize=32: box=1: boxcolor=0x00000099"
  fi

  echo "Creating: $out"
  # Use -- to separate options from filenames
  ffmpeg -hide_banner -loglevel error -y -i "$video" -vf "$drawtxt" -c:a copy -- "$out" \
    && echo "Saved $out" || echo "Failed for $video"

done

echo "All done."
