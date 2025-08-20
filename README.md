# DYG-Software
Software Repository for DYG

## Requirements
- Docker (optional, for containerized runs)
- Python 3.11

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Build (optional)
To build a Docker image:

```bash
docker build -t dyg-software .
```

Run a container (interactive):

```bash
docker run -it --rm -v <PATH_WITH_VIDEOS>:/app/DYG-Software/input_videos -v <OUTPUT_PATH>:/app/DYG-Software/default_project -v <MODEL_STORAGE>:/app/DYG-Software/model_files dyg-software /bin/bash
```

Note: the runner `src/run_all.sh` uses the following default paths (relative to the repository root):
- input videos: `DYG-Software/input_videos`
- annotated output (project): `DYG-Software/default_project`
- model files: `DYG-Software/model_files`

If you mount your host input/output/model folders to these container paths you can run the pipeline inside the container without passing command-line arguments to `run_all.sh`.

## Run
A convenience script `DYG-Software/src/run_all.sh` runs the three-step pipeline in order:

0. `0_prepare.sh` - downloads model files into `model_files/`
1. `1_annotate.py` - runs YOLO pose prediction on the provided videos
2. `2_extract_ts.py` - extracts time-series CSVs and plots

Example test run (from repository root):

```bash
bash DYG-Software/src/run_all.sh --input-dir DYG-Software/tests/ --project test_project
```

If you started a Docker container and mounted your directories to the default container paths shown above you can run the script inside the container without arguments:

```bash
# inside container where /app is the repo root
bash DYG-Software/src/run_all.sh
```

Or run it directly with docker by invoking the script as the container command:

```bash
docker run -it --rm \
  -v <PATH_WITH_VIDEOS>:/app/DYG-Software/input_videos \
  -v <OUTPUT_PATH>:/app/DYG-Software/default_project \
  -v <MODEL_STORAGE>:/app/DYG-Software/model_files \
  dyg-software /app/DYG-Software/src/run_all.sh
```

Adjust paths as needed for your environment.


## Notes
- The pipeline depends on the `yolo` CLI from ultralytics; ensure it is available in your environment or inside the Docker image.
- The Docker image installs Python requirements listed in `requirements.txt` but may need additional system packages (e.g. ffmpeg) depending on your data and usage.

## Demo video
A demo video used for testing in this repository was taken from YouTube: https://www.youtube.com/watch?v=FFki8FtaByw

Please ensure you have the appropriate rights or permission to use this video in your environment. If you redistribute results derived from third-party videos, cite the original source and comply with the video's license and YouTube terms. Accessed 20 August 2025.
