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

Run a container:

```bash
docker run -it --rm -v <PATH_WITH_VIDEOS>:/app/videos -v <OUTPUT_PATH>:/app/output -v <MODEL_STORAGE>:/app/DYG_Software/model_files dyg-software /bin/bash
```

## Run
A convenience script `DYG-Software/src/run_all.sh` runs the three-step pipeline in order:

0. `0_prepare.sh` - downloads model files into `model_files/`
1. `1_annotate.py` - runs YOLO pose prediction on the provided videos
2. `2_extract_ts.py` - extracts time-series CSVs and plots

Example test run (from repository root):

```bash
bash DYG-Software/src/run_all.sh --input-dir DYG-Software/tests/ --project test_project
```
Or in Docker: 
```bash
docker run -it --rm -v <PATH_WITH_VIDEOS>:/app/videos -v <OUTPUT_PATH>:/app/output -v <MODEL_STORAGE>:/app/DYG_Software/model_files dyg-software /app/src/run_all.sh --input-dir /app/videos/ --project test_project
``` 

Adjust paths as needed for your environment.

## Notes
- The pipeline depends on the `yolo` CLI from ultralytics; ensure it is available in your environment or inside the Docker image.
- The Docker image installs Python requirements listed in `requirements.txt` but may need additional system packages (e.g. ffmpeg) depending on your data and usage.
