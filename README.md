# DYG-Software
Software Repository for DYG

## Requirements
- Docker (optional, for containerized runs)
- Python 3.11

## CUDA / GPU setup
If you plan to run inference on GPU, install a CUDA-enabled PyTorch wheel that matches your host NVIDIA driver before installing the rest of the Python requirements. Examples:

- Check driver and CUDA compatibility on the host with:

```bash
nvidia-smi
```

- Install a matching PyTorch wheel (examples):

```bash
# CUDA 11.7
pip install --index-url https://download.pytorch.org/whl/cu117 torch torchvision torchaudio --upgrade

# CUDA 12.x (replace cu121 with the accurate tag for your driver)
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio --upgrade
```

After that, install the rest of the requirements:

```bash
pip install -r requirements.txt
```

See https://pytorch.org/get-started/locally/ for exact tags for your environment.

## Build (optional)
To build a Docker image:

```bash
docker build -t dyg-software .
```

# Build with PyTorch CUDA wheel index (optional)
The `Dockerfile` accepts a build argument `TORCH_INDEX_URL` which is prepended to the `pip install` call inside the builder stage. Use this to point pip at the PyTorch extra-index for CUDA-enabled wheels that match your host drivers. Example:

```bash
# CUDA 11.7 (example)
docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu117 --build-arg TORCH_CUDA=cu117 --build-arg TORCH_VERSION=2.0.1 --build-arg TORCHVISION_VERSION=0.15.2 --build-arg TORCHAUDIO_VERSION=2.0.1 -t dyg-software .

# CUDA 12.x (example; replace with correct tag for your driver)
docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 --build-arg TORCH_CUDA=cu121 --build-arg TORCH_VERSION=2.7.1 --build-arg TORCHVISION_VERSION=0.22.1 --build-arg TORCHAUDIO_VERSION=2.7.1 -t dyg-software .
```

The `Dockerfile` in this repository installs `torch`, `torchvision` and `torchaudio` during the builder stage using the optional `TORCH_INDEX_URL` build argument.



- The Docker build should use the `--build-arg TORCH_INDEX_URL=...` option to point pip at the PyTorch extra-index that provides CUDA-enabled wheels. Example:

```bash
# CUDA 11.7 (example)
docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu117 -t dyg-software .
```

- Verify which torch wheel / CUDA tag to use by checking your host NVIDIA driver and CUDA compatibility. You can find supported/old releases and exact wheel tags on the PyTorch site — check the "Previous Versions" page if you need a specific older wheel:

https://pytorch.org/get-started/previous-versions/

- In short: choose a CUDA-enabled base image whose runtime matches the wheel tag you will install; pass the matching `TORCH_INDEX_URL` when building the image; then run the container with `--gpus all` (and having NVIDIA Container Toolkit installed on the host).

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

### GPU support
If you want to run inference on GPU inside Docker, install the NVIDIA Container Toolkit on the host and run the container with --gpus. Example (after building the image):

```bash
# Install nvidia-container-toolkit on the host (install steps depend on distro)
# Run container with GPU access and bind mounts to defaults used by the runner
docker run --gpus all -it --rm \
  -v <PATH_WITH_VIDEOS>:/app/DYG-Software/input_videos \
  -v <OUTPUT_PATH>:/app/DYG-Software/default_project \
  -v <MODEL_STORAGE>:/app/DYG-Software/model_files \
  dyg-software /app/DYG-Software/src/run_all.sh
```

Inside the container the `yolo` CLI (ultralytics) can then use available NVIDIA GPUs if CUDA libraries are present in the image or mounted from the host.

### Singularity / Apptainer
You can also produce a Singularity / Apptainer image from the Docker image and run it. Example workflow:

```bash
# 1) Build the Docker image locally (if not already built)
docker build -t dyg-software .

# 2) Build a Singularity image from the local Docker image (requires Singularity/Apptainer with docker-daemon support)
singularity build dyg-software.sif docker-daemon://dyg-software:latest

# 3) Run the Singularity image (bind host dirs into the container). Use --nv for NVIDIA GPU support.
singularity exec --nv \
  -B <PATH_WITH_VIDEOS>:/app/DYG-Software/input_videos \
  -B <OUTPUT_PATH>:/app/DYG-Software/default_project \
  -B <MODEL_STORAGE>:/app/DYG-Software/model_files \
  dyg-software.sif /app/DYG-Software/src/run_all.sh
```

If your Singularity build environment cannot access the local Docker daemon, push the image to a registry and use `docker://<repo>:<tag>` in the `singularity build` command.

## Notes
- The pipeline depends on the `yolo` CLI from ultralytics; ensure it is available in your environment or inside the Docker image.
- The Docker image installs Python requirements listed in `requirements.txt` but may need additional system packages (e.g. ffmpeg) depending on your data and usage.

## Demo video
A demo video used for testing in this repository was taken from YouTube: https://www.youtube.com/watch?v=FFki8FtaByw

Please ensure you have the appropriate rights or permission to use this video in your environment. If you redistribute results derived from third-party videos, cite the original source and comply with the video's license and YouTube terms. Accessed 20 August 2025.
