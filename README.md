# DYG-Software
Software Repository for DYG

## Requirements
- Docker (optional, for containerized runs)
- Python 3.11

## CUDA / GPU setup
If you plan to run inference on GPU, ensure your host NVIDIA driver and CUDA compatibility are correct. Check with:

```bash
nvidia-smi
```

This repository uses a `constraints.txt` to pin exact package versions (including `torch`/`torchvision`/`torchaudio` when needed). Rather than installing `torch` separately, install all Python dependencies with pip while supplying the constraints file so versions are resolved against your pinned constraints.

- If you want to install packages natively on the host, use:

```bash
# Install all requirements using the pinned versions in constraints.txt
pip install --extra-index-url https://download.pytorch.org/whl/<CUDA_VERSION> -r requirements.txt -c constraints.txt
```

- If you need CUDA-enabled PyTorch wheels that are not available via PyPI, adjust `constraints.txt` to reference the correct wheel filenames or use the appropriate extra-index/find-links. See the PyTorch docs for exact wheel names and compatibility:

https://pytorch.org/get-started/locally/
https://pytorch.org/get-started/previous-versions/

Note: when pinning versions in `constraints.txt`, ensure the `numpy` version is compatible with your chosen `torch` wheel to avoid binary incompatibilities.

## Build (optional)
To build a Docker image (defaults, CPU-only):

```bash
docker build -t dyg-software .
```

# CUDA / Torch packaging note (NEW)
This repository's `Dockerfile` now installs Python packages using a `constraints.txt` file to pin exact package versions (including `torch`, `torchvision`, `torchaudio` when needed). The Docker build accepts a single build argument `TORCH_CUDA` which selects the CUDA variant tag used inside the image (for example `cu117` for CUDA 11.7).

- Edit `constraints.txt` in the repo root to specify the exact `torch`/`torchvision`/`torchaudio` versions you want. Example constraint lines (illustrative only — use the exact versions that match your environment):

```
# Example lines for CUDA 11.7 (replace versions with those you need)
torch==2.0.1+cu117
torchvision==0.15.2+cu117
torchaudio==2.0.2+cu117
numpy>=1.24,<2
```

Note: depending on the wheel format you may also need to include the appropriate find-links or use the PyTorch index when installing. If you need to install CUDA-enabled wheels from PyTorch's index, follow the instructions below before building or include the correct `-f`/index in your constraints/install commands.

- Build the Docker image and pass the CUDA tag via `TORCH_CUDA` (example):

```bash
docker build --build-arg TORCH_CUDA=cu117 -t dyg-software .
```

- Make sure the versions pinned in `constraints.txt` are compatible with the CUDA tag you pass and with the base image you choose for GPU runtime. Check the host driver with `nvidia-smi` and consult the PyTorch compatibility pages:

https://pytorch.org/get-started/locally/
https://pytorch.org/get-started/previous-versions/

If you do not need GPU support, omit the `--build-arg TORCH_CUDA` and use CPU-only constraints in `constraints.txt`.

# Build with PyTorch CUDA wheel index (optional)
The `Dockerfile` accepts a build argument `TORCH_INDEX_URL` which is prepended to the `pip install` call inside the builder stage. Use this to point pip at the PyTorch extra-index for CUDA-enabled wheels that match your host drivers. Example:

```bash
# CUDA 11.7 (example)
docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu117 --build-arg TORCH_CUDA=cu117 --build-arg TORCH_VERSION=2.0.1 --build-arg TORCHVISION_VERSION=0.15.2 --build-arg TORCHAUDIO_VERSION=2.0.2 -t dyg-software .

# CUDA 12.x (example; replace with correct tag for your driver)
docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 --build-arg TORCH_CUDA=cu121 --build-arg TORCH_VERSION=2.7.1 --build-arg TORCHVISION_VERSION=0.22.1 --build-arg TORCHAUDIO_VERSION=2.7.1 -t dyg-software .
```

The `Dockerfile` in this repository installs `torch`, `torchvision` and `torchaudio` during the builder stage using the optional `TORCH_INDEX_URL`, `TORCH_CUDA`, `TORCH_VERSION`, `TORCHVISION_VERSION`, and `TORCHAUDIO_VERSION` build arguments.

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
