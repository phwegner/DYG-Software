# ---------- BUILDER STAGE ----------
FROM python:3.11-slim as builder

WORKDIR /app

# Allow specifying an optional extra index URL for CUDA-enabled wheels (e.g. PyTorch)
ARG TORCH_INDEX_URL=""

# Build args
ARG TORCH_VERSION="2.0.1"
ARG TORCH_CUDA="cu117"
ARG TORCHVISION_VERSION="0.15.2"
ARG TORCHAUDIO_VERSION="2.0.1"

# Copy only requirements first (caching trick)
COPY requirements.txt ./

# Install build dependencies and Python packages into /install
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        build-essential \
    && pip install --upgrade pip \
    && pip install --prefix=/install \
         torch==${TORCH_VERSION}+${TORCH_CUDA} \
         torchvision==${TORCHVISION_VERSION}+${TORCH_CUDA} \
         torchaudio==${TORCHAUDIO_VERSION} \
         --extra-index-url https://download.pytorch.org/whl/${TORCH_CUDA} -c pytorch -c nvidia \
    && pip install --prefix=/install -r requirements.txt \
    && apt-get purge -y --auto-remove gcc build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---------- RUNTIME STAGE ----------
FROM python:3.11-slim

WORKDIR /app

# Install only minimal system deps needed at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        curl \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy project files
COPY . /app

# Make scripts executable (ignore if none)
RUN chmod +x /app/DYG-Software/src/*.sh || true

ENV PATH="/app:$PATH"

ENTRYPOINT ["/bin/bash"]
