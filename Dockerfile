# ---------- BUILDER STAGE ----------
FROM python:3.11-slim as builder

WORKDIR /app

# Copy only requirements first (caching trick)
COPY requirements.txt ./

# Install build dependencies and Python packages into /install
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        build-essential \
    && pip install --upgrade pip \
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
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy project files
COPY . /app

# Make scripts executable (ignore if none)
RUN chmod +x /app/DYG-Software/src/*.sh || true

ENV PATH="/app:$PATH"

ENTRYPOINT ["/bin/bash"]
