FROM python:3.11-slim

WORKDIR /app

# Copy project files
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    libgl1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --upgrade pip \
    && if [ -f requirements.txt ]; then pip install -r requirements.txt; fi \
    && chmod +x /app/DYG-Software/src/*.sh || true

ENV PATH="/app:$PATH"

ENTRYPOINT ["/bin/bash"]
