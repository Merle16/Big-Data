FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# System deps: Java for PySpark + basic tools
# default-jre-headless creates /usr/lib/jvm/default-java symlink on both amd64 and arm64
RUN apt-get update && apt-get install -y --no-install-recommends \
    default-jre-headless \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# JAVA_HOME is required by PySpark.
# /usr/lib/jvm/default-java is a distro-managed symlink that works on both
# amd64 and arm64 without hardcoding the architecture in the path.
ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# Copy project files
COPY requirements.txt ./requirements.txt
COPY config ./config
COPY src ./src
COPY data ./data
# members/ is volume-mounted at runtime (see README/docker_setup_instructions.md)
# so each member's work is picked up without rebuilding the image.
# To bake members/ into the image instead, uncomment:
# COPY members ./members

# Install Python deps
RUN python -m pip install --upgrade pip --trusted-host pypi.org --trusted-host files.pythonhosted.org && \
    python -m pip install -r requirements.txt --trusted-host pypi.org --trusted-host files.pythonhosted.org

ENV PYTHONPATH=/app/src

# Default: drop into bash. Override with specific script for CI:
#   docker run big-data-imdb python members/ilesh/pipeline.py
CMD ["bash"]

