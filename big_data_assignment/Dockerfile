FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# System deps for Spark (Java) and basic tools
# openjdk-17-jre-headless is no longer available on Debian trixie,
# so we install the replacement openjdk-21-jre-headless instead.
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-21-jre-headless \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files for the ML + Spark/DuckDB pipeline
COPY requirements.txt ./requirements.txt
COPY config ./config
COPY src ./src
COPY data ./data

# Install Python deps
RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements.txt

ENV PYTHONPATH=/app/src

CMD ["bash"]

