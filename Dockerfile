# Build stage
FROM python:3.13-bullseye as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.13-slim-bullseye

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libcairo2 \
    libfreetype6 \
    libgdk-pixbuf2.0-0 \
    libjpeg62-turbo \
    liblcms2-2 \
    libopenjp2-7 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    poppler-utils \
    postgresql-client \
    python3-pil \
    ffmpeg \
    libopus0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /DiscordBot
COPY . .

CMD ["python3", "main.py"]