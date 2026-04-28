FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONPATH=/app

# 1. Install dependencies (cached unless pyproject.toml changes)
#    Stub package lets setuptools resolve deps without full source.
COPY pyproject.toml .
RUN mkdir -p seed_storage && touch seed_storage/__init__.py \
    && pip install --no-cache-dir . \
    && rm -rf seed_storage

# 2. Download Whisper model (cached — never changes)
RUN python -c "import whisper; whisper.load_model('base')"

# 3. Copy source code last (changes frequently — seconds to rebuild)
COPY seed_storage/ seed_storage/
COPY scripts/ scripts/
COPY ingestion/ ingestion/
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

EXPOSE 8080

CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
