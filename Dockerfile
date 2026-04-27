FROM node:18-slim AS viz-builder
WORKDIR /build
COPY viz/package.json viz/package-lock.json* ./
RUN npm ci
COPY viz/ .
RUN npm run build

FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml .
COPY seed_storage/ seed_storage/
COPY scripts/ scripts/
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY --from=viz-builder /build/dist viz/dist/

RUN pip install --no-cache-dir .
RUN python -c "import whisper; whisper.load_model('base')"

EXPOSE 8080 7860

CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
