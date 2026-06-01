FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

COPY requirements.txt requirements-web.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-web.txt

COPY app/ ./app/
COPY web/ ./web/

ENV DATA_DIR=/data
EXPOSE 8000

CMD ["gunicorn", "--workers", "1", "--threads", "8", "--timeout", "1200", \
     "--bind", "0.0.0.0:8000", "web.wsgi:app"]
