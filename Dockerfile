FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

COPY requirements.txt requirements-web.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-web.txt

COPY app/ ./app/
COPY web/ ./web/

# RNNoise model for the optional "RNNoise" noise reduction (BSD-licensed, ~300 KB).
ADD https://raw.githubusercontent.com/GregorR/rnnoise-models/master/somnolent-hogwash-2018-09-01/sh.rnnn /srv/models/sh.rnnn
RUN chmod 644 /srv/models/sh.rnnn
ENV RNNOISE_MODEL=/srv/models/sh.rnnn

ENV DATA_DIR=/data
EXPOSE 8000

CMD ["gunicorn", "--workers", "1", "--threads", "8", "--timeout", "1200", \
     "--bind", "0.0.0.0:8000", "web.wsgi:app"]
