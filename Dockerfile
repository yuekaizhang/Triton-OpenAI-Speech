FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt  tts_server.py ref_audios/ .
RUN pip install --no-cache-dir -r requirements.txt