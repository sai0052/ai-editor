FROM python:3.11-slim

# ffmpeg isn't a Python package — it's a system program the app calls out to,
# so it has to be installed here, not via pip.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# HF Spaces expects the app to listen on port 7860 by default.
ENV PORT=7860
EXPOSE 7860

CMD ["python", "main.py"]
