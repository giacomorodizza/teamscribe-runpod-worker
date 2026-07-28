FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/models LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
RUN apt-get update && apt-get install -y --no-install-recommends cuda-compat-12-6 ffmpeg libsndfile1 python3 python3-pip && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir --upgrade pip && python3 -m pip install --no-cache-dir torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126 && python3 -m pip install --no-cache-dir -r requirements.txt
COPY app.py speaker_matching.py .
EXPOSE 80
CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
