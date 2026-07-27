FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/models
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 python3 python3-pip && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir --upgrade pip && python3 -m pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124 && python3 -m pip install --no-cache-dir -r requirements.txt
COPY app.py .
EXPOSE 80
CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
