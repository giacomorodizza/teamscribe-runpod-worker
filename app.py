import asyncio
import os
import tempfile
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

MODEL = os.environ.get("WHISPER_MODEL", "large-v3")
HF_TOKEN = os.environ.get("HF_TOKEN")
GPU_LOCK = asyncio.Lock()
JOBS: dict[str, dict] = {}


@lru_cache(maxsize=1)
def _models() -> tuple[WhisperModel, Pipeline]:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is required")
    whisper = WhisperModel(MODEL, device="cuda", compute_type="float16")
    diarizer = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", **{"use_auth_token": HF_TOKEN}
    )
    if diarizer is None:
        raise RuntimeError("Unable to load pyannote pipeline")
    return whisper, diarizer.to(torch.device("cuda"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[startup] Loading Whisper and PyAnnote models...", flush=True)
    await asyncio.to_thread(_models)
    print("[startup] Models ready", flush=True)
    yield


app = FastAPI(lifespan=lifespan)


def _speaker(start: float, end: float, turns: list[tuple[float, float, str]]) -> str:
    overlap, label = 0.0, "UNK"
    for turn_start, turn_end, speaker in turns:
        value = max(0.0, min(end, turn_end) - max(start, turn_start))
        if value > overlap:
            overlap, label = value, speaker
    return label


def _process(path: Path, language: str) -> list[dict]:
    whisper, diarizer = _models()
    diarization = diarizer(str(path))
    annotation = getattr(diarization, "speaker_diarization", diarization)
    labels = {label: f"SPK{i}" for i, label in enumerate(sorted(annotation.labels()))}
    turns = [
        (turn.start, turn.end, labels[label])
        for turn, _, label in annotation.itertracks(yield_label=True)
    ]
    whisper_segments, _ = whisper.transcribe(
        str(path), language=language, word_timestamps=True, vad_filter=True
    )
    output = []
    for segment in whisper_segments:
        text = segment.text.strip()
        if text:
            output.append(
                {
                    "start": segment.start,
                    "end": segment.end,
                    "speaker": _speaker(segment.start, segment.end, turns),
                    "text": text,
                }
            )
    return output


@app.get("/ping")
def ping() -> dict:
    return {"status": "ok"}


async def _run_job(job_id: str, path: Path, language: str) -> None:
    try:
        print(f"[job {job_id[:8]}] Processing started", flush=True)
        async with GPU_LOCK:
            JOBS[job_id] = {
                "status": "completed",
                "segments": await asyncio.to_thread(_process, path, language),
            }
        print(f"[job {job_id[:8]}] Processing completed", flush=True)
    except Exception as exc:
        JOBS[job_id] = {"status": "failed", "error": str(exc)}
        print(f"[job {job_id[:8]}] Failed: {exc}", flush=True)
    finally:
        path.unlink(missing_ok=True)


@app.post("/jobs/{job_id}")
async def create_job(
    job_id: str, file: UploadFile = File(...), language: str = Form("it")
) -> dict:
    if job_id in JOBS:
        return {"status": JOBS[job_id]["status"]}
    suffix = Path(file.filename or "audio.ogg").suffix or ".ogg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
        temp_path = Path(temp.name)
        while chunk := await file.read(1024 * 1024):
            temp.write(chunk)
    JOBS[job_id] = {"status": "running"}
    JOBS[job_id]["task"] = asyncio.create_task(_run_job(job_id, temp_path, language))
    print(f"[job {job_id[:8]}] Upload accepted", flush=True)
    return {"status": "running"}


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] == "completed":
        return {"status": "completed", "segments": job["segments"]}
    if job["status"] == "failed":
        return {"status": "failed", "error": job["error"]}
    return {"status": "running"}
