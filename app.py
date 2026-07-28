import asyncio
import os
import tempfile
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

from speaker_matching import speaker_for_segment

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


def _process(
    path: Path,
    language: str,
    num_speakers: int | None,
    batch_size: int,
    speaker_gap_seconds: float,
) -> list[dict]:
    whisper, diarizer = _models()
    started = time.perf_counter()
    diarization_kwargs = {"num_speakers": num_speakers} if num_speakers else {}
    diarization = diarizer(str(path), **diarization_kwargs)
    diarization_seconds = time.perf_counter() - started
    annotation = getattr(diarization, "speaker_diarization", diarization)
    labels = {label: f"SPK{i}" for i, label in enumerate(sorted(annotation.labels()))}
    turns = [
        (turn.start, turn.end, labels[label])
        for turn, _, label in annotation.itertracks(yield_label=True)
    ]
    started = time.perf_counter()
    whisper_segments, _ = whisper.transcribe(
        str(path),
        language=language,
        word_timestamps=True,
        vad_filter=True,
    )
    output = []
    for segment in whisper_segments:
        text = segment.text.strip()
        if text:
            output.append(
                {
                    "start": segment.start,
                    "end": segment.end,
                    "speaker": speaker_for_segment(
                        segment.start, segment.end, turns, speaker_gap_seconds
                    ),
                    "text": text,
                }
            )
    print(
        f"[metrics] diarization={diarization_seconds:.1f}s "
        f"whisper={time.perf_counter() - started:.1f}s batch={batch_size} "
        f"peak_vram={torch.cuda.max_memory_allocated() / 1024**3:.1f}GiB",
        flush=True,
    )
    return output


@app.get("/ping")
def ping() -> dict:
    return {"status": "ok"}


async def _run_job(
    job_id: str,
    path: Path,
    language: str,
    num_speakers: int | None,
    batch_size: int,
    speaker_gap_seconds: float,
) -> None:
    try:
        print(f"[job {job_id[:8]}] Processing started", flush=True)
        async with GPU_LOCK:
            torch.cuda.reset_peak_memory_stats()
            JOBS[job_id] = {
                "status": "completed",
                "segments": await asyncio.to_thread(
                    _process,
                    path,
                    language,
                    num_speakers,
                    batch_size,
                    speaker_gap_seconds,
                ),
            }
        print(f"[job {job_id[:8]}] Processing completed", flush=True)
    except Exception as exc:
        JOBS[job_id] = {"status": "failed", "error": str(exc)}
        print(f"[job {job_id[:8]}] Failed: {exc}", flush=True)
    finally:
        path.unlink(missing_ok=True)


@app.post("/jobs/{job_id}")
async def create_job(
    job_id: str,
    file: UploadFile = File(...),
    language: str = Form("it"),
    num_speakers: int | None = Form(None),
    batch_size: int = Form(8),
    speaker_gap_seconds: float = Form(1.0),
) -> dict:
    if num_speakers is not None and num_speakers < 1:
        raise HTTPException(status_code=422, detail="num_speakers must be positive")
    if batch_size != 1:
        raise HTTPException(status_code=422, detail="batch_size must be 1")
    if speaker_gap_seconds < 0:
        raise HTTPException(
            status_code=422, detail="speaker_gap_seconds must be non-negative"
        )
    if job_id in JOBS:
        return {"status": JOBS[job_id]["status"]}
    suffix = Path(file.filename or "audio.ogg").suffix or ".ogg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
        temp_path = Path(temp.name)
        while chunk := await file.read(1024 * 1024):
            temp.write(chunk)
    JOBS[job_id] = {"status": "running"}
    JOBS[job_id]["task"] = asyncio.create_task(
        _run_job(
            job_id,
            temp_path,
            language,
            num_speakers,
            batch_size,
            speaker_gap_seconds,
        )
    )
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
