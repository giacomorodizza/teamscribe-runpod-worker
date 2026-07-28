# TeamScribe RunPod worker

RunPod Load Balancing worker: Faster-Whisper `large-v3` + Pyannote speaker diarization.

Set RunPod environment variables `HF_TOKEN` and `WHISPER_MODEL=large-v3`. Configure port `80` and health path `/ping`.

`POST /jobs/{job_id}` accepts optional `num_speakers`, `speaker_gap_seconds` and `batch_size=1`. Worker logs diarization time, Whisper time and peak VRAM for every job.
