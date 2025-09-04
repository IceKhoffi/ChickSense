import numpy as np
import random
import json
import os
import subprocess
from io import BytesIO
from typing import Optional, Dict
from datetime import datetime
import soundfile as sf
from .vocalization_prediction import vocalization_prediction

def _resolve_youtube_audio_url(url: str) -> str:
    if "youtube.com" in url or "youtu.be" in url:
        try:
            meta = subprocess.run(
                ["yt-dlp", "--no-cache-dir", "-J", url],
                capture_output=True, text=True, check=True, timeout=30
            )
            info = json.loads(meta.stdout)
            is_live = bool(info.get("is_live"))
            fmt = "best[protocol^=m3u8]/best" if is_live else "bestaudio/best"
            out = subprocess.run(
                ["yt-dlp", "--no-cache-dir", "--get-url", "-f", fmt, url],
                capture_output=True, text=True, check=True, timeout=30
            )
            return out.stdout.strip()
        except Exception as e:
            print(f"[FFmpeg Audio] yt-dlp resolve failed: {e}")
    return url

def record_audio_ffmpeg(src_url: str, duration: int = 60, sample_rate: int = 22050, seek_seconds: int = 0) -> Optional[bytes]:
    src = _resolve_youtube_audio_url(src_url)

    # --- This block is to make sure different kind of audio type can be extracted correctly ---
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]

    if seek_seconds > 0 and not ("youtube.com" in src_url or "youtu.be" in src_url):
         cmd += ["-ss", str(seek_seconds)]

    if str(src).startswith(("http://", "https://")):
        cmd += [
            "-reconnect", "1",
            "-reconnect_streamed", "1",
            "-reconnect_on_network_error", "1",
            "-fflags", "+genpts",
            "-flags", "low_delay"
        ]

    cmd += [
        "-i", src,
        "-map", "a:0?",    
        "-t", str(duration),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",
        "-f", "wav", "pipe:1",
        "-y"
    ]
    # --- ---


    try:
        print(f"[FFmpeg] extracting audio: {src} ({duration}s @ {sample_rate}Hz)")
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              check=True, timeout=duration + 15)
        return proc.stdout
    except subprocess.CalledProcessError as e:
        print(f"[FFmpeg] error (rc={e.returncode}): {e.stderr.decode(errors='ignore')}")
    except subprocess.TimeoutExpired:
        print("[FFmpeg] timed out reading audio.")
    return None

def background_audio_task(
        audio_url: str,
        duration: int,
        vocal_model,
        vocal_device,
        target_dict: Dict,
        seek_seconds: int = 0
):
    global analysis_in_progress
    try:
        analysis_in_progress = True
        wav_bytes = record_audio_ffmpeg(audio_url, duration=duration, seek_seconds=seek_seconds)
        if not wav_bytes:
            raise RuntimeError("FFmpeg no data")
        
        # --- Below is to save analyzed audio into local ---
        # output_dir = os.path.join("static", "audio_captures")
        # os.makedirs(output_dir, exist_ok=True)
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # output_filename = f"capture_{timestamp}.wav"
        # output_filepath = os.path.join(output_dir, output_filename)
        # with open(output_filepath, "wb") as f:
        #    f.write(wav_bytes)
        # print(f"[Audio Task] Saved captured audio to {output_filepath}")

        with BytesIO(wav_bytes) as bio:
            y, sr = sf.read(bio, dtype="float32")
        
        if hasattr(y, "ndim") and y.ndim > 1:
            y = np.mean(y, axis=1)
        sr = int(sr)

        pred, probs = vocalization_prediction(y, sr, vocal_model, vocal_device)

        target_dict["prediction"] = pred
        target_dict["probabilities"] = probs
        print("[Audio Task] Completed (ffmpeg path)")
    
    except Exception as e:
        print(f"[Audio Task] Failed: {e}")
        target_dict["prediction"] = "Error"
        target_dict["probabilities"] = None
    finally:
        analysis_in_progress = False
        