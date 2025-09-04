import os
import re
import cv2
import time
import threading
import subprocess
from collections import deque
from config import TUNING

class FrameReader(threading.Thread):
    def __init__(self, video_url: str):
        super().__init__(daemon=True)
        self.video_url = video_url
        self.buffer = deque(maxlen=TUNING["FRAME_READER_BUFFER_SIZE"])
        self.fps = TUNING["FRAME_READER_FPS"]
        self.running = threading.Event()
        self.cap = None
        self._is_file = self._looks_like_file(video_url)
    
    def _looks_like_file(self, url: str) -> bool:
        if os.path.exists(url):
            return True
        return not re.match(r'^[a-zA-Z]+://', url or "")
    
    def _resolve_url(self, url: str) -> str:
        if "youtube.com" in url or "youtu.be" in url:
            try:
                print(f"[FrameReader] Resolving YouTube URL: {url}")

                result = subprocess.run(
                    ["yt-dlp", "--get-url", url],
                    capture_output=True, text=True, check=True, timeout=30
                )
                urls = result.stdout.strip().splitlines()
                if not urls:
                    print("[FrameReader] No URLs returned by yt-dlp")
                    return None

                stream_url = urls[0]
                print(f"[FrameReader] Resolved to stream: {stream_url}")
                return stream_url

            except subprocess.CalledProcessError as e:
                stderr = e.stderr.strip()
                print(f"[FrameReader] yt-dlp failed with error: {stderr}")
                if "unavailable" in stderr:
                    print("[FrameReader] Video may be offline, private, or geo-restricted.")
                return None
            except Exception as e:
                print(f"[FrameReader] Unexpected error resolving YouTube URL: {e}")
                return None

        return url
    
    def run(self):
        """ Read each frame of a video and store in buffer """
        resolved = self._resolve_url(self.video_url)
        self.cap = cv2.VideoCapture(resolved, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            print(f"[FrameReader] Cannot open: {resolved}")
            return
        
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        self.running.set()

        # Count Time & FPS
        src_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if not src_fps or src_fps <= 1e-3:
            src_fps = float(self.fps) if self.fps else 30.0
        frame_period = 1.0 / float(src_fps)
        next_ts = time.monotonic()

        # Read Frame Loop
        while self.running.is_set():
            ret, frame = self.cap.read()
            if not ret:
                if os.path.exists(self.video_url):
                    break
                time.sleep(0.5)
                continue

            if frame is not None:
                self.buffer.append(frame)
            
            # Time Sync
            next_ts += frame_period
            sleep_for = next_ts - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_ts = time.monotonic()
        
        if self.cap:
            self.cap.release()
    
    def stop(self):
        self.running.clear()
        self.join()
    
    def read(self):
        return self.buffer[-1].copy() if self.buffer else None
    