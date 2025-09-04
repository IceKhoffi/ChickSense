import os
import json
import time
import asyncio
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional
from datetime import datetime, timezone, timedelta
import csv
import io

import cv2
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from ultralytics import YOLO

from config import APP_CONFIG, TUNING, VISUALS, NOTIFIER
from utils.audio.audio_ingest import background_audio_task
from utils.video.stream_processor import StreamRegistry
from utils.audio.vocalization_prediction import load_model
from utils.metrics_store import MetricsStore
import utils.notifications as notifications

# -- FastAPI ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"]
)

app.mount("/static", StaticFiles(directory = "static"), name = "static")
INDEX_FILE_PATH = os.path.join("static", "index.html")

@app.get("/", response_class = FileResponse)
async def read_index():
    if not os.path.exists(INDEX_FILE_PATH):
        return HTMLResponse("index.html missing", status_code = 404)
    return FileResponse(INDEX_FILE_PATH)

# -- Global state --
latest_audio_result = {
    "prediction": None,
    "probabilities": None
}

audio_result_pending = False
analysis_in_progress = False
_last_appended_audio_sig = None

last_audio_trigger_time = 0.0
stream_registry: Optional[StreamRegistry] = None
executor = ThreadPoolExecutor(max_workers=4)

audio_seek_state = {"url": None, "seek_seconds": 0}
current_audio_url: Optional[str] = None
audio_loop_task: Optional[asyncio.Task] = None

# - Notifications JSON builder -
_sensor_json_lock = threading.Lock()
sensor_state: Dict[str, dict] = {}
inactive_since: Dict[str, float] = {}

global_vocal_history: deque[str] = deque(maxlen=5)
metrics_store = MetricsStore()

# -- Models --
class AudioTriggerRequest(BaseModel):
    audio_url_: str

# -- Helpers method --

def _draw_overlays(frame, tracks, display_settings: Dict[str, bool]):
    """ Ensure boxes/labels displayed according to the toggles """
    for t in tracks:
        x1, y1, x2, y2 = map(int, t["box"])
        color = None
        if t.get("inactive") and display_settings["show_inactive"]:
            color = VISUALS["COLOR_INACTIVE"]
        elif t.get("dense") and display_settings["show_density"]:
            color = VISUALS["COLOR_DENSE"]
        elif display_settings["show_detected"]:
            color = VISUALS["COLOR_DETECTED"]

        if color is not None:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID {t['id']}", (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return frame

def _encode_frame_jpeg(frame, jpeg_q: int) -> Optional[bytes]:
    ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_q])
    return jpg.tobytes() if ok else None

def _scale_frame(frame, max_w: int):
    h, w = frame.shape[:2]
    if w <= max_w:
        return frame
    scale = max_w / float(w)
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

def _update_sensor_json(camera_id: int, detected_count: int, inactive_count: int, dense_clusters: int, latest_audio: Dict):
    """ Build or rewrite camera record in sensor_data.json """
    now = time.time()
    cam_key = str(camera_id)

    # Update Inactivity
    if inactive_count > 0:
        if cam_key not in inactive_since:
            inactive_since[cam_key] = now
    else:
        inactive_since.pop(cam_key, None)

    raw_pred = (latest_audio or {}).get("prediction")
    global audio_result_pending, _last_appended_audio_sig

    pred = str(raw_pred).strip() if raw_pred else None
    sig = None
    probs = {}
    if latest_audio and pred:
        try:
            probs = dict(latest_audio.get("probabilities") or {})
            rounded_probs = {k: round(float(v), 4) for k, v in probs.items()}
            sig = (pred, tuple(sorted(rounded_probs.items())))
        except (ValueError, TypeError, OverflowError):
            sig = (pred, None)
    else:
        sig = (pred, None) if pred else None
    
    # Append "IF" prediction changed
    if pred and (audio_result_pending or sig != _last_appended_audio_sig):
        global_vocal_history.append(pred)
        _last_appended_audio_sig = sig
        audio_result_pending = False
    
    mic_entry = {
        "camera_id": "MIC",
        "vocalization_history": list(global_vocal_history)  # Full history
    }

    if pred:
        rounded_probs = {k: round(float(v), 4) for k, v in probs.items()}
        mic_entry["latest_prediction"] = pred
        mic_entry["latest_probabilities"] = rounded_probs
    
    sensor_state["MIC"] = mic_entry

    # Update entry
    sensor_state[cam_key] = {
        "camera_id": cam_key,
        "detected_count": int(detected_count),
        "inactive_count": int(inactive_count),
        "density_count": int(dense_clusters or 0)
    }

    # Write to sensor_data.json
    json_path = NOTIFIER["SENSOR_DATA_JSON_PATH"]
    tmp_path = json_path + ".tmp"
    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)

    with _sensor_json_lock:
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(list(sensor_state.values()), f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, json_path)
        except Exception as e:
            print(f"[Error] Failed to write sensor_data.json: {e}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

def clear_sensor_data_json():
    """ Initialize empty sensor_data.json went startup """
    json_path = NOTIFIER["SENSOR_DATA_JSON_PATH"]
    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([], f, indent=2)

async def periodic_audio_analyzer():
    """ Handle audio analysis """
    global last_audio_trigger_time, latest_audio_result, audio_result_pending, audio_seek_state

    await asyncio.sleep(15)

    print("[Audio Loop] Starting audio analysis.")

    while True:
        if not current_audio_url:
            print("[Audio Loop] Waiting for an audio URL to be provided by a client.")
            await asyncio.sleep(APP_CONFIG["AUDIO_ANALYSIS_INTERVAL_S"])
            continue

        if current_audio_url != audio_seek_state["url"]:
            print(f"[Audio Loop] New audio source detected. Resetting seek time for {current_audio_url}")
            audio_seek_state["url"] = current_audio_url
            audio_seek_state["seek_seconds"] = 0

        now = time.time()
        if now - last_audio_trigger_time >= APP_CONFIG["AUDIO_ANALYSIS_INTERVAL_S"]:

            # Get the current bookmark/seek time
            seek_time = audio_seek_state["seek_seconds"]
            duration = APP_CONFIG["AUDIO_ANALYSIS_DURATION_S"]

            print(f"[Audio Loop] Triggering analysis for: {current_audio_url} (starting at {seek_time}s)")
            last_audio_trigger_time = now

            latest_audio_result["prediction"] = None
            latest_audio_result["probabilities"] = None
            audio_result_pending = True

            executor.submit(
                background_audio_task,
                current_audio_url,
                duration,
                vocal_model,
                vocal_device,
                latest_audio_result,
                seek_seconds=seek_time  # Pass the seek time to the task
            )

            # Move the bookmark forward for the next run
            audio_seek_state["seek_seconds"] += duration
        
        await asyncio.sleep(1)

# -- Websocket --
@app.websocket("/ws/video_feed")
async def websocket_endpoint(websocket: WebSocket):
    """ Push JPEG frames with stats over WS """
    await websocket.accept()
    video_url = None
    audio_url = None
    sp = None

    global current_audio_url

    try:
        # Made initial handshake with config
        init_msg = await asyncio.wait_for(websocket.receive_text(), timeout = 10.0)
        init_data = json.loads(init_msg)

        target_fps = int(init_data.get("target_fps", TUNING["WEBSOCKET_TARGET_FPS"]))
        jpeg_q = int(init_data.get("jpeg_quality", TUNING["WEBSOCKET_JPEG_QUALITY"]))
        max_w = int(init_data.get("display_max_width", TUNING["WEBSOCKET_DISPLAY_MAX_WIDTH"]))

        min_dt = 1.0 / max(1, target_fps)
        last_send = 0.0

        video_url = init_data.get("video_url")
        audio_url_from_client = init_data.get("audio_url", video_url)
        if audio_url_from_client:
            current_audio_url = audio_url_from_client

        camera_id = int(init_data.get("camera_id", 0))
        last_stats_sent = 0.0

        if not video_url:
            await websocket.send_text(json.dumps({"type": "status", "message": "Error: URL type unkown"}))
            raise WebSocketDisconnect(code=1008, reason="URL not provided")
        
        display_settings = {
            "show_detected": bool(init_data.get("show_detected", False)),
            "show_density": bool(init_data.get("show_density", False)),
            "show_inactive": bool(init_data.get("show_inactive", False))
        }

        print(f"[WebSocket {websocket.client}] Stream start : {video_url}, Audio start: {audio_url}")

        # Get or create the stream processor instance
        sp = stream_registry.get(
            video_url,
            model = yolo_model,
            device = _device,
            half = (_device == "cuda")
        )

        while True:

            # WS control messages (toggles display)
            try:
                msg_str = await asyncio.wait_for(websocket.receive_text(), timeout=0.3)
                msg = json.loads(msg_str)
                if msg.get("type") == "display_settings_update":
                    for key in ["show_detected", "show_density", "show_inactive"]:
                        if key in msg:
                            display_settings[key] = bool(msg[key])
                    await websocket.send_text(json.dumps({"type": "status", "message": "Display settings updated"}))
                elif msg.get("type") == "update_audio_url" and msg.get("audio_url"):
                    audio_url = msg["audio_url"]
                    await websocket.send_text(json.dumps({"type": "status", "message": f"Audio start updated: {audio_url}"}))
            except asyncio.TimeoutError:
                pass

            # Fetch latest processed frame and tracks
            payload = sp.get_latest()
            if payload is None:
                await asyncio.sleep(0.01)
                continue

            frame = payload["frame"]
            tracks = payload["tracks"]

            _draw_overlays(frame, tracks, display_settings)

            # Scale and encode
            frame = _scale_frame(frame, max_w)
            jpg_bytes = _encode_frame_jpeg(frame, jpeg_q)
            if jpg_bytes is not None:
                now = time.time()
                if now - last_send >= min_dt:  # Cap send rate
                    await websocket.send_bytes(jpg_bytes)
                    last_send = now
            
            # Stats
            stats = payload.get("stats", {}) or {}
            now = time.time()
            if now - last_stats_sent > 1.0:
                detected = int(stats.get("detected", 0))
                inactive = int(stats.get("inactive", 0))
                dense_clusters = int(stats.get("dense_clusters", 0))

                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "stats",
                            "detected": detected,
                            "inactive": inactive,
                            "dense_areas": dense_clusters
                        }
                    )
                )

                # Write to DB
                metrics_store.write(
                    ts=now,
                    camera_id=camera_id,
                    detected=detected,
                    dense_areas=dense_clusters,
                    inactive=inactive
                )
                last_stats_sent = now

                # Update sensor JSON with full counts
                try:
                    _update_sensor_json(
                        camera_id=camera_id,
                        detected_count=detected,
                        inactive_count=inactive,
                        dense_clusters=dense_clusters,
                        latest_audio=latest_audio_result
                    )
                except Exception as e:
                    print(f"[Notif] sensor_data.json failed to update on camera {camera_id}: {e}")
            
            await asyncio.sleep(max(0.005, min_dt * 0.25))

    except WebSocketDisconnect as e:
        print(f"WebSocket client {websocket.client} disconnected: (Code: {e.code}, Reason: {e.reason})")
    except asyncio.TimeoutError:
        print(f"WebSocket {websocket.client} timed out waiting for initial message.")
    except Exception as e:
        print(f"[Error] WebSocket failed for {websocket.client}: {e}")
    finally:
        if video_url and stream_registry:
            stream_registry.release(video_url)
        try:
            await websocket.close()
        except Exception:
            pass
        print(f"[WebSocket] Cleaned up for {websocket.client}.")

# Audio Endpoints
@app.post("/trigger_audio_analysis", status_code=202)
async def trigger_audio_analysis(request: AudioTriggerRequest):
    """Manual, fire-and-forget trigger for short audio analysis window."""
    global audio_result_pending, analysis_in_progress
    if analysis_in_progress:
        return {"message": "Analysis already running; skipping duplicate trigger."}

    print(f"[API] Audio analysis triggered: {request.audio_url_}")
    latest_audio_result["prediction"] = None
    latest_audio_result["probabilities"] = None
    audio_result_pending = True

    executor.submit(
        background_audio_task,
        request.audio_url_,
        APP_CONFIG["AUDIO_ANALYSIS_DURATION_S"],
        vocal_model,
        vocal_device,
        latest_audio_result,
    )
    return {"message": "Audio analysis started"}

@app.get("/get_latest_audio_result")
async def get_latest_audio_result_endpoint():
    """ Return last result or status if analysis is pending """
    if analysis_in_progress:
        return {
            "prediction": None,
            "probabilities": None,
            "status": "analyzing"
        }
    elif latest_audio_result["prediction"] is not None:
        return {
            **latest_audio_result,
            "status": "completed"
        }
    else:
        return {
            "prediction": None,
            "probabilities": None,
            "status": "no_data"
        }

# Startup
@app.on_event("startup")
async def startup_event():
    """ setup for model and background notifier """
    clear_sensor_data_json()

    global yolo_model, vocal_model, vocal_device, _device, stream_registry, audio_loop_task
    print("Trying to load model")

    _device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load YOLO model
    try:
        yolo_model = YOLO(APP_CONFIG["YOLO_MODEL_PATH"])
        yolo_model.fuse()
        yolo_model.to(_device)
        try:
            if _device == "cuda" and hasattr(yolo_model, "model"):
                yolo_model.model.half()  # FP16 on CUDA
                print("[Startup] yolo model run on FP16 CUDA")
        except Exception:
            pass
        print(f"YOLO model loaded on {_device}")
    except Exception as e:
        print(f"[Error] Failed to load YOLO model: {e}")
    
    # Load Vocal model
    try:
        vocal_device = torch.device(_device)
        vocal_model = load_model(APP_CONFIG["VOCAL_MODEL_PATH"]).to(vocal_device).eval()
        print(f"Vocal model loaded on {vocal_device}.")
    except Exception as e:
        print("[Error] Failed to load Vocal model", e)

    stream_registry = StreamRegistry()
    print("Models has been load succesfully")

    # Start single centralized audio analysis loop
    audio_loop_task = asyncio.create_task(periodic_audio_analyzer())

    # Run Notifications
    try:
        if NOTIFIER["ENABLE_TELEGRAM_NOTIFICATIONS"]:
            threading.Thread(target=notifications.main, daemon=True).start()
            print("[Notifications] started")
        else:
            print("[Notifications] Disabled (ENABLE_TELEGRAM_NOTIFICATIONS=False)")
    except Exception as e:
        print(f"[Notifications] Failed to start: {e}")

# CSV exporter helper and endpoints
def _parse_date_yyyy_mm_dd(s: str) -> int:
    """Parse 'YYYY-MM-DD' as midnight UTC (returns seconds)."""
    try:
        dt = datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except Exception:
        raise HTTPException(status_code=400, detail="Dates must be in YYYY-MM-DD format")

def _fallback_offset_minutes() -> int:
    off = datetime.now().astimezone().utcoffset() or timedelta(0)
    return int(off.total_seconds() // 60)

@app.get("/metrics/available")
async def metrics_available(tz_offset_minutes: int | None = None):
    min_ts, max_ts = metrics_store.get_bounds()
    if min_ts is None or max_ts is None:
        return {"min": None, "max": None}
    minutes = tz_offset_minutes if tz_offset_minutes is not None else _fallback_offset_minutes()
    offset = timedelta(minutes=int(minutes))
    min_day = (datetime.fromtimestamp(min_ts, tz=timezone.utc) + offset).date().isoformat()
    max_day = (datetime.fromtimestamp(max_ts, tz=timezone.utc) + offset).date().isoformat()
    return {"min": min_day, "max": max_day}

@app.get("/metrics/export")
async def export_metrics(
    start: str,
    end: str,
    camera_id: int | None = None,
    tz_offset_minutes: int = 7 * 60,
    raw: bool = True # ensure retained (we want to use raw data)
):
    """ Export metrics into CSV : datetime_local,camera_id,detected,dense_areas,inactive """
    start_ts_utc = _parse_date_yyyy_mm_dd(start)
    end_ts_utc = _parse_date_yyyy_mm_dd(end) + 86399 # 86399 (seconds) -> 24-hours - 1 sec

    rows = metrics_store.fetch_range(start_ts_utc, end_ts_utc, camera_id=camera_id)

    minutes = tz_offset_minutes if tz_offset_minutes is not None else _fallback_offset_minutes()
    offset_seconds = int(minutes) * 60
    
    start_local = datetime.strptime(start, "%Y-%m-%d").date()
    end_local = datetime.strptime(end, "%Y-%m-%d").date()

    def csv_iter_raw():
        sio = io.StringIO()
        w = csv.writer(sio)
        w.writerow(["datetime_local", "camera_id", "detected", "dense_areas", "inactive"])
        yield sio.getvalue(); sio.seek(0); sio.truncate(0)

        for ts, cam, det, dense, ina in rows:
            local_dt = datetime.fromtimestamp(ts + offset_seconds, tz=timezone.utc)
            d = local_dt.date()
            if start_local <= d <= end_local:
                w.writerow([local_dt.strftime("%Y-%m-%d %H:%M:%S"), cam, int(det), int(dense), int(ina)])
                yield sio.getvalue(); sio.seek(0); sio.truncate(0)
    
    fname = f"metrics_raw_{start}_to_{end}{('_cam' + str(camera_id)) if camera_id else ''}.csv"
    return StreamingResponse(
        csv_iter_raw(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", reload=False)
    