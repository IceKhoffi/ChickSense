import numpy as np
import time
from typing import Dict, List, Optional 
import threading
import torch
from sklearn.cluster import DBSCAN

# -- Local Imports --
from config import APP_CONFIG, TUNING, INACTIVITY_CFG, DENSITY_DBSCAN_CFG
from .frame_reader import FrameReader
from .tracker import MOTTracker

class StreamProcessor:
    def __init__(self,
                 video_url: str,
                 model,
                 device: str = "cpu",
                 half: bool = False,
                 infer_lock=None):
        
        # -- Inisialisasi --
        self.video_url = video_url
        self.model = model
        self.device = device
        self.detection_interval = max(1, int(TUNING["DETECTION_INTERVAL_FRAMES"]))
        self.imgsz = int(TUNING["YOLO_IMG_SIZE"])
        self.half = bool(half and device == "cuda")
        self._infer_lock = infer_lock

        self.frame_reader = FrameReader(video_url)
        self.tracker = MOTTracker(tracker_type="bytetrack", device=device)
        self._running = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._lock = threading.Lock()

        self._latest_payload = None
        self._frame_idx = 0
        self._last_dets = None
        self.id_state: Dict[int, Dict] = {}

    def start(self):
        if self._running:
            return
        self._running = True
        self.frame_reader.start()
        self._thread.start()
    
    def stop(self):
        self._running = False
        self.frame_reader.stop()
        self._thread.join()
    
    def get_latest(self) -> Optional[Dict]:
        with self._lock:
            if self._latest_payload is None:
                return None
            return {
                "frame" : self._latest_payload["frame"].copy(), 
                "tracks" : list(self._latest_payload["tracks"]),
                "timestamp" : self._latest_payload["timestamp"],
                "frame_idx" : self._latest_payload["frame_idx"],
                "stats" : dict(self._latest_payload.get("stats", {}))
            }
    
    
    
    def _size_norm(self, box: List[float]) -> float:
        x1, y1, x2, y2 = box
        w = max(1.0, x2- x1)
        h = max(1.0, y2 - y1)
        return (w*w + h*h) ** 0.5
    
    # -- Inactivity Logic --
    def _update_inactivity(self, tracks: List[Dict], now: float):
        current_ids = set(t["id"] for t in tracks)

        for t in tracks:
            tid = t["id"]
            cx, cy = self._center(t["box"])
            diag = self._size_norm(t["box"])
            st = self.id_state.get(tid)

            if st is None:
                # please refer to "config.py" for each definition
                st = {"pos": (cx, cy), "t": now, "ema_v": 0.0, "inactive": False, "since": None, "last_seen": now}
                self.id_state[tid] = st
                t["inactive"] = False
                continue

            dt = max(1e-3, now - st["t"])
            dx = cx - st["pos"][0]
            dy = cy - st["pos"][1]

            v_norm = ((dx*dx + dy * dy)**0.5 / dt) / max(1.0, diag)
            alpha = INACTIVITY_CFG["EMA_ALPHA"]
            ema_v = alpha * v_norm + (1.0 - alpha) * st["ema_v"]

            entry = INACTIVITY_CFG["ENTER_THRESH_NORM_SPEED"]
            exit_ = INACTIVITY_CFG["EXIT_THRESH_NORM_SPEED"]
            dwell = INACTIVITY_CFG["MIN_DURATION_S"]

            if st["inactive"]:
                if ema_v > exit_:
                    st["since"] = st.get("since") or now
                    if (now - st["since"]) >= dwell:
                        st["inactive"] = False
                        st['since'] = None
                else:
                    st["since"] = None
            else:
                if ema_v < entry:
                    st["since"] = st.get("since") or now
                    if (now - st["since"]) >= dwell:
                        st["inactive"] = True
                        st["since"] = None
                else:
                    st["since"] = None

            st.update(pos = (cx, cy), t = now, ema_v = ema_v, last_seen = now)
            t["inactive"] = st["inactive"]

        # ensure old unseen ID removed
        stale = [
            tid for tid, st in list(self.id_state.items())
            if tid not in current_ids and (now - st.get("last_seen", now)) > INACTIVITY_CFG["MAX_UNSEEN_GAP_S"]
        ]
        for tid in stale:
            self.id_state.pop(tid, None)
        
    def _center(self, box: List[float]) -> tuple[float, float]:
        x1, y1, x2, y2 = box
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

    # -- Density Logic -- 
    def _compute_density_dbscan(self, tracks: List[Dict]) -> set:
        if not tracks:
            return set(), 0
        
        centers = np.array([self._center(t["box"]) for t in tracks], dtype=np.float32)
        # please refer to "config.py" for each definition
        min_samples = max(1, DENSITY_DBSCAN_CFG["MIN_NEIGHBORS"] + 1)
        labels = DBSCAN(eps=DENSITY_DBSCAN_CFG["EPS_PX"], min_samples=min_samples).fit_predict(centers)

        # below to count cluster that happend ( -1 is noise )
        n_clusters = int(len(set(lbl for lbl in labels if lbl != -1)))
        dense_ids = {t["id"] for t, lbl in zip(tracks, labels) if lbl != -1}
        return dense_ids, n_clusters

        # return {t["id"] for t, lbl in zip(tracks, labels) if lbl != -1}
    

    def _run(self):
        """ Detect Object from The Frame and add its metadata (Inactivity / Density) """
        while self._running:
            frame = self.frame_reader.read()
            if frame is None:
                time.sleep(0.01)
                continue
            self._frame_idx += 1

            # ensure frame to detect only on the interval
            if self._frame_idx % self.detection_interval == 1:
                try:
                    with torch.no_grad():
                        res = self.model.predict(
                            frame,
                            imgsz = self.imgsz,
                            device = self.device,
                            half = self.half,
                            verbose = False 
                        )[0]
                    
                    boxes = res.boxes
                    if boxes is not None and len(boxes) > 0:
                        dets = np.concatenate([
                            boxes.xyxy.cpu().numpy(),
                            boxes.conf.cpu().numpy()[:, None],
                            boxes.cls.cpu().numpy()[:, None]
                        ], axis = 1).astype("float32")
                    else:
                        dets = np.empty((0, 6), dtype="float32")
                    self._last_dets = dets
                except Exception as e:
                    print(f"[StreamProcessor] detection error: {e}")
                    self._last_dets = None
            dets = self._last_dets # use last know tracking to ensure trackig is keept

            # Update tracker
            try:
                tracks = self.tracker.update(dets, frame)
            except Exception as e:
                print(f"[StreamProcessor] tracking error: {e}")
                tracks = []
            
            # Update metadata
            now = time.time()
            self._update_inactivity(tracks, now)
            dense_ids, n_clusters = self._compute_density_dbscan(tracks)
            for t in tracks:
                t["dense"] = (t["id"] in dense_ids)
            
            stats = {
                "detected": len(tracks),
                "inactive": sum(1 for t in tracks if t.get("inactive")),
                "dense_clusters": n_clusters
            }

            # Store latest result
            with self._lock:
                self._latest_payload = {
                    "frame" : frame,
                    "tracks" : tracks,
                    "timestamp" : now,
                    "frame_idx" : self._frame_idx,
                    "stats" : stats
                }

class StreamRegistry:
    def __init__(self):
        self._by_url: Dict[str, StreamProcessor] = {}
        self._ref_count: Dict[str, int] = {}
        self._lock = threading.Lock()
    
    def get(self, url: str, model, device="cpu", half=False) -> StreamProcessor:
        """ Gather and made new StreamProcessor from each Video URL"""
        with self._lock:
            sp = self._by_url.get(url)
            if sp is None:
                print(f"[Registry] Creating new stream processor for: {url}")
                sp = StreamProcessor(url, model=model, device=device, half=half)
                sp.start()
                self._by_url[url] = sp
                self._ref_count[url] = 0
            self._ref_count[url] += 1
            print(f"[Registry] URL {url} ref count is now {self._ref_count[url]}")
            return sp
    
    def release(self, url: str):
        """ Ensure when stream video stop, threading is stop too """
        with self._lock:
            if url in self._by_url:
                self._ref_count[url] -= 1
                print(f"[Registry] URL {url} ref count is now {self._ref_count[url]}")
                if self._ref_count[url] <= 0:
                    print(f"[Registry] Stopping and removing processor for {url}")
                    try:
                        self._by_url[url].stop()
                    finally:
                        self._by_url.pop(url, None)
                        self._ref_count.pop(url, None)