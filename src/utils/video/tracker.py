import os
import numpy as np
import boxmot
from boxmot.tracker_zoo import create_tracker

def _find_tracker_config(tracker_type: str) -> str:
    env_path = os.environ.get("BOXMOT_CFG", "")
    if env_path and os.path.exists(env_path):
        return env_path
    
    try:
        bm_dir = os.path.dirname(boxmot.__file__)
        candidates = [
            os.path.join(bm_dir, "configs", "trackers", f"{tracker_type}.yaml"),
            os.path.join(bm_dir, "trackers", "configs", f"{tracker_type}.yaml"),
            os.path.join(bm_dir, "configs", f"{tracker_type}.yaml")
        ]

        for p in candidates:
            if os.path.exists(p):
                return p
    except Exception:
        pass
    return ""

class MOTTracker:
    def __init__(self, tracker_type: str = "bytetrack", device: str = "cpu"):
        cfg_path = _find_tracker_config(tracker_type)
        if not cfg_path:
            raise RuntimeError(
                f"Could not locate config for '{tracker_type}'. "
                f"Set BOXMOT_CFG to the YAML path or ensure boxmot is installed correctly."
            )
        
        self.tracker = create_tracker(
            tracker_type = tracker_type,
            tracker_config = cfg_path,
            reid_weights = None,
            device = device,
            half = False
        )
    
    def update(self, dets_xyxy_conf_cls, frame):
        """ ensure each tracker is new """
        if dets_xyxy_conf_cls is None or len(dets_xyxy_conf_cls) == 0:
            dets_xyxy_conf_cls = np.empty((0, 6), dtype = "float32")
        
        tracks = self.tracker.update(dets_xyxy_conf_cls.astype("float32"), frame)
        
        out = []
        if tracks.shape[0] > 0:
            for row in tracks:
                x1, y1, x2, y2, tid, conf, cls = map(float, row[:7])
                out.append({"id": int(tid), "box": [x1, y1, x2, y2], "conf": conf, "cls": int(cls)})
        return out
    