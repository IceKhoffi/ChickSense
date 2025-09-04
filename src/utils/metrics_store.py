import os
import sqlite3
import threading
import queue
import time
from config import DATA_STORAGE

SCHEMA = """
CREATE TABLE IF NOT EXISTS metrics (
    ts INTEGER NOT NULL, -- seconds (unix)
    camera_id INTEGER NOT NULL,
    detected INTEGER NOT NULL,
    dense_areas INTEGER NOT NULL,
    inactive INTEGER NOT NULL,
    PRIMARY KEY (ts, camera_id)
);
CREATE INDEX IF NOT EXISTS idx_metrics_cam_ts ON metrics(camera_id, ts);
"""

class MetricsStore:
    def __init__(self):
        path = DATA_STORAGE["SQLITE_DB_PATH"]
        retention_days = DATA_STORAGE["DB_RETENTION_DAYS"]
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.executescript(SCHEMA)
        self.retention_days = retention_days
        self.q = queue.Queue(maxsize=10000)
        self.t = threading.Thread(target=self._worker, daemon=True)
        self.t.start()
    
    def _worker(self):
        buf, last_flush, last_ret = [], time.time(), time.time()
        while True:
            try:
                item = self.q.get(timeout = 1.0)
                buf.append(item)
            except queue.Empty:
                pass

            now = time.time()
            if buf and (len(buf) >= DATA_STORAGE["DB_WRITE_BUFFER_SIZE"] or now - last_flush > DATA_STORAGE["DB_WRITE_INTERVAL_S"]):
                self.conn.executemany(
                    "INSERT OR REPLACE INTO metrics(ts,camera_id,detected,dense_areas,inactive) VALUES (?,?,?,?,?)",
                    buf
                )
                self.conn.commit()
                buf.clear()
                last_flush = now

            # keep retention on check every hour
            if now - last_ret > 3600:
                cutoff = int(time.time()) - self.retention_days * 86400
                self.conn.execute("DELETE FROM metrics WHERE ts < ?", (cutoff,))
                self.conn.commit()
                last_ret = now
                
    def write(self, ts, camera_id, detected, dense_areas, inactive):
        try:
            self.q.put_nowait((int(ts), int(camera_id), int(detected), int(dense_areas), int(inactive)))
        except queue.Full:
            pass  # drop if overloaded (rare), avoids blocking video loop
    
    def fetch_range(self, start_ts, end_ts, camera_id=None):
        """ retrieve data from database with specific time window """
        cur = self.conn.cursor()
        sql = "SELECT ts, camera_id, detected, dense_areas, inactive FROM metrics WHERE ts BETWEEN ? AND ?"
        params = [int(start_ts), int(end_ts)]
        if camera_id is not None:
            sql += " AND camera_id = ?"
            params.append(int(camera_id))
        sql += " ORDER BY ts ASC, camera_id ASC"
        cur.execute(sql, params)
        return cur.fetchall()

    def get_bounds(self):
        """ help to find the earliest and latest timestamps on the table """
        cur = self.conn.cursor()
        cur.execute("SELECT MIN(ts), MAX(ts) FROM metrics")
        row = cur.fetchone() or (None, None)
        return row