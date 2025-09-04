import os

# Main Config
APP_CONFIG = {
    "YOLO_MODEL_PATH" : "./models/yolov8n.pt",
    "VOCAL_MODEL_PATH" : "./models/Chicken_CNN_Disease_Detection_Model.pth",

    "AUDIO_ANALYSIS_DURATION_S" : 30,
    "AUDIO_ANALYSIS_INTERVAL_S" : 60
}

# Processing Config
TUNING = {
    # How Video Processed
    "YOLO_IMG_SIZE" : 512,
    "DETECTION_INTERVAL_FRAMES" : 24,
    "FRAME_READER_BUFFER_SIZE" : 5,
    "FRAME_READER_FPS" : 15,

    # How Streaming Processed
    "WEBSOCKET_TARGET_FPS" : 8,
    "WEBSOCKET_JPEG_QUALITY" : 40, # JPEG quality ( 0-100 )
    "WEBSOCKET_DISPLAY_MAX_WIDTH" : 640
}

# Inactivity Config
INACTIVITY_CFG = {
    "EMA_ALPHA" : 0.2,
    "ENTER_THRESH_NORM_SPEED" : 0.02,
    "EXIT_THRESH_NORM_SPEED" : 0.05,
    "MIN_DURATION_S" : 60, # (7200) 2 Hours inactive of an object before flag as inactive
    "MAX_UNSEEN_GAP_S" : 1.5
}

# Density Clustering Config
DENSITY_DBSCAN_CFG = {
    "EPS_PX" : 60.0, # Max pixels between to object to be called neighbors
    "MIN_NEIGHBORS" : 4 # Min objects required to form a dense cluster
}

# Database Config
DATA_STORAGE = {
    "SQLITE_DB_PATH" : "data/metrics.sqlite",
    "DB_RETENTION_DAYS" : 90,
    "DB_WRITE_BUFFER_SIZE" : 500,
    "DB_WRITE_INTERVAL_S" : 1.0
}

# Visual Config
VISUALS = {
    "COLOR_DETECTED" : (0, 255, 0), # Green
    "COLOR_DENSE" : (0, 165, 255), # Orange
    "COLOR_INACTIVE": (0, 0, 255), # Red
}

# Notifier Config
NOTIFIER = {
    "ENABLE_TELEGRAM_NOTIFICATIONS" : os.environ.get("ENABLE_TELEGRAM_NOTIFICATIONS", "n").lower().startswith("y"),
    "SENSOR_DATA_JSON_PATH" : "data/sensor_data.json",
    "TELEGRAM_BOT_TOKEN": os.environ.get("TELEGRAM_BOT_TOKEN", ""),
    "TELEGRAM_CHAT_ID": os.environ.get("TELEGRAM_CHAT_ID", ""),

    "INACTIVE_PERCENTAGE_THRESHOLD": 0.15, # The percentage threshold of inactive objects
    "UNHEALTHY_HISTORY_LENGTH": 5, # The number of recent data points to be stored in the history
    "UNHEALTHY_ALERT_THRESHOLD": 5, # The number of consecutive "unhealthy" statuses required within the history
    "DENSITY_COUNT_THRESHOLD": 5 # The number of Unique Density before alret was sent
}