import asyncio
import json
import time
import logging
from telegram import Bot
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from config import NOTIFIER

ENABLE_TELEGRAM_NOTIFICATIONS = NOTIFIER["ENABLE_TELEGRAM_NOTIFICATIONS"]
TELEGRAM_BOT_TOKEN = NOTIFIER["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = NOTIFIER["TELEGRAM_CHAT_ID"]

INACTIVE_PERCENTAGE_THRESHOLD = NOTIFIER["INACTIVE_PERCENTAGE_THRESHOLD"]
UNHEALTHY_HISTORY_LENGTH = NOTIFIER["UNHEALTHY_HISTORY_LENGTH"]
UNHEALTHY_ALERT_THRESHOLD = NOTIFIER["UNHEALTHY_ALERT_THRESHOLD"]
DENSITY_COUNT_THRESHOLD = NOTIFIER["DENSITY_COUNT_THRESHOLD"]

# --- Helper ---
def _now_local() -> datetime:
    return datetime.now().astimezone()

@dataclass
class AlertState:
    """ Tracks each camera alert state and cooldown """
    sent_inactive_danger: bool = False
    sent_unhealthy_danger: bool = False
    sent_density_danger: bool = False
    last_danger_alert_time: float = 0.0 # timestamp
    last_warning_alert_time: float = 0.0
    cooldown_period_danger: int = 1800 # 30 minutes
    cooldown_period_warning: int = 3600

class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str, *, logger: Optional[logging.Logger] = None):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self._logger = logger or logging.getLogger(__name__)
    
    async def _send_async(self, text: str) -> bool:
        try:
            bot = Bot(token=self.bot_token)
            await bot.send_message(chat_id=self.chat_id, text=text, parse_mode="HTML", disable_web_page_preview=True)
            return True
        except Exception as e:
            self._logger.error("[Telegram] Gagal kirim pesan: %s", e)
            return False
    
    def send(self, text: str) -> bool:
        """ Send message """
        try:
            return asyncio.run(self._send_async(text))
        except Exception as e:
            self._logger.error("[Telegram] Gagal jalankan async: %s", e)
            return False

SENSOR_JSON_PATH = Path(NOTIFIER["SENSOR_DATA_JSON_PATH"])
alert_states: Dict[str, AlertState] = {}

def _load_sensor_data() -> List[Dict[str, Any]]:
    """ Load sensor_data.json """
    if not SENSOR_JSON_PATH.exists():
        return []
    try:
        with SENSOR_JSON_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception as e:
        logging.error("[Sensor JSON] Gagal baca: %s", e)
        return []

def _count_unhealthy(vocal_history: List[str]) -> int:
    """ Count 'Unhealthy' """
    recent = vocal_history[-UNHEALTHY_HISTORY_LENGTH:]
    return sum(1 for s in recent if isinstance(s, str) and s.strip() == "Unhealthy")

def _get_prediction_with_prob(entry: Dict) -> tuple[str, float]:
    pred = entry.get("prediction")
    probs = entry.get("probabilities", {})
    if pred and pred in probs:
        return pred, float(probs[pred])
    return pred or "Unknown", 0.0

def generate_alert_message(
        camera_id: str,
        inactive_ratio: float,
        audio_pred: str,
        audio_prob: float,
        density_count: int,
        unhealthy_count: int,
        timestamp: str
) -> str:
    details = []
    is_danger = False
    is_warning = False
    is_mic = camera_id == "MIC"

    # Inactivity Logic
    if not is_mic:
        if inactive_ratio >= INACTIVE_PERCENTAGE_THRESHOLD:
            is_danger = True
            pcnt = round(inactive_ratio * 100, 1)
            details.append(f"Kamera {camera_id}: Inaktivitas ayam mencapai <b>{pcnt}%</b>.")
        elif inactive_ratio > 0:
            is_warning = True
            details.append(f"Kamera {camera_id}: Terdeteksi beberapa ayam tidak aktif.")
    
    # Density Logic
    if not is_mic:
        if density_count >= DENSITY_COUNT_THRESHOLD:
            is_danger = True
            details.append(f"Kamera {camera_id}: Terdeteksi <b>{density_count}</b> area dengan kepadatan tinggi.")
        elif density_count > 1:
            is_warning = True
            details.append(f"Kamera {camera_id}: Terdeteksi kepadatan ayam lokal.\n({density_count} cluster)")
    
    # Audio Logic
    if audio_pred == "Unhealthy":
        pcnt = round(audio_prob * 100, 1)
        if unhealthy_count >= UNHEALTHY_ALERT_THRESHOLD:
            is_danger = True
            details.append(f"Audio {camera_id}: Tanda <b>Tidak Sehat</b> terdeteksi berulang (Prob: {pcnt}%).")
        else:
            is_warning = True
            details.append(f"Audio {camera_id}: Terdengar tanda awal <b>Tidak Sehat</b> (Prob: {pcnt}%).")
    
    if not is_danger and not is_warning:
        return ""
    
    end_parts = []
    if is_danger:
        end_parts.append("🔴 <b>[BAHAYA]</b>")
    elif is_warning:
        end_parts.append("🟠 <b>[PERINGATAN]</b>")
    
    # Build the message
    end_parts.append("")
    end_parts.extend(details)
    end_parts.append("")
    end_parts.append("<b>Mohon segera periksa kondisi kandang!</b>")
    end_parts.append(f"<i>Waktu: {timestamp}</i>")

    return "\n".join(end_parts)

def monitor_once(notifier: Optional[TelegramNotifier]) -> None:
    data = _load_sensor_data()
    if not data:
        return
    
    timestamp_str = _now_local().strftime("%Y-%m-%d %H:%M:%S")

    for item in data:
        camera_id = str(item.get("camera_id", "unknown"))
        detected_count = item.get("detected_count", 0)
        inactive_count = item.get("inactive_count", 0)
        density_count = item.get("density_count", 0)
        vocal_history = item.get("vocalization_history", [])
        latest_audio_pred = vocal_history[-1] if vocal_history else None
    
        audio_pred, audio_prob = _get_prediction_with_prob({
            "prediction": latest_audio_pred,
            "probabilities": item.get("latest_probabilities", {})
        })
        unhealthy_count = _count_unhealthy(vocal_history)

        inactive_ratio = inactive_count / detected_count if detected_count > 0 else 0.

        if camera_id not in alert_states:
            alert_states[camera_id] = AlertState()
        state = alert_states[camera_id]
    
        # Send alert State
        now = time.time()
        should_send = False
        message_type = "info"
        is_mic = camera_id == "MIC"

        # Type : Danger
        danger_cooldown_expired = (now - state.last_danger_alert_time) >= state.cooldown_period_danger
        if not is_mic and inactive_ratio >= INACTIVE_PERCENTAGE_THRESHOLD:
            if not state.sent_inactive_danger or danger_cooldown_expired:
                should_send = True
                message_type = "danger"
                state.sent_inactive_danger = True
                state.last_danger_alert_time = now
        
        if unhealthy_count >= UNHEALTHY_ALERT_THRESHOLD:
            if not state.sent_unhealthy_danger or danger_cooldown_expired:
                should_send = True
                message_type = "danger"
                state.sent_unhealthy_danger = True
                state.last_danger_alert_time = now
        
        if not is_mic and density_count >= DENSITY_COUNT_THRESHOLD:
            if not state.sent_density_danger or danger_cooldown_expired:
                should_send = True
                message_type = "danger"
                state.sent_density_danger = True
                state.last_danger_alert_time = now
    
        # Type : Warning
        warning_cooldown_expired = (now - state.last_warning_alert_time) >= state.cooldown_period_warning
        if not should_send and not is_mic and inactive_count > 0 and not state.sent_inactive_danger:
            if warning_cooldown_expired:
                should_send = True
                message_type = "warning"
                state.last_warning_alert_time = now
        
        if not should_send and audio_pred == "Unhealthy" and not state.sent_unhealthy_danger:
            if warning_cooldown_expired:
                should_send = True
                message_type = "warning"
                state.last_warning_alert_time = now
        
        if not should_send and not is_mic and density_count > 1 and not state.sent_density_danger:
            if warning_cooldown_expired:
                should_send = True
                message_type = "warning"
                state.last_warning_alert_time = now
    
        # Send alert if TRUE
        if should_send:
            msg = generate_alert_message(
                camera_id=camera_id,
                inactive_ratio=inactive_ratio,
                audio_pred=audio_pred,
                audio_prob=audio_prob,
                density_count=density_count,
                unhealthy_count=unhealthy_count,
                timestamp=timestamp_str
            )

            if msg.strip() and notifier and ENABLE_TELEGRAM_NOTIFICATIONS:
                success = notifier.send(msg)
                logging.info("[Notif] %s alert sent for Camera %s: %s", message_type.upper(), camera_id, success)
            else:
                print(f"[Notif][Dry-Run] {message_type.upper()}:\n{msg}\n")
    
        # State Reset
        if inactive_ratio < INACTIVE_PERCENTAGE_THRESHOLD:
            state.sent_inactive_danger = False
        if unhealthy_count < UNHEALTHY_ALERT_THRESHOLD:
            state.sent_unhealthy_danger = False
        if density_count < DENSITY_COUNT_THRESHOLD:
            state.sent_density_danger = False

def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

    # Initialize notifier
    notifier: Optional[TelegramNotifier] = None
    if ENABLE_TELEGRAM_NOTIFICATIONS and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        notifier = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, logger=logging.getLogger("Telegram"))
        logging.info("[Notif] Telegram aktif.")
    else:
        logging.info("[Notif] Telegram dinonaktifkan. Mode dry-run.")

    logging.info("[Notif] Memantau sensor_data.json setiap 60 detik...")
    
    while True:
        try:
            monitor_once(notifier)
        except Exception as e:
            logging.error("[Notif] Error saat monitoring: %s", e)
        time.sleep(60)

if __name__ == "__main__":
    main()


