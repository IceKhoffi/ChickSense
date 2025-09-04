import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F

class ModdifiedModel(nn.Module):
    def __init__(self, num_classes=3):
        super(ModdifiedModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(25088, 256),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

def audio_to_log_mel_spec(audio_data, sr=22050, wav_size=int(1.5 * 22050)):
    """
    Convert raw audio waveform to log-mel spectrogram.
    Args:
        audio_data: np.ndarray, audio waveform
        sr: int, sample rate
        wav_size: int, target length in samples (1.5 seconds)
    Returns:
        mel_spec_db: log-mel spectrogram (n_mels, time)
    """
    wav = librosa.util.normalize(audio_data)
    if len(wav) > wav_size:
        wav = wav[:wav_size]
    else:
        wav = np.pad(wav, (0, max(0, wav_size - len(wav))), "constant")
    
    n_fft = 2048
    hop_length = 256
    n_mels = 128
    mel_spec = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=n_fft,
                                              hop_length=hop_length, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db

def load_model(model_path):
    """Load trained model from .pth file."""
    model = ModdifiedModel()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def vocalization_prediction(audio_data, sample_rate, model, device=None):
    """
    Predict vocalization class from long audio by averaging over 1.5s windows.
    Args:
        audio_data: np.ndarray, long audio (e.g., 30 seconds)
        sample_rate: int, sample rate
        model: trained PyTorch model
        device: torch.device
    Returns:
        predicted_label: str
        probabilities: dict
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wav_size_samples = int(1.5 * sample_rate)
    step_size = wav_size_samples // 2  # 50% overlap
    class_labels = {0: 'Healthy', 1: 'Noise', 2: 'Unhealthy'}
    total_probs = np.zeros(len(class_labels))
    n_predictions = 0

    for start in range(0, len(audio_data) - wav_size_samples + 1, step_size):
        chunk = audio_data[start:start + wav_size_samples]
        try:
            mel_spec_db = audio_to_log_mel_spec(chunk, sr=sample_rate)
            X_tensor = torch.tensor(mel_spec_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            model = model.to(device)

            with torch.no_grad():
                output = model(X_tensor)
                probs = F.softmax(output, dim=1)[0].cpu().numpy()
            total_probs += probs
            n_predictions += 1
        except Exception as e:
            print(f"[Error] Failed on audio chunk: {e}")
            continue

    if n_predictions == 0:
        return "Error", {"Healthy": 0.0, "Noise": 0.0, "Unhealthy": 0.0}

    avg_probs = total_probs / n_predictions
    pred_class = np.argmax(avg_probs)
    predicted_label = class_labels[pred_class]
    prob_dict = {class_labels[i]: avg_probs[i] for i in range(len(class_labels))}

    return predicted_label, prob_dict