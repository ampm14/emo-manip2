import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import torch
import torchaudio
import torch.nn.functional as F
from models.ravdess.simple_audio_cnn import SimpleAudioCNN  # adjust path if needed

MODEL_PATH = "models/ravdess/simple_audio_cnn_aug2_state_dict.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = SimpleAudioCNN(num_classes=8).to(device)
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.eval()

# Emotion labels (same as RAVDESS)
labels = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]

def predict_emotion(audio_path):
    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_mels=64, n_fft=1024, hop_length=512
    )(waveform)
    mel_spec = (mel_spec - mel_spec.mean()) / mel_spec.std()
    mel_spec = mel_spec.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(mel_spec)
        probs = F.softmax(logits, dim=1)
        pred_idx = probs.argmax(dim=1).item()
        pred_label = labels[pred_idx]
        return pred_label, probs.squeeze().cpu().numpy()

if __name__ == "__main__":
    audio_file = "data/eddie.wav"
    emotion, probs = predict_emotion(audio_file)
    print(f"Predicted emotion: {emotion}")
