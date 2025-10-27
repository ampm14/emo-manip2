# src/rafdb/infer.py
"""
Lightweight RAF-DB inference module.
- Quietly forces CPU-only ONNXRuntime (avoids GPU discovery noise).
- Falls back to TorchScript if ONNX not available.
- Uses MTCNN (facenet-pytorch) if available, otherwise OpenCV Haar.
- Exposes: predict_image(PIL.Image) and predict_frame_bgr(numpy BGR frame).
"""

import os
# Prevent ONNXRuntime from probing GPUs (must be set before importing onnxruntime)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["ORT_LOG_VERBOSITY_LEVEL"] = "0"

import json
import warnings
from pathlib import Path
from typing import Tuple, Dict, Optional

import numpy as np
from PIL import Image
import cv2

# ---- model & artifact paths (adjusted to repo root: ../models/rafdb) ----
MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "rafdb"
ONNX_PATH = MODEL_DIR / "raf_model.onnx"
TS_PATH = MODEL_DIR / "raf_model_ts.pt"
LABEL_MAP = MODEL_DIR / "raf_label_map.json"

# ---- try ONNXRuntime (CPU only) and silence its warnings/logs ----
SESSION = None
INPUT_NAME = None
BACKEND = None
try:
    import onnxruntime as ort
    # guard Python-level warnings from onnxruntime
    warnings.filterwarnings("ignore", category=UserWarning, module="onnxruntime")
    # reduce ORT logger noise if available
    try:
        ort.set_default_logger_severity(3)  # 0=verbose .. 4=fatal
    except Exception:
        pass
    # create a CPU-only session to avoid GPU discovery noise on machines without proper drivers
    SESSION = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    INPUT_NAME = SESSION.get_inputs()[0].name
    BACKEND = "onnx"
except Exception:
    SESSION = None
    BACKEND = None

# ---- preprocessing constants (must match training) ----
IMG_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ---- load label map (fail fast with helpful message) ----
if not Path(LABEL_MAP).exists():
    raise FileNotFoundError(f"Label map not found at {LABEL_MAP}. Place raf_label_map.json in models/rafdb/")
with open(LABEL_MAP, "r") as f:
    label_map = json.load(f)

# Accept either {"idx2label": {...}} or direct dict mapping strings
if "idx2label" in label_map:
    raw_idx2label = label_map["idx2label"]
else:
    raw_idx2label = label_map

# ensure int -> str mapping
idx2label = {int(k): v for k, v in raw_idx2label.items()}

# ---- if ONNX wasn't available, try TorchScript fallback ----
model_ts = None
if BACKEND is None:
    try:
        import torch
        if Path(TS_PATH).exists():
            model_ts = torch.jit.load(str(TS_PATH), map_location="cpu")
            model_ts.eval()
            BACKEND = "torchscript"
        else:
            raise FileNotFoundError("TorchScript model not found.")
    except Exception as e:
        raise RuntimeError("No valid model found. Place raf_model.onnx or raf_model_ts.pt in models/rafdb/") from e

# ---- face detector: prefer MTCNN, fallback to OpenCV Haar ----
MTCNN_DETECTOR = None
DETECTOR = "haar"
try:
    import torch as _torch  # local alias to check CUDA availability
    from facenet_pytorch import MTCNN

    mtcnn_device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
    MTCNN_DETECTOR = MTCNN(keep_all=False, device=mtcnn_device)
    DETECTOR = "mtcnn"
except Exception:
    MTCNN_DETECTOR = None
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    DETECTOR = "haar"

# ---- helpers ----
def _preprocess_pil(img: Image.Image) -> np.ndarray:
    """Resize, normalize and convert to C,H,W float32 numpy array."""
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = np.transpose(arr, (2, 0, 1)).astype(np.float32)  # C,H,W
    return arr

def detect_face_pil(img: Image.Image) -> Optional[Image.Image]:
    """Detect a single face and return the cropped PIL Image or None."""
    if DETECTOR == "mtcnn" and MTCNN_DETECTOR is not None:
        boxes, _ = MTCNN_DETECTOR.detect(img)
        if boxes is None or len(boxes) == 0:
            return None
        x1, y1, x2, y2 = map(int, boxes[0])
        # safe crop
        return img.crop((x1, y1, x2, y2))
    else:
        # Haar cascade fallback (uses OpenCV BGR arrays)
        bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) == 0:
            return None
        x, y, w, h = faces[0]
        crop = bgr[y : y + h, x : x + w]
        return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

# ---- core predict functions ----
def predict_image(pil_img: Image.Image) -> Tuple[int, str, Dict[str, float]]:
    """
    Given a PIL.Image (RGB), detect face, preprocess, and return (pred_idx, pred_label, {label:prob}).
    If no face found returns (-1, "no_face", {}).
    """
    face = detect_face_pil(pil_img)
    if face is None:
        return -1, "no_face", {}

    inp = _preprocess_pil(face)[None, :, :, :].astype(np.float32)  # 1,C,H,W

    if BACKEND == "onnx":
        # ONNXRuntime expects NCHW float32
        out = SESSION.run(None, {INPUT_NAME: inp})
        logits = out[0][0]
    else:
        # TorchScript path
        import torch as _torch
        x = _torch.from_numpy(inp)
        with _torch.no_grad():
            logits = model_ts(x).cpu().numpy()[0]

    # stable softmax
    logits = np.asarray(logits, dtype=np.float32)
    probs = np.exp(logits - logits.max())
    probs = probs / probs.sum()
    idx = int(probs.argmax())
    label = idx2label.get(idx, str(idx))
    prob_dict = {idx2label[i]: float(probs[i]) for i in range(len(probs))}
    return idx, label, prob_dict

def predict_frame_bgr(frame_bgr: np.ndarray) -> Tuple[int, str, Dict[str, float]]:
    """Convenience wrapper: accepts OpenCV BGR frame."""
    pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    return predict_image(pil)

# ---- CLI test ----
if __name__ == "__main__":
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else None
    assert p, "Usage: python -m src.rafdb.infer path/to/image.jpg"
    if not Path(p).exists():
        raise FileNotFoundError(p)
    img = Image.open(p).convert("RGB")
    idx, lab, probs = predict_image(img)
    print("Pred:", idx, lab)
    print(probs)
