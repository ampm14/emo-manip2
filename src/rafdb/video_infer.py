# src/rafdb/video_infer.py
"""
Video inference script for RAF-DB model.
- Uses models/rafdb/raf_model.onnx (repo-relative).
- Forces ONNXRuntime CPU provider and silences GPU discovery noise.
- Uses MTCNN if available (torch.cuda.is_available() to choose device), else OpenCV Haar.
- Writes annotated output video: out_<inputname>.
"""

import os
# Prevent ONNX from probing GPUs (must be set before onnxruntime import)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["ORT_LOG_VERBOSITY_LEVEL"] = "0"

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="onnxruntime")

import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# repo-relative model path
MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "rafdb"
ONNX_PATH = MODEL_DIR / "raf_model.onnx"
if not ONNX_PATH.exists():
    raise FileNotFoundError(f"ONNX model not found at {ONNX_PATH}. Place raf_model.onnx in models/rafdb/")

# try import onnxruntime and create CPU-only session (quiet)
try:
    import onnxruntime as ort
    try:
        ort.set_default_logger_severity(3)
    except Exception:
        pass
    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
except Exception as e:
    raise RuntimeError(f"ONNXRuntime failed to create session: {e}")

# Try MTCNN if available; choose device based on torch.cuda.is_available()
try:
    import torch
    from facenet_pytorch import MTCNN
    mtcnn_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mtcnn = MTCNN(keep_all=True, device=mtcnn_device)  # keep_all True returns all boxes
    face_detector = 'mtcnn'
except Exception:
    mtcnn = None
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    face_detector = 'haar'

# config (match training)
IMG_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SMOOTH_WINDOW = 7         # smoothing window (frames)
FRAME_SKIP = 1            # process every frame; set >1 to skip frames for speed
CONF_THRESHOLD = 0.25     # if top prob < threshold, mark as uncertain
COMMON = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

def preprocess_crop(pil_img):
    img = pil_img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    arr = (arr - MEAN) / STD
    # NHWC -> NCHW
    arr = np.transpose(arr, (2,0,1)).astype(np.float32)
    return arr

def detect_faces_opencv(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    boxes = []
    for (x,y,w,h) in faces:
        boxes.append([x, y, x+w, y+h])
    return boxes

def detect_faces(frame_bgr):
    # returns list of boxes [x1,y1,x2,y2]
    if face_detector == 'mtcnn' and mtcnn is not None:
        img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        boxes, _ = mtcnn.detect(img)
        if boxes is None:
            return []
        # ensure list of lists
        return [[int(b[0]), int(b[1]), int(b[2]), int(b[3])] for b in boxes]
    else:
        return detect_faces_opencv(frame_bgr)

def draw_label(frame, box, text, prob):
    x1,y1,x2,y2 = map(int, box)
    cv2.rectangle(frame, (x1,y1), (x2,y2), (8, 150, 255), 2)
    label = f"{text} {prob:.2f}"
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
    cv2.rectangle(frame, (x1, y1 - t_size[1] - 6), (x1 + t_size[0] + 6, y1), (8,150,255), -1)
    cv2.putText(frame, label, (x1+3, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

def run_on_video(input_path, output_path=None):
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = output_path or ("out_" + Path(input_path).name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w,h))
    frame_idx = 0

    # smoothing buffers per face (simple approach: single global buffer for largest face)
    probs_buf = deque(maxlen=SMOOTH_WINDOW)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % FRAME_SKIP != 0:
            writer.write(frame)
            continue

        boxes = detect_faces(frame)   # list of [x1,y1,x2,y2]
        if len(boxes) == 0:
            # nothing found — optionally flush buffer
            probs_buf.clear()
            writer.write(frame)
            continue

        # pick largest box by area (simple policy)
        areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
        largest = boxes[int(np.argmax(areas))]
        x1,y1,x2,y2 = map(int, largest)
        # pad a little
        pad = int(0.15 * max(x2-x1, y2-y1))
        x1 = max(0, x1-pad); y1 = max(0, y1-pad)
        x2 = min(w, x2+pad); y2 = min(h, y2+pad)
        if x2 <= x1 or y2 <= y1:
            writer.write(frame)
            continue
        face = frame[y1:y2, x1:x2]
        pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        inp = preprocess_crop(pil)[None, :, :, :].astype(np.float32)

        # ONNX inference (CPU session)
        out = sess.run(None, {input_name: inp})
        logits = out[0][0]
        # stable softmax
        logits = np.asarray(logits, dtype=np.float32)
        probs = np.exp(logits - logits.max())
        probs = probs / probs.sum()

        probs_buf.append(probs)
        avg_probs = np.mean(np.stack(list(probs_buf)), axis=0)

        top_idx = int(np.argmax(avg_probs))
        top_prob = float(avg_probs[top_idx])
        label = COMMON[top_idx] if top_prob >= CONF_THRESHOLD else "Uncertain"

        draw_label(frame, (x1,y1,x2,y2), label, top_prob)
        writer.write(frame)

    cap.release()
    writer.release()
    print("Wrote:", out_path)
    return out_path

if __name__ == "__main__":
    import sys
    inp = sys.argv[1] if len(sys.argv)>1 else "input.mp4"
    run_on_video(inp)
