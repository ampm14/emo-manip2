# src/rafdb/video_summary.py
"""
Video summary tool for RAF-DB model.
- Reuses src.rafdb.infer.predict_frame_bgr for per-frame emotion probs.
- Writes: out_<videoname>.mp4 (annotated), per_frame_probs_<videoname>.csv, summary_<videoname>.png
- Prints one-line summary: predominant emotion and average probability.
"""

import os
# reduce ONNX probing noise if any import triggers it
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["ORT_LOG_VERBOSITY_LEVEL"] = "0"

import csv
import sys
from pathlib import Path
from collections import deque
import numpy as np
import cv2
from PIL import Image

# plotting
import matplotlib.pyplot as plt

# import your inference helper (must be reachable as module)
from src.rafdb.infer import predict_frame_bgr, idx2label, predict_image

# Build canonical COMMON label list from idx2label mapping (ensures consistent ordering)
COMMON = [idx2label[i] for i in sorted(idx2label.keys())]

# small drawing util (keeps style same as video_infer)
def draw_label_on_frame(frame, box, text, prob):
    x1,y1,x2,y2 = map(int, box)
    cv2.rectangle(frame, (x1,y1), (x2,y2), (8, 150, 255), 2)
    label = f"{text} {prob:.2f}"
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
    cv2.rectangle(frame, (x1, y1 - t_size[1] - 6), (x1 + t_size[0] + 6, y1), (8,150,255), -1)
    cv2.putText(frame, label, (x1+3, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

def extract_face_box_from_predict(frame, predict_fn):
    """
    Uses the model's detection path by running predict on the cropped PIL image.
    We need the face box for overlay; fallback: run a quick OpenCV detector here.
    This function returns (box, idx, label, probs) where box is [x1,y1,x2,y2] or None.
    """
    try:
        idx, label, probs = predict_fn(frame)
        if idx == -1:
            return None, idx, label, probs
        # use Haar to get simple box for overlay
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) == 0:
            return None, idx, label, probs
        x,y,w,h = faces[0]
        return [x, y, x+w, y+h], idx, label, probs
    except Exception:
        return None, -1, "error", {}

def run_and_summarize(input_path, output_path=None, frame_skip=1, smooth_k=7, conf_threshold=0.15):
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_name = Path(input_path).name
    out_path = output_path or ("out_" + video_name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w,h))

    frame_idx = 0
    probs_buf = deque(maxlen=smooth_k)

    # storage for csv: frame_idx, detected_flag, probs...
    csv_rows = []
    frames_with_face = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % frame_skip != 0:
            writer.write(frame)
            continue

        idx, label, prob_dict = predict_frame_bgr(frame)
        if idx == -1:
            csv_rows.append([frame_idx, 0] + [0.0]*len(COMMON))
            writer.write(frame)
            continue

        frames_with_face += 1
        probs = np.array([prob_dict.get(c, 0.0) for c in COMMON], dtype=np.float32)
        probs_buf.append(probs)
        avg_probs = np.mean(np.stack(list(probs_buf)), axis=0)

        top_idx = int(np.argmax(avg_probs))
        top_prob = float(avg_probs[top_idx])
        top_label = COMMON[top_idx] if top_prob >= conf_threshold else "Uncertain"

        box, _, _, _ = extract_face_box_from_predict(frame, predict_frame_bgr)
        if box is not None:
            draw_label_on_frame(frame, box, top_label, top_prob)

        csv_rows.append([frame_idx, 1] + [float(x) for x in avg_probs.tolist()])
        writer.write(frame)

    cap.release()
    writer.release()

    # write CSV
    csv_path = Path(f"per_frame_probs_{video_name}.csv")
    header = ["frame_idx", "has_face"] + COMMON
    with open(csv_path, "w", newline="") as cf:
        writer_csv = csv.writer(cf)
        writer_csv.writerow(header)
        writer_csv.writerows(csv_rows)

    # compute global averages across frames that contained faces
    arr = np.array([row[2:] for row in csv_rows if row[1] == 1], dtype=np.float32)
    if arr.size == 0:
        avg = np.zeros(len(COMMON), dtype=np.float32)
    else:
        avg = arr.mean(axis=0)

    # save summary bar chart
    summary_png = Path(f"summary_{video_name}.png")
    plt.figure(figsize=(8,4))
    ax = plt.gca()
    x = np.arange(len(COMMON))
    ax.bar(x, avg)
    ax.set_xticks(x)
    ax.set_xticklabels(COMMON, rotation=30, ha='right')
    ax.set_ylim(0,1)
    ax.set_ylabel("Average probability")
    ax.set_title(f"Video emotion summary — {video_name}")
    plt.tight_layout()
    plt.savefig(summary_png, dpi=150)
    plt.close()

    # determine predominant emotion
    pred_idx = int(np.argmax(avg))
    pred_label = COMMON[pred_idx]
    pred_prob = float(avg[pred_idx])

    one_liner = f"Video summary: Predominant emotion — {pred_label} (avg p = {pred_prob:.2f}), frames_with_face = {frames_with_face}"
    print(one_liner)

    return {
        "out_video": out_path,
        "csv": str(csv_path),
        "summary_png": str(summary_png),
        "one_liner": one_liner
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.rafdb.video_summary path/to/video.mp4")
        sys.exit(1)
    inp = sys.argv[1]
    res = run_and_summarize(inp)
    print("Artifacts written:", res)
