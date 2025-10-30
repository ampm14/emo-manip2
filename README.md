

# emo-manip2

Multimodal Emotion Recognition Pipeline (video + audio + text).  
Runs full inference and analytics through a single script on Ubuntu.

---

## Setup

```bash
git clone https://github.com/<yourusername>/emo-manip2.git
cd emo-manip2
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
sudo apt install ffmpeg
chmod +x analyze_video.sh
````

---

## Usage

```bash
./analyze_video.sh data/example.mp4
```

Outputs are stored in:

```
outputs/example/
```

Generated files include:

* `<video>_annotated.mp4` – labeled output video
* `<video>_annotations.json` – per-frame predictions
* `<video>_audio_labels.csv` – audio emotion timeline
* `<video>_summary.json` – overall metrics
* `<video>_timeline.png`, `<video>_heatmap.png`, `<video>_latency.png` – analytics plots
* `<video>_report.md` – summarized report

---

## Models

| Modality | Path                                                 | Dataset    |
| -------- | ---------------------------------------------------- | ---------- |
| Visual   | `models/rafdb/raf_best.pth`                          | RAF-DB     |
| Audio    | `models/ravdess/simple_audio_cnn_aug2_state_dict.pt` | RAVDESS    |
| Text     | `models/goemotions/*`                                | GoEmotions |

---

## Notes

* Requires `ffmpeg` and Python 3.10+.
* Run from project root.
* Outputs saved under `outputs/<video_name>/`.
* GPU recommended; falls back to CPU if unavailable.

---

## Quick Run Example

```bash
source .venv/bin/activate
./analyze_video.sh data/eddie.mp4
```

Results appear in `outputs/eddie/`.

