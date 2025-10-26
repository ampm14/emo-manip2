import joblib
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ImportWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="Trying to unpickle estimator")

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "goemotions_model"

tfidf = joblib.load(MODEL_DIR / "goemotions_tfidf.joblib")
clf   = joblib.load(MODEL_DIR / "goemotions_clf.joblib")
thrs  = np.load(MODEL_DIR / "goemotions_per_label_thresholds.npy")

emotions = [
    "admiration","amusement","anger","annoyance","approval","caring","confusion",
    "curiosity","desire","disappointment","disapproval","disgust","embarrassment",
    "excitement","fear","gratitude","grief","joy","love","nervousness","optimism",
    "pride","realization","relief","remorse","sadness","surprise","neutral"
]

def predict(text: str, thr_override=None):
    """Predict emotion labels for a given text."""
    X = tfidf.transform([text])
    probas = np.vstack([est.predict_proba(X)[:, 1] for est in clf.estimators_]).T
    thr = thr_override if thr_override is not None else thrs
    preds = (probas >= thr.reshape(1, -1)).astype(int)[0]
    labels = [emotions[i] for i in np.where(preds == 1)[0]]
    probs = {emotions[i]: float(probas[0, i]) for i in range(len(emotions)) if probas[0, i] > 0.05}
    return {"text": text, "labels": labels, "probs": probs}

if __name__ == "__main__":
    import sys
    text = " ".join(sys.argv[1:]) or "I have faith in you"
    print(predict(text))
