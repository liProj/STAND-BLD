"""Shared paths, metadata, splits and metrics for the beech-leaf-disease round."""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from pathlib import Path

BASE = str(Path(os.environ.get("STAND_ROOT", Path(__file__).resolve().parents[1])).expanduser().resolve())

ARCHS = ["efficientnet_v2_s", "inception_v3", "mobilenet_v3_large", "resnet50"]
NICE = {"efficientnet_v2_s": "EfficientNetV2-S", "inception_v3": "InceptionV3",
        "mobilenet_v3_large": "MobileNetV3-L", "resnet50": "ResNet50"}


def meta():
    """source dataset metadata with y (1 = BLD), parsed capture time and the device family."""
    d = pd.read_csv(f"{BASE}/data/bld/meta.csv")
    d["y"] = (d.label == "BLD").astype(int)
    d["t"] = pd.to_datetime(d.datetime, format="%Y:%m:%d %H:%M:%S", errors="coerce")
    d["device"] = np.where(d.model.str.contains("iPhone", na=False), "iPhone",
                           np.where(d.model.str.contains("NIKON", na=False), "Nikon", "other"))
    return d


def metrics(y, p, thr=0.5):
    """Accuracy, precision, recall, F1 (BLD positive) and ROC-AUC, all in percent."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    yh = (p >= thr).astype(int)
    tp = int(((yh == 1) & (y == 1)).sum())
    fp = int(((yh == 1) & (y == 0)).sum())
    fn = int(((yh == 0) & (y == 1)).sum())
    tn = int(((yh == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    auc = roc_auc_score(y, p) if 0 < y.sum() < len(y) else float("nan")
    return dict(n=len(y), acc=100 * (tp + tn) / len(y), prec=100 * prec, rec=100 * rec,
                f1=100 * f1, auc=100 * auc, tp=tp, fp=fp, fn=fn, tn=tn,
                bacc=100 * 0.5 * (rec + (tn / (tn + fp) if tn + fp else 0.0)))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def boot_ci(y, p, fn, B=2000, seed=0, groups=None):
    """Percentile bootstrap interval of a metric; resamples whole groups when given."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    out = []
    if groups is None:
        for _ in range(B):
            i = rng.integers(0, len(y), len(y))
            out.append(fn(y[i], p[i]))
    else:
        g = pd.factorize(np.asarray(groups))[0]
        idx = [np.flatnonzero(g == k) for k in range(g.max() + 1)]
        for _ in range(B):
            pick = rng.integers(0, len(idx), len(idx))
            i = np.concatenate([idx[k] for k in pick])
            out.append(fn(y[i], p[i]))
    return np.nanpercentile(out, [2.5, 97.5])
