"""Frozen-feature heads under the four evaluation protocols.

    python run_ours.py <tag> key=value ...

keys   model, res        backbone features to use (default DINOv3 ViT-B/16 at 384)
       feat              cls | avg   (which pooled token the whole-image head reads)
       k                 nuisance directions: integer, or "auto" = chosen by session-grouped
                         cross-validation inside the training data only
       C                 inverse L2 strength of the logistic head
       mil               0 | 1  add the patch multiple-instance head (needs patch features)
       tau, pool         log-mean-exp temperature / pooling rule of the patch head
       protocols         comma list from random,session,lodo,inat (default all)
Every protocol fits on exactly the images the reference CNN could see (train + validation
parts) and predicts the test part; "inat" fits on all of source dataset and predicts the external
set. Output: results/preds/ours_<tag>.csv with one row per (protocol, split, image).
"""
import glob
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import BASE
from methods import ClsHead, feat_path

KGRID = [0, 2, 4, 8, 12, 16]


def select_k(X, y, sess, C, seed=0):
    """Pooled session-grouped CV AUC inside the training data, for each k in KGRID."""
    ng = len(np.unique(sess))
    cv = StratifiedGroupKFold(min(5, ng), shuffle=True, random_state=seed)
    folds = list(cv.split(X, y, sess))
    best, scores = 0, {}
    for k in KGRID:
        p = np.zeros(len(y))
        for tr, te in folds:
            if len(np.unique(y[tr])) < 2:
                continue
            p[te] = ClsHead(k, C).fit(X[tr], y[tr], sess[tr]).decision(X[te])
        scores[k] = roc_auc_score(y, p)
    best = max(scores, key=lambda k: (round(scores[k], 4), -k))
    return best, scores


def main():
    tag = sys.argv[1]
    a = dict(model="vit_base_patch16_dinov3.lvd1689m", res="384", feat="cls", k="auto", C="0.1",
             mil="0", tau="1.0", pool="lme", protocols="random,session,lodo,inat", wmil="0.5",
             steps="400", ntrain="0", nsess="0", seed="0")
    a.update(dict(x.split("=", 1) for x in sys.argv[2:]))
    out = f"{BASE}/results/preds/ours_{tag}.csv"
    if os.path.exists(out) and not os.environ.get("FORCE"):
        print("exists, skip", out)
        return
    C, mil = float(a["C"]), a["mil"] == "1"
    gr = pd.read_csv(f"{BASE}/data/bld/groups.csv")
    e = pd.read_csv(f"{BASE}/data/inat/eval.csv")
    f = np.load(feat_path("bld", a["model"], a["res"]), allow_pickle=True)
    g = np.load(feat_path("inat", a["model"], a["res"]), allow_pickle=True)
    assert (f["files"] == gr.file.values).all() and (g["files"] == e.file.values).all()
    X, Xe = f[a["feat"]].astype(np.float64), g[a["feat"]].astype(np.float64)
    y, sess = gr.y.values, gr.session.values
    if mil:
        import torch
        from methods import MilHead
        P = torch.from_numpy(f["patch"]).cuda()
        Pe = torch.from_numpy(g["patch"]).cuda()

    jobs = []                                   # (protocol, split, train idx, test idx or None)
    want = a["protocols"].split(",")
    sp_dir = f"{BASE}/data/bld/splits"
    pats = {"random": "random_s*.csv", "session": "session_f[0-9].csv", "lodo": "lodo_*.csv"}
    for proto in ("random", "session", "lodo"):
        if proto in want:
            for p in sorted(glob.glob(f"{sp_dir}/{pats[proto]}")):
                s = pd.read_csv(p)
                assert (s.file.values == gr.file.values).all()
                jobs.append((proto, os.path.basename(p)[:-4], np.flatnonzero(s.part != "test"),
                             np.flatnonzero(s.part == "test")))
    if "inat" in want:
        jobs.append(("inat", "inat", np.arange(len(y)), None))

    rng = np.random.default_rng(int(a["seed"]))
    rows, t0 = [], time.time()
    for proto, split, tr, te in jobs:
        if int(a["nsess"]) > 0:                 # session-count curve: keep n sessions per class
            keep = []
            for c in (0, 1):
                ss = np.unique(sess[tr][y[tr] == c])
                keep += list(rng.choice(ss, min(int(a["nsess"]), len(ss)), replace=False))
            tr = tr[np.isin(sess[tr], keep)]
        if int(a["ntrain"]) > 0 and int(a["ntrain"]) < len(tr):
            tr = rng.choice(tr, int(a["ntrain"]), replace=False)
        ytr, str_ = y[tr], sess[tr]
        if a["k"] == "auto":
            k, sc = select_k(X[tr], ytr, str_, C)
        else:
            k, sc = int(a["k"]), {}
        h = ClsHead(k, C).fit(X[tr], ytr, str_)
        Xt = Xe if te is None else X[te]
        zc = h.decision(Xt)
        zm = np.full(len(zc), np.nan)
        if mil:
            mh = MilHead(k=k, tau=float(a["tau"]), steps=int(a["steps"]),
                         seed=int(a["seed"])).fit(P[torch.from_numpy(tr).cuda()], ytr, str_)
            if a["pool"] != "lme":
                mh._pool = {"mean": lambda s: s.mean(1), "max": lambda s: s.max(1).values,
                            "top10": lambda s: s.topk(max(1, s.shape[1] // 10), 1).values.mean(1)
                            }[a["pool"]]
                mh.fit(P[torch.from_numpy(tr).cuda()], ytr, str_)
            zm = mh.decision(Pe if te is None else P[torch.from_numpy(te).cuda()])
        w = float(a["wmil"]) if mil else 0.0
        z = (1 - w) * zc + w * np.nan_to_num(zm)
        files = e.file.values if te is None else gr.file.values[te]
        yy = e.y.values if te is None else y[te]
        rows.append(pd.DataFrame(dict(protocol=proto, split=split, file=files, y=yy, logit=z,
                                      logit_cls=zc, logit_mil=zm, k=k, n_train=len(tr))))
        print(f"{tag} {proto} {split} k={k} n_train={len(tr)} "
              f"acc={100 * ((z > 0) == (yy > 0)).mean():.2f} "
              f"auc={100 * roc_auc_score(yy, z) if 0 < yy.sum() < len(yy) else float('nan'):.2f} "
              f"{time.time() - t0:.0f}s " + " ".join(f"{kk}:{v:.3f}" for kk, v in sc.items()),
              flush=True)
    pd.concat(rows).assign(tag=tag).to_csv(out, index=False)
    print("DONE", tag, flush=True)


if __name__ == "__main__":
    main()
