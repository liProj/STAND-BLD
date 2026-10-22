"""Shortcut and leakage audit of source dataset. Writes results/audit.json and results/tables/audit_*.csv.

Every probe here answers one question: how much of the label can be recovered WITHOUT looking
at the leaf, and how much of a random test set is a near-copy of the training set?
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import BASE, meta, metrics
from methods import feat_path


def main():
    os.makedirs(f"{BASE}/results/tables", exist_ok=True)
    d = meta()
    gr = pd.read_csv(f"{BASE}/data/bld/groups.csv")
    assert (gr.file.values == d.file.values).all()
    d["session"], d["burst"], d["daydev"] = gr.session.values, gr.burst.values, gr.daydev.values
    y = d.y.values
    out = dict(n=len(d), n_pos=int(y.sum()), n_sessions=int(d.session.nunique()),
               n_bursts=int(d.burst.nunique()), n_days=int(d.t.dt.date.nunique()))

    # ---- structure tables
    pd.crosstab(d.daydev, d.label).to_csv(f"{BASE}/results/tables/audit_daydev.csv")
    st = d.groupby("session").agg(n=("y", "size"), n_bld=("y", "sum"), device=("device", "first"),
                                  start=("t", "min"), end=("t", "max")).reset_index()
    st["minutes"] = (st.end - st.start).dt.total_seconds() / 60
    st["pure"] = (st.n_bld == 0) | (st.n_bld == st.n)
    st.to_csv(f"{BASE}/results/tables/audit_sessions.csv", index=False)
    out["sessions_single_class"] = int(st.pure.sum())
    out["images_in_single_class_sessions"] = int(st[st.pure].n.sum())
    out["median_gap_s"] = float(d.sort_values(["device", "t"]).groupby("device").t.diff()
                                .dt.total_seconds().median())
    out["device_by_label"] = pd.crosstab(d.label, d.device).to_dict()

    # ---- probes that never see the leaf
    rk = StratifiedKFold(5, shuffle=True, random_state=0)
    gk = StratifiedGroupKFold(5, shuffle=True, random_state=0)
    tsec = (d.t - d.t.min()).dt.total_seconds().values
    exif = pd.DataFrame(dict(
        dev=(d.device == "iPhone").astype(int), w=d.width, h=d.height, ori=d.orientation,
        focal=d.focal, fn=d.fnumber,
        ex=np.log(d.exposure.replace(0, np.nan)).fillna(0),
        iso=pd.to_numeric(d.iso, errors="coerce").fillna(0), bytes=d.bytes))
    thumbs = {}
    for s in (1, 4, 8):
        thumbs[s] = np.stack([np.asarray(Image.open(f"{BASE}/data/bld/img/{f}").convert("RGB")
                                         .resize((s, s), Image.BOX), dtype=np.float32).ravel()
                              for f in d.file])
    probes = {
        "capture time only": pd.DataFrame(dict(t=tsec)),
        "device only": exif[["dev"]],
        "EXIF without time": exif,
        "EXIF + capture time": exif.assign(t=tsec),
        "mean colour (1 pixel)": pd.DataFrame(thumbs[1]),
        "4x4 thumbnail": pd.DataFrame(thumbs[4]),
        "8x8 thumbnail": pd.DataFrame(thumbs[8]),
    }
    rows = []
    for name, Xp in probes.items():
        hg = HistGradientBoostingClassifier(max_iter=200, random_state=0)
        for proto, cv, kw in (("random 5-fold", rk, {}),
                              ("session-grouped 5-fold", gk, dict(groups=d.session))):
            p = cross_val_predict(hg, Xp, y, cv=cv, method="predict_proba", **kw)[:, 1]
            m = metrics(y, p)
            rows.append(dict(probe=name, protocol=proto, acc=m["acc"], auc=m["auc"], f1=m["f1"]))
            print(f"{name:26s} {proto:24s} acc {m['acc']:6.2f} auc {m['auc']:6.2f}", flush=True)
    # image features for reference, same folds
    f = np.load(feat_path("bld", "vit_base_patch16_dinov3.lvd1689m", 384), allow_pickle=True)
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000,
                                                            class_weight="balanced"))
    for proto, cv, kw in (("random 5-fold", rk, {}),
                          ("session-grouped 5-fold", gk, dict(groups=d.session))):
        p = cross_val_predict(lr, f["cls"], y, cv=cv, method="predict_proba", **kw)[:, 1]
        m = metrics(y, p)
        rows.append(dict(probe="DINOv3 image features", protocol=proto, acc=m["acc"],
                         auc=m["auc"], f1=m["f1"]))
        print(f"{'DINOv3 image features':26s} {proto:24s} acc {m['acc']:6.2f}", flush=True)
    pd.DataFrame(rows).to_csv(f"{BASE}/results/tables/audit_probes.csv", index=False)

    # ---- leakage of the article's random split
    leak = []
    X = f["cls"] / np.linalg.norm(f["cls"], axis=1, keepdims=True)
    for seed in range(5):
        sp = pd.read_csv(f"{BASE}/data/bld/splits/random_s{seed}.csv")
        tr, te = (sp.part == "train").values, (sp.part == "test").values
        same_burst = d.burst[te].isin(set(d.burst[tr])).mean()
        same_sess = d.session[te].isin(set(d.session[tr])).mean()
        gaps, nn_lab = [], []
        for dev in d.device.unique():
            a = d[tr & (d.device == dev).values]
            b = d[te & (d.device == dev).values]
            ta = np.sort(a.t.values.astype("int64") // 10**9)
            ya = a.sort_values("t").y.values
            tb = b.t.values.astype("int64") // 10**9
            j = np.clip(np.searchsorted(ta, tb), 1, len(ta) - 1)
            left, right = np.abs(tb - ta[j - 1]), np.abs(ta[j] - tb)
            gaps += list(np.minimum(left, right))
            nn_lab += list((np.where(left <= right, ya[j - 1], ya[j]) == b.y.values))
        S = X[te] @ X[tr].T
        nn = S.argmax(1)
        leak.append(dict(seed=seed, test_in_train_burst=100 * same_burst,
                         test_in_train_session=100 * same_sess,
                         median_gap_to_train_s=float(np.median(gaps)),
                         share_within_10s=100 * float(np.mean(np.array(gaps) <= 10)),
                         time_neighbour_acc=100 * float(np.mean(nn_lab)),
                         feature_1nn_acc=100 * float((y[tr][nn] == y[te]).mean()),
                         feature_1nn_same_session=100 * float(
                             (d.session.values[tr][nn] == d.session.values[te]).mean())))
    lk = pd.DataFrame(leak)
    lk.to_csv(f"{BASE}/results/tables/audit_leak.csv", index=False)
    print(lk.round(2).to_string())
    out["leak_mean"] = lk.drop(columns="seed").mean().round(3).to_dict()
    out["probes"] = rows
    json.dump(out, open(f"{BASE}/results/audit.json", "w"), indent=1, default=str)
    print("DONE audit", flush=True)


if __name__ == "__main__":
    main()
