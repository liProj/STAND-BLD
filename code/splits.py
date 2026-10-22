"""Capture sessions and every split used in the round.

Session = a run of consecutive photographs from one device with no pause longer than 300 s.
Burst   = the same with a 10 s pause (near-duplicate frames of one leaf or twig).

Splits written to data/bld/splits/:
  random_s<seed>.csv   the article's protocol: stratified random 70/15/15
  session_f<k>.csv     5-fold StratifiedGroupKFold over sessions; test = fold k, validation =
                       fold k+1 (also session-disjoint, so early stopping cannot see the test
                       sessions' neighbours), training = the other three folds
  lodo_<group>.csv     leave one capture day x device out (7 groups); validation = a
                       session-disjoint 15% of the rest
"""
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from common import BASE, meta


def groups():
    d = meta()
    s = d.sort_values(["device", "t", "stem"])
    gap = s.groupby("device").t.diff().dt.total_seconds()
    d["session"] = ((gap.isna()) | (gap > 300)).cumsum().reindex(d.index)
    d["burst"] = ((gap.isna()) | (gap > 10)).cumsum().reindex(d.index)
    d["daydev"] = d.t.dt.strftime("%m-%d") + "_" + d.device
    return d


def main():
    d = groups()
    out = f"{BASE}/data/bld/splits"
    os.makedirs(out, exist_ok=True)
    d[["file", "y", "device", "t", "session", "burst", "daydev"]].to_csv(
        f"{BASE}/data/bld/groups.csv", index=False)
    for seed in range(5):
        tr, rest = train_test_split(d, test_size=0.30, stratify=d.y, random_state=seed)
        va, te = train_test_split(rest, test_size=0.50, stratify=rest.y, random_state=seed)
        part = pd.Series("train", index=d.index)
        part[va.index] = "val"
        part[te.index] = "test"
        d.assign(part=part)[["file", "part"]].to_csv(f"{out}/random_s{seed}.csv", index=False)
    fold = np.zeros(len(d), int)
    sgkf = StratifiedGroupKFold(5, shuffle=True, random_state=0)
    for k, (_, te) in enumerate(sgkf.split(d, d.y, d.session)):
        fold[te] = k
    d["fold"] = fold
    d[["file", "fold"]].to_csv(f"{out}/session_folds.csv", index=False)
    for k in range(5):
        part = np.where(fold == k, "test", np.where(fold == (k + 1) % 5, "val", "train"))
        d.assign(part=part)[["file", "part"]].to_csv(f"{out}/session_f{k}.csv", index=False)
    print(pd.crosstab(d.fold, d.label).to_string())
    print(d.groupby("fold").session.nunique().to_dict())
    rng = np.random.default_rng(0)
    for g in sorted(d.daydev.unique()):
        rest = d[d.daydev != g]
        sess = rest.session.unique()
        vs = set(rng.choice(sess, max(2, round(0.2 * len(sess))), replace=False))
        part = np.where(d.daydev == g, "test", np.where(d.session.isin(vs), "val", "train"))
        d.assign(part=part)[["file", "part"]].to_csv(f"{out}/lodo_{g}.csv", index=False)
    print(pd.crosstab(d.daydev, d.label).to_string())
    print("sessions", d.session.nunique(), "bursts", d.burst.nunique())


if __name__ == "__main__":
    main()
