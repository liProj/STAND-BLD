"""Transductive version of the self-distillation on source dataset's held-out capture sessions.

For each session-grouped fold the teacher is the model already fitted on the other folds
(its logits on the held-out fold are read from results/preds/ours_L3.csv). The held-out
photographs, without their labels, play the role of the unlabelled pool: the top and bottom 30%
of teacher scores become pseudo-labels, a student (whole-image head + patch head) is fitted on
them with the fold's own statistics, and scores the same fold. This is the situation of a
surveyor who photographs a new stand and lets the model adapt to that day's pictures.

Output results/preds/ours_L4trans.csv (protocol = session).
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import BASE
from methods import ClsHead, MilHead, feat_path, nuisance_basis

M, R, FRAC, ROUNDS = "vit_base_patch16_dinov3.lvd1689m", 384, 0.3, 2


def main():
    gr = pd.read_csv(f"{BASE}/data/bld/groups.csv")
    f = np.load(feat_path("bld", M, R), allow_pickle=True)
    t = pd.read_csv(f"{BASE}/results/preds/ours_L3.csv")
    t = t[t.protocol == "session"]
    X, y, sess = f["cls"].astype(np.float64), gr.y.values, gr.session.values
    P = torch.from_numpy(f["patch"]).cuda()
    pos = {fn: i for i, fn in enumerate(gr.file)}
    rows = []
    for split, g in t.groupby("split"):
        te = np.array([pos[fn] for fn in g.file])
        tr = np.setdiff1d(np.arange(len(y)), te)
        k = int(g.k.iloc[0])
        A = (X[tr] - X[tr].mean(0)) / (X[tr].std(0) + 1e-8)
        Vc = nuisance_basis(A, y[tr], sess[tr], k)
        Ptr = P[torch.from_numpy(tr).cuda()].float()
        Am = ((Ptr.mean(1) - Ptr.mean((0, 1))) / (Ptr.std((0, 1)) + 1e-6)).cpu().numpy()
        Vm = torch.tensor(nuisance_basis(Am.astype(np.float64), y[tr], sess[tr], k),
                          dtype=torch.float32, device="cuda")
        del Ptr
        Pt = P[torch.from_numpy(te).cuda()]
        z = g.logit.values.copy()
        for _ in range(ROUNDS):
            m = int(FRAC * len(z))
            o = np.argsort(z)
            sel = np.concatenate([o[-m:], o[:m]])
            ps = np.concatenate([np.ones(m, int), np.zeros(m, int)])
            c = ClsHead(0, 0.1).fit(X[te][sel], ps, V=Vc)
            c.mu, c.sd = X[te].mean(0), X[te].std(0) + 1e-8
            Af = (X[te][sel] - c.mu) / c.sd
            c.clf.fit(Af - Af @ Vc.T @ Vc, ps)
            mh = MilHead(k=0).fit(Pt[torch.from_numpy(sel).cuda()], ps,
                                  V=Vm if len(Vm) else None)
            mh.mu, mh.sd = Pt.float().mean((0, 1)), Pt.float().std((0, 1)) + 1e-6
            z = 0.5 * c.decision(X[te]) + 0.5 * mh.decision(Pt)
        acc0 = 100 * ((g.logit.values > 0) == (g.y.values > 0)).mean()
        acc1 = 100 * ((z > 0) == (g.y.values > 0)).mean()
        print(f"{split}: teacher acc {acc0:.2f} -> transductive student acc {acc1:.2f}", flush=True)
        rows.append(pd.DataFrame(dict(protocol="session", split=split, file=g.file.values,
                                      y=g.y.values, logit=z, logit_teacher=g.logit.values, k=k)))
    pd.concat(rows).assign(tag="L4trans").to_csv(f"{BASE}/results/preds/ours_L4trans.csv",
                                                 index=False)
    print("DONE L4trans", flush=True)


if __name__ == "__main__":
    main()
