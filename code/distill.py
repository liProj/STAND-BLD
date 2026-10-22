"""Two-round self-training on frozen global and patch features.

Run the paper configuration with:
    python code/train.py --config configs/stand.json

The paper uses conf=adapt0.6. Initial teacher predictions set the two class
quotas once; each round re-ranks the pool. Other confidence rules are retained
for the reported comparisons. Pool source tags are read only for diagnostics
and for source-composition experiments, never as symptom training labels.
The labeled development set is used for configuration selection. See
docs/PROTOCOL.md for evaluation history and exact normalization behavior.

Advanced interface: python code/distill.py <tag> key=value ...
"""
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import BASE
from methods import ClsHead, MilHead, feat_path, gmm_posterior, nuisance_basis

M, R = "vit_base_patch16_dinov3.lvd1689m", 384
CACHE = f"{BASE}/results/distill/teacher_{{}}.npz"


def teacher_scores(kind, f, g, p, y, sess, k):
    """Teacher logits on the evaluation set and on the pool (cached: two GPU fits)."""
    path = CACHE.format(kind)
    if os.path.exists(path):
        z = np.load(path)
        return z["ev"], z["pool"]
    kk = k if kind == "L3" else 0
    c = ClsHead(kk, 0.1).fit(f["cls"], y, sess)
    ev, po = c.decision(g["cls"]), c.decision(p["cls"])
    if kind == "L3":
        m = MilHead(k=kk).fit(torch.from_numpy(f["patch"]).cuda(), y, sess)
        ev = 0.5 * ev + 0.5 * m.decision(torch.from_numpy(g["patch"]).cuda())
        po = 0.5 * po + 0.5 * m.decision(torch.from_numpy(p["patch"]).cuda())
        del m
        torch.cuda.empty_cache()
    np.savez(path, ev=ev, pool=po)
    return ev, po


def main():
    tag = sys.argv[1]
    a = dict(conf="0.9", rounds="2", src="0", mil="1", nap="1", tstd="1", npool="0",
             share="-1", seed="0", teacher="L3", k="8", C="0.1")
    a.update(dict(x.split("=", 1) for x in sys.argv[2:]))
    out = f"{BASE}/results/preds/distill_{tag}.csv"
    if os.path.exists(out) and not os.environ.get("FORCE"):
        print("exists, skip", out)
        return
    os.makedirs(f"{BASE}/results/distill", exist_ok=True)
    t0 = time.time()
    gr = pd.read_csv(f"{BASE}/data/bld/groups.csv")
    e = pd.read_csv(f"{BASE}/data/inat/eval.csv")
    pl = pd.read_csv(f"{BASE}/data/inat/pool.csv")
    f = np.load(feat_path("bld", M, R), allow_pickle=True)
    g = np.load(feat_path("inat", M, R), allow_pickle=True)
    p = np.load(feat_path("pool", M, R), allow_pickle=True)
    assert (p["files"] == pl.file.values).all() and (g["files"] == e.file.values).all()
    assert not set(pl.user) & set(e.user), "pool and evaluation observers must be disjoint"
    y, sess, k, C = gr.y.values, gr.session.values, int(a["k"]), float(a["C"])
    use_mil = a["mil"] == "1"
    z_ev_t, z_pool_t = teacher_scores(a["teacher"], f, g, p, y, sess, k)

    # ---- which pool images this run may use
    rng = np.random.default_rng(int(a["seed"]))
    idx = np.arange(len(pl))
    n, share = int(a["npool"]), float(a["share"])
    if share >= 0:
        n = n or 1000
        d_idx = idx[(pl.source == "bld_taxon").values]
        b_idx = idx[(pl.source == "beech_taxon").values]
        nd = int(round(share * n))
        idx = np.concatenate([rng.choice(d_idx, min(nd, len(d_idx)), replace=False),
                              rng.choice(b_idx, min(n - nd, len(b_idx)), replace=False)])
    elif n:
        idx = rng.choice(idx, min(n, len(idx)), replace=False)
    idx = np.sort(idx)

    # ---- nuisance basis from the labelled sessions (in source dataset's standardised space)
    Xs = f["cls"].astype(np.float64)
    As = (Xs - Xs.mean(0)) / (Xs.std(0) + 1e-8)
    Vc = nuisance_basis(As, y, sess, k if a["nap"] == "1" else 0)
    Xp, Xe = p["cls"][idx].astype(np.float64), g["cls"].astype(np.float64)
    if use_mil:
        Ps = torch.from_numpy(f["patch"]).cuda()
        Pp = torch.from_numpy(p["patch"][idx]).cuda()
        Pe = torch.from_numpy(g["patch"]).cuda()
        Am = ((Ps.float().mean(1) - Ps.float().mean((0, 1))) / (Ps.float().std((0, 1)) + 1e-6)
              ).cpu().numpy().astype(np.float64)
        Vm = torch.tensor(nuisance_basis(Am, y, sess, k if a["nap"] == "1" else 0),
                          dtype=torch.float32, device="cuda")

    z_pool, z_ev, diag = z_pool_t[idx].copy(), z_ev_t.copy(), []
    for r in range(int(a["rounds"])):
        if a["conf"].startswith("rank"):
            m = int(float(a["conf"][4:]) * len(z_pool))
            o = np.argsort(z_pool)
            pos, neg = o[-m:], o[:m]
        elif a["conf"].startswith("adapt"):
            # composition-aware: the teacher's own calls estimate how much of the pool is
            # symptomatic; the most confident fraction rho of EACH estimated class is labelled
            rho = float(a["conf"][5:])
            pi_hat = float(np.clip((z_pool_t[idx] > 0).mean(), 0.05, 0.95))
            o = np.argsort(z_pool)
            npos = max(10, int(rho * pi_hat * len(z_pool)))
            nneg = max(10, int(rho * (1 - pi_hat) * len(z_pool)))
            pos, neg = o[-npos:], o[:nneg]
        else:
            post = gmm_posterior(z_pool)
            pos = np.flatnonzero(post >= float(a["conf"]))
            neg = np.flatnonzero(post <= 1 - float(a["conf"]))
        sel = np.concatenate([pos, neg])
        ps = np.concatenate([np.ones(len(pos), int), np.zeros(len(neg), int)])
        src_true = (pl.source.values[idx][sel] == "bld_taxon").astype(int)
        diag.append(dict(round=r + 1, n_pool=len(idx), n_pos=len(pos), n_neg=len(neg),
                         teacher_positive_share=float((z_pool_t[idx] > 0).mean()),
                         pseudo_agree_with_source_taxon=float((ps == src_true).mean())))
        if len(pos) < 10 or len(neg) < 10:
            print("too few confident pseudo-labels; keeping the previous model", diag[-1])
            break
        # whole-image student
        if a["src"] == "1":
            Xtr = np.vstack([Xs, Xp[sel]])
            ytr = np.concatenate([y, ps])
            w = np.concatenate([np.ones(len(y)), np.full(len(sel), len(y) / len(sel))])
        else:
            Xtr, ytr, w = Xp[sel], ps, None
        c = ClsHead(0, C)
        if a["tstd"] == "1":
            c.fit(Xtr, ytr, V=Vc, sample_weight=w)
            c.mu, c.sd = Xp.mean(0), Xp.std(0) + 1e-8          # target statistics
            A = (Xtr - np.where(np.arange(len(Xtr))[:, None] < (len(y) if a["src"] == "1" else 0),
                                Xs.mean(0), c.mu)) / np.where(
                np.arange(len(Xtr))[:, None] < (len(y) if a["src"] == "1" else 0),
                Xs.std(0) + 1e-8, c.sd)
            A = A - A @ Vc.T @ Vc
            c.clf.fit(A, ytr, sample_weight=w)                  # each domain in its own units
        else:
            c.fit(Xtr, ytr, V=Vc, sample_weight=w)
            c.mu, c.sd = Xs.mean(0), Xs.std(0) + 1e-8
            A = (Xtr - c.mu) / c.sd
            c.clf.fit(A - A @ Vc.T @ Vc, ytr, sample_weight=w)
        zc_pool, zc_ev = c.decision(Xp), c.decision(Xe)
        z_pool, z_ev = zc_pool, zc_ev
        if use_mil:
            st = torch.from_numpy(sel).cuda()
            Ptr = torch.cat([Ps, Pp[st]]) if a["src"] == "1" else Pp[st]
            mh = MilHead(k=0, seed=int(a["seed"])).fit(Ptr, ytr, V=Vm if len(Vm) else None)
            if a["tstd"] == "1":
                mh.mu, mh.sd = Pp.float().mean((0, 1)), Pp.float().std((0, 1)) + 1e-6
            z_pool = 0.5 * zc_pool + 0.5 * mh.decision(Pp)
            z_ev = 0.5 * zc_ev + 0.5 * mh.decision(Pe)
        from sklearn.metrics import roc_auc_score
        dv = (e.part == "dev").values
        diag[-1].update(dev_auc=float(roc_auc_score(e.y[dv], z_ev[dv])),
                        dev_acc=float(((z_ev[dv] > 0) == (e.y[dv] > 0)).mean()))
        print(tag, diag[-1], f"{time.time() - t0:.0f}s", flush=True)

    if os.environ.get("SAVE_HEADS") and use_mil:
        torch.save(dict(mil_w=mh.w.cpu(), mil_b=mh.b.cpu(), mil_mu=mh.mu.cpu(), mil_sd=mh.sd.cpu(),
                        mil_V=None if mh.V is None else mh.V.cpu(), tau=mh.tau,
                        cls_coef=c.clf.coef_, cls_intercept=c.clf.intercept_, cls_mu=c.mu,
                        cls_sd=c.sd, cls_V=Vc, pool_logit=z_pool, pool_idx=idx),
                   f"{BASE}/results/distill/{tag}_heads.pt")
    pd.DataFrame(dict(protocol="inat", split="inat", file=e.file, y=e.y, part=e.part,
                      logit=z_ev, logit_teacher=z_ev_t, tag=tag)).to_csv(out, index=False)
    json.dump(dict(args=a, rounds=diag,
                   pool_source=pl.source.iloc[idx].value_counts().to_dict()),
              open(f"{BASE}/results/distill/{tag}.json", "w"), indent=1)
    print("DONE", tag, flush=True)


if __name__ == "__main__":
    main()
