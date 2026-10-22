"""Source-only linear probes of cached DINOv2 and SigLIP 2 image features.

Both candidates are retained irrespective of their external scores. The classifier
and regularization selection are shared. Target labels never select parameters.
This is a retrospective addition to the existing study, not a new untouched test.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import hashlib
import json
import platform
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from common import BASE
ROOT = Path(BASE)
OUT = ROOT / 'results/recent_baselines_20261022'
MODELS = [
    dict(tag='dinov2_b14', name='DINOv2-B/14', year=2024, key='oquab2024',
         cache='vit_base_patch14_dinov2.lvd142m__392', feature='cls', resolution=392,
         paper='https://arxiv.org/abs/2304.07193', publication='TMLR 2024',
         official='https://github.com/facebookresearch/dinov2'),
    dict(tag='siglip2_b16', name='SigLIP 2-B/16', year=2025, key='tschannen2025',
         cache='siglip2-base-patch16-224', feature='emb', resolution=224,
         paper='https://arxiv.org/abs/2502.14786', publication='arXiv preprint, 2025',
         official='https://huggingface.co/google/siglip2-base-patch16-224'),
]
CGRID = [0.001, 0.01, 0.1, 1.0, 10.0]

def digest(data):
    return hashlib.sha256(data).hexdigest()

def fit(X, y, C):
    mu, sd = X.mean(0), X.std(0) + 1e-8
    clf = LogisticRegression(C=C, class_weight='balanced', max_iter=5000,
                             solver='lbfgs', random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        clf.fit((X - mu) / sd, y)
    return clf, mu, sd

def metrics(y, z):
    pred = z > 0
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(acc=100*accuracy_score(y, pred), precision=100*precision_score(y, pred),
                recall=100*recall_score(y, pred), f1=100*f1_score(y, pred),
                auc=100*roc_auc_score(y, z), n=len(y), TP=int(tp), FP=int(fp), FN=int(fn), TN=int(tn))

def main():
    OUT.mkdir(exist_ok=False)
    plan = dict(created_utc=datetime.now(timezone.utc).isoformat(), models=MODELS,
                include_all_candidates=True, source='data/bld/groups.csv', source_n=1934,
                external='data/inat/eval.csv', external_test_n=1856,
                feature_normalization='source-train-only per-dimension population SD + 1e-8',
                classifier='class-balanced L2 logistic regression', C_grid=CGRID,
                selection='pooled source-only fivefold StratifiedGroupKFold AUC, shuffle=True, seed=0; ties to 4 decimals choose smaller C',
                decision_threshold='logit > 0', nuisance_projection=False, target_adaptation=False,
                context='retrospective new baselines added to an already inspected test set',
                sklearn=sklearn.__version__, numpy=np.__version__, python=platform.python_version())
    (OUT/'protocol.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
    gr = pd.read_csv(ROOT/'data/bld/groups.csv')
    ev = pd.read_csv(ROOT/'data/inat/eval.csv')
    assert len(gr) == 1934 and (ev.part == 'test').sum() == 1856
    assert not set(ev.loc[ev.part == 'dev', 'user']) & set(ev.loc[ev.part == 'test', 'user'])
    y = gr.y.to_numpy()
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(gr, y, gr.session))
    for tr, va in folds:
        assert set(gr.session.iloc[tr]).isdisjoint(set(gr.session.iloc[va]))
        assert len(np.unique(y[tr])) == len(np.unique(y[va])) == 2
    split_rows = []
    for fold_id, (_, va) in enumerate(folds):
        split_rows.extend(dict(file=gr.file.iloc[i], fold=fold_id) for i in va)
    pd.DataFrame(split_rows).to_csv(OUT/'source_cv_folds.csv', index=False)
    all_rows, records = [], []
    for model in MODELS:
        t0 = time.time()
        with np.load(ROOT/f'results/feats/bld__{model["cache"]}.npz', allow_pickle=True) as f:
            assert np.array_equal(f['files'].astype(str), gr.file.to_numpy().astype(str))
            X = f[model['feature']].astype(np.float64)
        with np.load(ROOT/f'results/feats/inat__{model["cache"]}.npz', allow_pickle=True) as f:
            assert np.array_equal(f['files'].astype(str), ev.file.to_numpy().astype(str))
            Xe = f[model['feature']].astype(np.float64)
        assert np.isfinite(X).all() and np.isfinite(Xe).all()
        assert X.shape[1] == Xe.shape[1]
        scores, cv_preds = {}, {}
        for C in CGRID:
            z = np.zeros(len(y))
            for tr, va in folds:
                clf, mu, sd = fit(X[tr], y[tr], C)
                z[va] = clf.decision_function((X[va]-mu)/sd)
            scores[C] = float(roc_auc_score(y, z))
            cv_preds[C] = z
            print(model['name'], 'C', C, 'source-CV AUC', round(100*scores[C], 4), flush=True)
        selected = max(CGRID, key=lambda C: (round(scores[C], 4), -C))
        clf, mu, sd = fit(X, y, selected)
        z = clf.decision_function((Xe-mu)/sd)
        pred = ev[['file', 'y', 'part']].copy()
        pred['protocol'] = 'inat'
        pred['logit'] = z
        pred['model'] = model['tag']
        pred.to_csv(OUT/f'{model["tag"]}_predictions.csv', index=False)
        np.savez(OUT/f'{model["tag"]}_linear_head.npz', mu=mu, sd=sd,
                 coef=clf.coef_, intercept=clf.intercept_, C=selected)
        pd.DataFrame(dict(file=gr.file, y=y, logit=cv_preds[selected])).to_csv(
            OUT/f'{model["tag"]}_source_oof.csv', index=False)
        result = dict(**model, selected_C=selected, source_cv_auc=scores,
                      source_feature_sha256=digest(X.tobytes()),
                      external_feature_sha256=digest(Xe.tobytes()),
                      source_file_sha256=digest((ROOT/'data/bld/groups.csv').read_bytes()),
                      eval_file_sha256=digest((ROOT/'data/inat/eval.csv').read_bytes()),
                      fit_seconds=time.time()-t0, iterations=clf.n_iter_.tolist(), parts={})
        for part in ('dev', 'test'):
            keep = (ev.part == part).to_numpy()
            result['parts'][part] = metrics(ev.y.to_numpy()[keep], z[keep])
            all_rows.append(dict(model=model['name'], tag=model['tag'], year=model['year'],
                                 part=part, C=selected, **result['parts'][part]))
        records.append(result)
        print(json.dumps(dict(model=model['name'], selected_C=selected,
                              test=result['parts']['test']), ensure_ascii=False), flush=True)
    (OUT/'results.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    pd.DataFrame(all_rows).to_csv(OUT/'metrics.csv', index=False)
    (OUT/'execution.json').write_text(json.dumps(dict(
        script_sha256=digest(Path(__file__).read_bytes()), completed_utc=datetime.now(timezone.utc).isoformat(),
        all_models_reported=True), indent=2), encoding='utf-8')

if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
