"""Classification heads on frozen features.

Class-conditioned capture-batch mean deviations define a weighted scatter
matrix. Its leading right singular vectors estimate prominent within-class
variation directions. Orthogonal projection removes their feature components.
These directions need not be exclusively non-disease variation; source labels
and capture groups, rather than test labels, define the projection.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from common import BASE


def feat_path(sset, model, res):
    return f"{BASE}/results/feats/{sset}__{model.replace(':', '_').replace('/', '_')}__{res}.npz"


def nuisance_basis(A, y, sess, k, min_n=5):
    """Top-k directions of between-session, within-class scatter. A is standardised (n x d)."""
    if k <= 0:
        return np.zeros((0, A.shape[1]))
    D = []
    for c in (0, 1):
        mc = A[y == c].mean(0)
        for s in np.unique(sess[y == c]):
            i = (sess == s) & (y == c)
            if i.sum() >= min_n:
                D.append(np.sqrt(i.sum()) * (A[i].mean(0) - mc))
    D = np.asarray(D)
    Vt = np.linalg.svd(D, full_matrices=False)[2]
    return Vt[:min(k, len(Vt))]


class ClsHead:
    """Standardise -> project out k session-nuisance directions -> L2 logistic regression."""

    def __init__(self, k=0, C=0.1):
        self.k, self.C = k, C

    def fit(self, X, y, sess=None, V=None, sample_weight=None):
        """V: a nuisance basis estimated elsewhere (the student reuses the teacher's)."""
        X = np.asarray(X, np.float64)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-8
        A = (X - self.mu) / self.sd
        self.V = nuisance_basis(A, y, sess, self.k) if V is None else np.asarray(V)
        A = A - A @ self.V.T @ self.V
        self.clf = LogisticRegression(C=self.C, max_iter=5000, class_weight="balanced").fit(
            A, y, sample_weight=sample_weight)
        return self

    def decision(self, X):
        A = (np.asarray(X, np.float64) - self.mu) / self.sd
        A = A - A @ self.V.T @ self.V
        return self.clf.decision_function(A)


class MilHead:
    """One linear scorer shared by all patch tokens; image logit = log-mean-exp of patch scores.

    Log-mean-exp sits between the mean (every patch must agree) and the max (one patch
    decides): an image is called symptomatic when a sizeable part of the leaf area is, which is
    how banding presents. The patch scores themselves give the localisation maps.
    """

    def __init__(self, k=0, tau=1.0, wd=1e-2, steps=400, lr=3e-3, seed=0, batch=128):
        self.k, self.tau, self.wd, self.steps, self.lr = k, tau, wd, steps, lr
        self.seed, self.batch = seed, batch

    def _z(self, P):
        Z = (P.float() - self.mu) / self.sd
        if self.V is not None:
            Z = Z - (Z @ self.V.T) @ self.V
        return Z

    def patch_scores(self, P):
        return self._z(P) @ self.w + self.b

    def _pool(self, s):
        return (torch.logsumexp(s * self.tau, 1) - np.log(s.shape[1])) / self.tau

    def fit(self, P, y, sess=None, V=None):
        """P: torch tensor (n x T x d) on the GPU (float16 is fine); y, sess: numpy arrays.

        V: a nuisance basis estimated elsewhere (torch, k x d); overrides self.k.
        """
        g = torch.Generator(device="cuda").manual_seed(self.seed)
        self.mu = P.float().mean((0, 1))
        self.sd = P.float().std((0, 1)) + 1e-6
        self.V = V
        if V is None and self.k > 0:
            A = ((P.float().mean(1) - self.mu) / self.sd).cpu().numpy().astype(np.float64)
            self.V = torch.tensor(nuisance_basis(A, y, sess, self.k), dtype=torch.float32,
                                  device="cuda")
        yt = torch.tensor(y, dtype=torch.float32, device="cuda")
        self.w = torch.zeros(P.shape[2], device="cuda", requires_grad=True)
        self.b = torch.zeros(1, device="cuda", requires_grad=True)
        opt = torch.optim.AdamW([self.w, self.b], lr=self.lr, weight_decay=self.wd)
        pw = (1 - yt.mean()) / yt.mean()
        for _ in range(self.steps):
            j = torch.randint(0, len(yt), (self.batch,), device="cuda", generator=g)
            z = self._pool(self.patch_scores(P[j]))
            loss = torch.nn.functional.binary_cross_entropy_with_logits(z, yt[j], pos_weight=pw)
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.w = self.w.detach()
        self.b = self.b.detach()
        return self

    @torch.no_grad()
    def decision(self, P):
        return torch.cat([self._pool(self.patch_scores(P[i:i + 256]))
                          for i in range(0, len(P), 256)]).cpu().numpy()


def gmm_threshold(z, iters=200):
    """Label-free decision threshold: two-component 1-D Gaussian mixture on target logits.

    Returns the point between the component means where the posteriors cross. Uses only
    unlabelled target scores; it makes no assumption about the class prior.
    """
    z = np.asarray(z, float)
    lo, hi = np.percentile(z, [25, 75])
    mu = np.array([lo, hi])
    sd = np.array([z.std(), z.std()]) * 0.5 + 1e-6
    pi = np.array([0.5, 0.5])
    for _ in range(iters):
        r = pi * np.exp(-0.5 * ((z[:, None] - mu) / sd) ** 2) / sd
        r /= r.sum(1, keepdims=True) + 1e-300
        n = r.sum(0)
        mu = (r * z[:, None]).sum(0) / n
        sd = np.sqrt((r * (z[:, None] - mu) ** 2).sum(0) / n) + 1e-6
        pi = n / len(z)
    grid = np.linspace(mu.min(), mu.max(), 2001)
    d = pi * np.exp(-0.5 * ((grid[:, None] - mu) / sd) ** 2) / sd
    lo_c, hi_c = np.argmin(mu), np.argmax(mu)
    return float(grid[np.argmin(np.abs(d[:, lo_c] - d[:, hi_c]))]), dict(mu=mu, sd=sd, pi=pi)


def gmm_posterior(z):
    """P(upper component | z) under the two-component mixture fitted to the scores z."""
    z = np.asarray(z, float)
    _, g = gmm_threshold(z)
    mu, sd, pi = g["mu"], g["sd"], g["pi"]
    d = pi * np.exp(-0.5 * ((z[:, None] - mu) / sd) ** 2) / sd
    return d[:, np.argmax(mu)] / (d.sum(1) + 1e-300)
