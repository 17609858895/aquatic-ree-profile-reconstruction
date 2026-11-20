"""LC-HG: Lanthanide-Contraction-informed Hierarchical Gaussian model for REE pattern completion.

Model (per sample i, y = log10 PAAS-normalised concentrations of the 14 REE, partially observed):

    y_i = r0 + B (gamma(x_i) + theta_i) + eps_i,   theta_i ~ N(0, Sigma),   eps_i ~ N(0, diag(s2))

B (14 x K) is fixed by physics: orthonormal polynomials of the REE3+ ionic radius (smooth
lanthanide-contraction shape; O'Neill 2016 lambda idea) plus unit columns for redox/anthropogenic
anomaly-prone elements (Ce, Eu, Gd). gamma(x) is a metadata-conditioned prior mean in coefficient
space learnt by gradient boosting inside an EM loop; study-grouped out-of-fold predictions are used
in the E-step so that Sigma reflects between-study uncertainty. Inference for any observed subset
is closed-form Gaussian conditioning, giving a predictive mean and variance for every element.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from ree_common import REE, basis_matrix


def grouped_folds(groups, n_splits, seed=0):
    """Size-balanced random assignment of whole groups (studies) to folds."""
    rng = np.random.default_rng(seed)
    g = pd.Series(groups)
    sizes = g.value_counts()
    order = sizes.index.to_numpy().copy()
    rng.shuffle(order)
    order = sorted(order, key=lambda k: -sizes[k])  # big studies first, ties already shuffled
    load = np.zeros(n_splits)
    assign = {}
    for k in order:
        j = int(np.argmin(load + rng.uniform(0, 1e-6, n_splits)))
        assign[k] = j
        load[j] += sizes[k]
    fold = g.map(assign).to_numpy()
    return [(np.where(fold != j)[0], np.where(fold == j)[0]) for j in range(n_splits)]


def _lgb(seed=0):
    return lgb.LGBMRegressor(n_estimators=200, learning_rate=0.03, num_leaves=8, min_child_samples=15,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, cat_smooth=10,
                             reg_lambda=1.0, verbose=-1, n_jobs=1, random_state=seed)


class LCHG:
    def __init__(self, degree=3, anomalies=("Ce", "Eu", "Gd"), use_meta=True, basis="physics",
                 n_em=4, inner_folds=5, seed=0, n_factors=None, nu=None, tetrad=False):
        self.tetrad = tetrad
        self.nu = nu   # None: Gaussian; float: multivariate-t scale mixture (robust, coherence-adaptive variance)
        self.degree, self.anomalies, self.use_meta = degree, tuple(anomalies), use_meta
        self.basis, self.n_em, self.inner_folds, self.seed = basis, n_em, inner_folds, seed
        self.n_factors = n_factors

    # ---------- Gaussian conditioning for arbitrary observed subsets ----------
    def _posterior(self, Y, G):
        """Posterior mean M (n x K) and covariance V (n x K x K) of theta given observed entries of Y."""
        n, K = Y.shape[0], self.B.shape[1]
        M = np.zeros((n, K)); V = np.zeros((n, K, K))
        self._d2 = np.zeros(n); self._no = np.zeros(n)
        obs = ~np.isnan(Y)
        Sinv = np.linalg.inv(self.Sigma)
        keys = obs @ (1 << np.arange(14))
        for key in np.unique(keys):
            rows = np.where(keys == key)[0]
            o = obs[rows[0]]
            if not o.any():
                V[rows] = self.Sigma
                continue
            Bo = self.B[o]; Dinv = 1.0 / self.s2[o]
            Vk = np.linalg.inv(Sinv + (Bo.T * Dinv) @ Bo)
            R = Y[rows][:, o] - self.r0[o] - G[rows] @ Bo.T
            M[rows] = (R * Dinv) @ Bo @ Vk
            V[rows] = Vk
            Om = Bo @ self.Sigma @ Bo.T + np.diag(self.s2[o])
            self._d2[rows] = np.einsum("ni,ij,nj->n", R, np.linalg.inv(Om), R)
            self._no[rows] = o.sum()
        return M, V

    def _weights(self):
        if self.nu is None:
            return np.ones(len(self._d2))
        return (self.nu + self._no) / (self.nu + self._d2)

    def _fit_gamma(self, X, T, groups, oof=True):
        """Boosted regression of coefficient targets T on metadata X; returns (oof_pred, models)."""
        K = T.shape[1]
        models, P = [], np.zeros_like(T)
        folds = grouped_folds(groups, self.inner_folds, self.seed + 17) if oof else []
        for k in range(K):
            if oof:
                for tr, te in folds:
                    P[te, k] = _lgb(self.seed).fit(X.iloc[tr], T[tr, k]).predict(X.iloc[te])
            models.append(_lgb(self.seed).fit(X, T[:, k]))
        return P, models

    def fit(self, Y, X=None, groups=None):
        Y = np.asarray(Y, float); n = len(Y); obs = ~np.isnan(Y)
        if self.basis == "physics":
            self.B, self.cols = basis_matrix(self.anomalies, self.degree, tetrad=self.tetrad)
        elif self.basis == "learned":      # same rank, loadings learnt by EM (probabilistic factor model)
            K = self.n_factors or (self.degree + 1 + len(self.anomalies))
            rng = np.random.default_rng(self.seed)
            self.B = np.linalg.qr(rng.normal(size=(14, K)))[0]; self.cols = [f"f{k}" for k in range(K)]
        K = self.B.shape[1]
        self.r0 = np.zeros(14)
        mu = np.nanmean(Y, 0)
        g0 = np.zeros(K) if self.basis == "learned" else np.linalg.lstsq(self.B, mu, rcond=None)[0]
        self.g0 = g0
        self.r0 = mu - self.B @ g0
        G = np.tile(g0, (n, 1))
        self.Sigma = np.eye(K); self.s2 = np.full(14, 0.05)
        self.models = None
        for it in range(self.n_em):
            for _ in range(6):   # inner EM for (Sigma, s2, r0[, B]) at fixed gamma
                M, V = self._posterior(Y, G)
                w = self._weights()
                self.Sigma = ((M * w[:, None]).T @ M + V.sum(0)) / n + 1e-8 * np.eye(K)
                Cf = G + M
                if self.basis == "learned":   # M-step for loadings, row by row of B
                    S2 = np.einsum("ni,nj->nij", Cf, Cf) + V
                    for e in range(14):
                        r = obs[:, e]
                        self.B[e] = np.linalg.solve(S2[r].sum(0) + 1e-8 * np.eye(K),
                                                    ((Y[r, e] - self.r0[e])[:, None] * Cf[r]).sum(0))
                Fit = Cf @ self.B.T + self.r0
                Res = Y - Fit
                bvb = np.einsum("ek,nkl,el->ne", self.B, V, self.B)
                for e in range(14):
                    r = obs[:, e]
                    self.s2[e] = max(np.mean(w[r] * Res[r, e] ** 2 + bvb[r, e]), 1e-5)
                    self.r0[e] += 0.5 * np.sum(w[r] * Res[r, e]) / np.sum(w[r])
            if not self.use_meta or X is None:
                break
            T = G + M                                   # current posterior mean of total coefficients
            P, self.models = self._fit_gamma(X, T, groups, oof=True)
            G = P                                       # study-out-of-fold prior mean for the next E-step
        if self.use_meta and X is not None:
            for _ in range(6):                          # final variance update at OOF prior
                M, V = self._posterior(Y, G)
                w = self._weights()
                self.Sigma = ((M * w[:, None]).T @ M + V.sum(0)) / n + 1e-8 * np.eye(K)
                Res = Y - ((G + M) @ self.B.T + self.r0)
                bvb = np.einsum("ek,nkl,el->ne", self.B, V, self.B)
                for e in range(14):
                    r = obs[:, e]
                    self.s2[e] = max(np.mean(w[r] * Res[r, e] ** 2 + bvb[r, e]), 1e-5)
        return self

    def prior_coef(self, X, n):
        if self.models is None or X is None:
            return np.tile(self.g0, (n, 1))
        return np.column_stack([m.predict(X) for m in self.models])

    def predict(self, Y_obs, X=None, return_coef=False):
        """Y_obs: n x 14 with NaN for unobserved. Returns predictive mean and SD for all 14 elements."""
        Y_obs = np.asarray(Y_obs, float)
        G = self.prior_coef(X, len(Y_obs))
        M, V = self._posterior(Y_obs, G)
        C = G + M
        mean = C @ self.B.T + self.r0
        var = np.einsum("ek,nkl,el->ne", self.B, V, self.B) + self.s2
        if self.nu is not None:   # coherence-adaptive scale of the multivariate-t conditional
            var = var * (1.0 / self._weights())[:, None]
        self.coherence_d2, self.n_obs = self._d2.copy(), self._no.copy()
        if return_coef:
            return mean, np.sqrt(var), C, V
        return mean, np.sqrt(var)


def lambda_ols(Y_obs, degree=3, anomalies=("Ce", "Eu", "Gd")):
    """Per-sample least-squares lambda fit (no pooling, no prior): the conventional geochemical approach.
    Degree is reduced when too few non-anomalous elements are observed; anomaly elements are predicted
    on the smooth curve."""
    Y_obs = np.asarray(Y_obs, float)
    out = np.full_like(Y_obs, np.nan)
    smooth = np.array([e not in anomalies for e in REE])
    for i, y in enumerate(Y_obs):
        o = ~np.isnan(y)
        os_ = o & smooth
        use = os_ if os_.sum() >= 1 else o
        k = int(use.sum())
        if k == 0:
            continue
        d = min(degree, k - 1)
        if k <= 4:
            d = min(d, 1)
        B, _ = basis_matrix((), d)
        th = np.linalg.lstsq(B[use], y[use], rcond=None)[0]
        out[i] = B @ th
    return out
